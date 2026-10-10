# huohua

在 Apple M4（macOS / Apple clang 17）上复现论文
*《Adaptive FP16 GEMM on Apple M4 Pro SME》* 的矩阵乘法实现。

- 输入：A（N×M）、B（M×K），均为 **FP16**、行主序一维数组；输出 C（N×K）为 FP32。
- 索引约定：`A[i*M + j]`、`B[j*K + k]`、`C[i*K + k]`。
- 算法口径：当前落地的是论文 **FP16→FP32 路径**（16-bit 输入、FP32 累加与输出）；
  **FP16 与 BF16** 精度都已切换到论文的**加宽 FMOPA / BFMOPA kr=2 内核**
  （`src/assemble_f16.s` / `src/assemble_bf16.s`，每条指令覆盖相邻 2 个 k，打包按 lane-pair
  交错）；FP32 精度仍走 **FP32 FMOPA** 内核（`src/assemble_f32.s`）。
- 整体思路：C++ 侧做「分块 + 打包 + 调度」编排，真正的算术由 `src/assemble_*.s` 中的
  **手写 SME 汇编微内核**完成（Apple clang 无 `arm_sme2.h` 且 C++ 直接开 `+sme` 会触发
  SVE 自动向量化导致 SIGILL，因此内核必须用 `-march=armv9-a+sme2` 单独手写汇编）。

## 目录结构

```
huohua/
├── .gitignore
├── README.md
├── Makefile                       # 构建方式
├── orchestrator_pseudocode.md     # 自适应 orchestrator 设计伪代码（多数尚未实现）
├── data/
│   └── test.in                    # 基准 shape 列表，每行一个 "N M K"
├── src/
│   ├── gemm.h                     # 公共声明（gemm_fp16 / baseline_gemm / SME asm 入口）
│   ├── gemm.cpp                   # 主体函数部分
│   ├── assemble_f16.s             # SME FP16 加宽 FMOPA 32x32 微内核（kr=2 lane pair）
│   ├── assemble_bf16.s            # SME BF16 加宽 BFMOPA 32x32 微内核（kr=2 lane pair）
│   ├── assemble_f32.s             # SME FP32 32x32 微内核
│   ├── assemble_i8.s              # SME INT8→INT32 32x32 微内核
│   ├── assemble_f64.s             # SME FP64 16x16 微内核
│   └── bench.cpp                  # main：baseline 对比 + 正确性校验
└── tmp/                           # 探针与临时工具（见下方“临时探针”一节）
```

## 进度总览

| # | 优化点 | 状态 |
|---|--------|------|
| 1 | SME 汇编微内核底座（FP32 FMOPA） | ✅ 完成 |
| 2 | **FP16 / BF16 加宽 FMOPA 内核（kr=2 lane pair）** | ✅ 完成 |
| 3 | 软件流水 / SME2 多向量 load | ⚠️ 已实现，实测收益 ≈ 0 |
| 4 | L2 定向 prefetch（`pldl2keep`） | ⚠️ 已实现（仅 FP32 精度路径） |
| 5 | 并行 / 重叠 packing | ⚠️ 部分（仅 RHS 双缓冲） |
| 6 | half tile split-K 尾核 | ❌ 未做 |
| 7 | cost model / dispatcher 与 tiny-shape 回退 | ⚠️ 部分 |
| 8 | 双 P-cluster 常驻 worker | ❌ 不适用（本机单 P-cluster） |

**当前性能**：4096³ 在 FP16 与 BF16 上分别达 **1455 / 1475 GFLOP/s**，为本机单 P-cluster
指令级峰值（~2009）的 **72% / 73%**；已达本机**饱和微基准实测上限**（~1829）的约 80%，
扣除 packing 后内核本身约 **93%**。

**本机硬件**：Apple M4（`Mac16,10`，4P+6E，**单 P-cluster / 单 SME 单元**，非 M4 Pro）。
论文的双 cluster（4018 GFLOP/s）优化在本机不适用，评测分母统一取 ~2009 GFLOP/s。

### 三条必须知道的实测陷阱

1. **加宽 FMOPA / BFMOPA 的谓词粒度**：`fmopa` / `bfmopa za.s, …, z.h, z.h` 必须配 **`.h`
   谓词**（32 lane 全 1）。误用 `.s` 谓词时奇数 half-lane 被**静默屏蔽**，指令退化为 rank-1
   更新（1024→512 FLOP），结果“看似正确”却少一半累加项。
2. **SME2 多向量 load 的汇编语法**：只能写成**连字符区间 + `pn8` 谓词**，即
   `ld1h {z0.h-z1.h}, pn8/z, [x13]`；逗号形式 `{z0.h, z1.h}` 或配 `p0/z` 会被汇编器
   判为 invalid operand（易被误认为“汇编器不支持”）。
3. **`SMSTART/SMSTOP` 会清零 d8–d15**（Apple M4 实测）：内核必须自行在进出 streaming mode
   前后保存/恢复，否则破坏调用者的 callee-saved 寄存器，且在 -O3 下表现为诡异的 NaN/inf。

### 本次变更

- 新增 `src/assemble_f16.s` / `src/assemble_bf16.s`：FP16 / BF16 加宽 32×32 SME 微内核
  （kr=2 lane-pair + SME2 多向量 load），并各自提供 `huohua_sme_svl_bytes`。
- `src/gemm.cpp`：打包 `pack_a_panel`/`pack_b_band` 与 `scalar_microkernel` 统一为通用 kr
  布局（kr=1 与改造前逐字节一致），新增 FP16 与 BF16 两条分派分支。
- `src/gemm.h`：新增 `huohua_sme_microkernel_f16_32x32` / `_bf16_32x32` 声明。
- `Makefile`：half / bfloat 目标分别加 `-DHUOHUA_FP16_WIDEN` / `-DHUOHUA_BF16_WIDEN`，
  分别链接 `assemble_f16.o` / `assemble_bf16.o`。
- `tmp/`：新增多组 SME 语义探针与 `time_best` 计时工具（见“临时探针”一节）。

## 优化点进度

### 优化点 1：SME 汇编微内核底座（已完成 ✅）

目标：在 Apple M4 上跑通“FP16 输入 → SME FMOPA → FP32 累加/输出”的全链路，任意 shape 正确。

实现要点：

- **SME 汇编微内核**（`src/assemble_f32.s`）：4 个 ZA32 16×16 tile 以 2×2 覆盖 32×32 输出块，
  K 循环每步发 4 条 `fmopa` 外积累加；整个 tile 只进出一次 streaming mode
  （`SMSTART/SMSTOP`，摊薄 ~9ns 切换），最后用 `mov z.s, p0/m, zaNh.s[w12, 0]` 逐行读回 ZA 写 C。
- **Apple 汇编器 SME 语法适配**：实测发现其 `mova`/FP16 加宽助记符支持残缺，开发中以
  反汇编官方 `arm_sme.h` intrinsic 的方式锁定精确编码（如 ZA 读回需 `w12` 寄存器 +
  `, 0` 偏移、`ld1w` 偏移用 `#N, MUL VL`）。
- **FP16 输入通路**（论文全含口径起点）：`gemm_fp16`/`baseline_gemm` 均读 `__fp16`
  输入；bench 随机生成 FP32 后转 FP16 存储，贴近论文“FP16 行主序输入”。
- **k-major 打包 + FP16→FP32 加宽**：A 面板 / B 列带在打包时把 FP16 转成 k-major FP32
  布局，使微内核内层 K 循环顺序读流。（这是 FP32 路径的做法；**FP16 / BF16 已在优化点 2
  改为加宽 FMOPA / BFMOPA + lane-pair 打包，不再在打包阶段加宽**。）
- **L2 列带切分 + 行分块**：按 12 MiB L2 预算（论文公式 5）把 RHS 切成列带（对齐 32），
  N 维按 32 行 m-block 分块，使 B 列带驻留 L2 并在各 m-block 间复用。
- **标量回退**：不满足 32×32 的边缘 tile（`mr/nr<32`）与非 aarch64 平台走标量微内核，
  保证任意规模正确。
- **基准与正确性验证**（`bench.cpp`）：随机生成 FP16 数据，与朴素 ijk 的 `baseline_gemm`
  做容差比较，对齐输出各 shape 的 `N M K 正确性 baseline/optimized(GFLOP/s) speedup`。

验证：`data/test.in` 11 类 shape 全部 `correct=yes`；下面这组是本底座（FP32 FMOPA）的
实测，作为后续优化的对照基线，峰值约 1049 GFLOP/s（约单 P-cluster 峰值 ~2009 的 52%）。
优化点 2 落地后 FP16 路径已切到加宽内核，峰值升到 1455 GFLOP/s（见优化点 2）。

具体情况如下（FP32 FMOPA 内核，对照口径）
```plain
      N      M      K     correct        baseline       optimized   speedup
                                        (GFLOP/s)       (GFLOP/s)          
   1024   1024   1024         yes           2.678        1047.156   390.979
   1024   4096   1024         yes           2.524         871.017   345.032
   4096   4096   4096         yes           1.029        1049.110  1020.017
    512   4096   4096         yes           1.018         890.238   874.640
    512   4096  11008         yes           1.076         801.580   744.974
   2048   2048     64         yes           2.444         354.438   145.030
     32     32     32         yes           7.021          51.862     7.386
     64     64     64         yes           5.375         154.718    28.785
    100    100    100         yes           4.379         119.500    27.288
    128    128    128         yes           3.864         418.830   108.390
    256    256    256         yes           3.547         664.449   187.304
```

---

### 优化点 2：FP16 / BF16 加宽 FMOPA 内核（kr=2、lane pair）（已完成 ✅）

- 状态：✅ 已完成（`src/assemble_f16.s` / `src/assemble_bf16.s` + `gemm.cpp` 的 kr=2 打包路径）
- 覆盖精度：**FP16**（`fmopa za.s, …, z.h, z.h`，FEAT_SME_F16F32）与 **BF16**
  （`bfmopa za.s, …, z.h, z.h`，FEAT_SME_B16F32）。两者共用完全相同的 kr=2 lane-pair 布局与
  内核结构，唯一差别是助记符；BF16 此前是「打包时加宽成 FP32 再走 FP32 内核」。
- 内核：`fmopa za0.s, p1/m, p1/m, z0.h, z1.h`，一条指令覆盖相邻 2 个 k，完成 1024 FLOP，
  4 个 ZA32 tile 覆盖 32×32 输出；整个 tile 只进出一次 streaming mode。
- 打包（论文 4.2 `mr=nr=32, kr=2`）：A/B panel 内每行相邻 2 个 k 交错成 lane pair——
  向量 lane `2i` = 第 i 行 k0、lane `2i+1` = 第 i 行 k1；`lda=ldb=64`（halves）。
- **关键坑（已在 `assemble_f16.s` / `assemble_bf16.s` 注释留档）**：加宽 FMOPA / BFMOPA 的谓词按 16-bit 粒度求值，
  必须用 **`.h` 谓词**（32 lane 全 1）。若误用 `.s` 谓词，奇数 half-lane 被静默屏蔽，
  指令退化成 `ZA[i][j] = A[2i]·B[2j]` 的 rank-1 更新（只剩 512 FLOP），结果看似正确、
  实则少一半累加项。实测：`.s` 谓词下 `A=[1,1]·B=[1,1] → ZA[0][0]=1`；`.h` 谓词下
  为 `2`（正确的 2-way 点积 `A[2i]·B[2j]+A[2i+1]·B[2j+1]`）。
- 打包实现：`gemm.cpp` 的 `pack_a_panel`/`pack_b_band`/`scalar_microkernel` 统一为通用
  kr 布局——kr=1 与改造前逐字节一致，kr=2 即 lane pair，kr=4 即原 INT8 布局。
- 验证：`data/test.in` 全部 `correct=yes`（含奇数归约维边界 33 等），峰值 **1455 GFLOP/s**，
  约单 P-cluster 峰值的 **72%**（FP32 FMOPA 时为 52%）。分 shape 对比：

```plain
      N      M      K   FP32 FMOPA   FP16 加宽   提升
                        (GFLOP/s)    (GFLOP/s)
   1024   1024   1024      1047.2      1240.3    +18%
   1024   4096   1024       871.0      1336.4    +53%
   4096   4096   4096      1049.1      1454.5    +39%
    512   4096   4096       890.2      1224.1    +37%
    512   4096  11008       801.6      1249.7    +56%
   2048   2048     64       354.4       388.4    +10%
     32     32     32        51.9        51.8      0%
     64     64     64       154.7       165.6     +7%
    100    100    100       119.5       142.6    +19%
    128    128    128       418.8       439.6     +5%
    256    256    256       664.4       718.6     +8%
```

BF16 同口径 A/B（`time_best`，15 次取最优；旧 = 打包加宽 + FP32 内核，新 = BF16 加宽 BFMOPA）：

```plain
      N      M      K   旧(FP32 内核)  BF16 加宽   提升
                        (GFLOP/s)      (GFLOP/s)
   1024   1024   1024      968.8        1267.8    +31%
   1024   4096   1024      917.8        1244.1    +36%
   4096   4096   4096     1181.6        1474.8    +25%
    512   4096   4096      975.9        1164.8    +19%
    512   4096  11008      903.4        1281.3    +42%
   2048   2048     64      385.2         397.3     +3%
    256    256    256      684.8         723.5     +6%
    128    128    128      414.3         432.0     +4%
    100    100    100      118.5         144.6    +22%
    232    505    129      221.0         246.4    +11%
   4096    256     64      359.2         356.7     -1%
```

- 备注：本机为单 P-cluster 的 M4（非 M4 Pro），分母用 ~2009 GFLOP/s；论文的 4018 双
  cluster 峰值与「双 cluster」优化在本机不适用。

### 优化点 3：微内核软件流水 / 多向量 load

- 状态：⚠️ 已验证，收益为 0（K 循环已是 FMOPA 吞吐受限）
- **SME2/SVE2.1 多向量 load（已落地）**：`src/assemble_f16.s` 的 K 循环用
  `ld1h {z0.h-z1.h}, pn8/z, [x13]` 一次拉入 128 B（整组 A/B 面板），每组 load 从 4 条
  降到 2 条。语法要点：汇编器**只接受连字符区间 + `pn8` 谓词**形式（写成 `{z0.h, z1.h}`
  或配 `p0/z` 会报 invalid operand）；本机 streaming mode 内可执行。
- **软件流水（load-ahead）**：试过「双缓冲整个 group」与「2× 展开 + 双寄存器 bank（无 mov）」
  两种写法，**均无收益**——乱序窗口已把 load 延迟藏住。
- **多向量 load 亦无收益**：A/B 实测（time_best，15 次取最优）4096³ 1481 vs 1477，其余 shape
  在 ±5% 噪声内。根因是 K 循环每 group 4 条加宽 FMOPA = 8 cycle，**发射受限**，load 4→2 条
  不改变吞吐。保留它只为与论文 SME2 内核形态一致。
- 结论：FP16 内核已接近本机指令级上限（饱和微基准 ≈ 1829 GFLOP/s；4096³ 全含
  1455（time_best 1477）≈ 80%，扣掉 packing 后内核本身 ≈ 93%）。要再往上走，杠杆在
  **packing 开销**与 1829→2009 的指令级天花板，而非 K 循环访存。

### 优化点 4：L2 定向 prefetch（`pldl2keep`）

- 状态：⚠️ 已实现并完成初步评测，收益依赖 shape，尚未确认为普遍收益
- 实现：FP32 SME K 循环在剩余 reduction 数大于 32 时，对当前 packed RHS panel 中前瞻 32 个 reduction 的行发出一次 `prfm pldl2keep`；`w6` 边界判断跳过短 K 和尾部预取，不预取 A。代码见 `src/assemble_f32.s` 的 K 循环（约第 111–117 行）。
- **覆盖范围**：预取目前只在 `src/assemble_f32.s`，即仅 **FP32 精度路径**受益。FP16
  （`assemble_f16.s`）与 BF16（`assemble_bf16.s`）的 K 循环都没有预取；如需验证其收益，
  需把同样的 `prfm pldl2keep` 加进对应内核。
- 正确性记录：预取开/关版本均在 Apple M4 独立构建；一次确定性比较覆盖 `(N,M,K)=(32,1,32)、(32,32,32)、(32,33,32)、(32,64,64)、(32,65,96)、(64,1024,1024)`，均与 FP32 标量参考一致且输出有限。另一次 `32×64×64` checksum 对照一致。测试 harness 和原始日志未纳入仓库，以上为本地实验记录。
- 性能记录（Apple M4，共享机器，交替 A/B）：首轮长 K 测量每版本 15 个样本，`512×2048×2048` 中位数 4.673 ms（关）/4.473 ms（开），约快 4.3%；`512×4096×1024` 为 6.079/5.578 ms，约快 8.3%。后续长 K 复测每版本 35 个样本，对应中位数为 4.453/4.420 ms（约快 0.8%）和 5.806/5.578 ms（约快 3.9%）。短/中 K 复测的最后三轮中位数显示 `128×512×512` 约快 3.2%，`256×1024×1024` 约慢 7.5%。原始逐样本日志未保存到仓库，且不同轮次有波动。
- 结论：构建和上述正确性比较通过；性能收益依 shape 且跨轮波动，尚未证明稳定的端到端收益，也没有确定通用启用阈值。以上数据是实验记录，不应视为可复现的基准档案或普遍加速结论。

### 优化点 5：并行 / 重叠 packing

- 状态：⚠️ 部分实现
- 已实现：当存在多个 RHS 列带、M≥64 且估算 packing 时间不超过估算 matmul 时间的 0.85 倍时，后台线程双缓冲打包后续 RHS 列带，与当前列带的 SME 计算重叠；不满足条件时使用同步打包。
- 尚未实现：常驻 worker、A panel 并行打包、首个 RHS 列带与计算并行，以及论文所述的 NEON packing helper / 完整双 P-cluster 计算分工。
- 代码位置：`src/gemm.cpp` 中 `BandPacker`（约第 153–238 行）、overlap 判定（约第 299–303 行）、A panel 同步打包与列带调度（约第 315–390 行）。
- 注意：当前后台线程仅用于 RHS packing；每次 GEMM 调用创建并回收，不是常驻 SME 计算 worker。

### 优化点 6：half tile split-K 尾核

- 状态：❌ 未做
- 目标：16×32 / 32×16 / 16×16 尾核（对齐粒度降到 16），减少边缘标量回退；空闲 tile 转 split-K
  partial sum 保持 4 条依赖链。
- 备注：小 / 不规则 shape（100³、232×129×505）收益最大。

### 优化点 7：cost model / dispatcher 与 tiny-shape 回退

- 状态：⚠️ 部分实现
- 已实现：依据 L2 预算计算 RHS 列带宽（`compute_nc`）；另有 packing / compute 时间的粗略估算，用于决定是否启用 RHS 后台双缓冲 overlap。
- 尚未实现：论文完整的 `est_tmatmul` 调度模型及专门的 tiny-shape 单线程 dispatcher；当前 cost model 仅用于列带和 overlap 门控，不选择不同的 SME 计算策略。
- 代码位置：`src/gemm.cpp` 中估算常量（约第 28–35 行）、`compute_nc`（约第 74–99 行）、overlap 判定（约第 299–303 行）及统一 tile 遍历（约第 320–364 行）。

### 优化点 8：双 P-cluster 常驻 SME 计算 worker

- 状态：❌ 不适用（本机为单 P-cluster 的 M4，没有第二个高性能 SME 单元）
- 说明：论文的双 P-cluster worker 扩展针对双 P-cluster 配置。本机实测为 Apple M4（`Mac16,10`，4P+6E），**只有一个 P-cluster / 一个 SME 单元**，因此该优化在本机不适用，评测分母取单 cluster 峰值 ~2009 GFLOP/s。代码层面当前也没有双计算 worker；现有后台线程只执行 RHS packing，且每次 GEMM 调用创建并回收（见优化点 5）。

## 实测结果（本机，data/test.in，11 类 shape）

- `data/test.in` 当前包含 11 个 shape；benchmark 对比朴素 ijk baseline，baseline 每个 shape 计时一次，优化路径预热一次后测量三次并取平均（见 `src/bench.cpp`）。
- 这些数据只描述当前测试集及测量流程，不代表复现了论文完整的 shape 集、对照实现或评测方法；不同机器与运行负载也会影响结果。
- 最近一次记录中，全部 shape `correct=yes`；相对朴素 baseline 加速约 7×（32³）–1421×（4096³）
  （朴素 baseline 太慢，加速比数值主要说明口径，不代表与论文 baseline 的对比）。
- FP16 与 BF16 路径峰值均约 **1455 GFLOP/s**（4096³），约为本机单 P-cluster SME 峰值
  （~2009 GFLOP/s）的 **72%**（优化点 2 落地前 FP16 为 52%）。本机饱和微基准测得的指令级
  实用上限约 1829 GFLOP/s，故 4096³ 已达实用上限的约 80%（扣掉 packing 后内核本身约 93%）；
  优化点 3 的软件流水 / 多向量 load 经验证无进一步收益（见优化点 3）。

## 临时探针与工具（`tmp/`）

验证 SME 语义时新增/使用的主要探针（源码，不属于构建产物）：

| 文件 | 用途 |
|------|------|
| `f16_map.cpp` / `f16_map2.cpp` | 用基向量刻画加宽 FMOPA 的 lane 映射（`ZA[i][j] ← A[2i]·B[2j]`） |
| `f16_sem.cpp` | 证明「`.h` 谓词 vs `.s` 谓词」→ 2-way 点积 / 静默退化 |
| `multivec_probe.cpp` | 验证 `ld1h {z0.h-z1.h}, pn8/z` 在 streaming mode 可执行 |
| `peak_f16_load.cpp` | 加宽 FMOPA 饱和微基准（本机指令级上限 ≈ 1829 GFLOP/s） |
| `timing.cpp` | 低噪声 `time_best` 计时 harness（论文口径，绕过朴素 baseline） |
| `bf16_probe.cpp` | 验证 BF16 加宽 `bfmopa` 语义与 FP16 一致（BF16 路径已据此接入） |

运行示例：

```bash
clang++ -std=c++17 -O0 -march=armv9-a+sme2 tmp/multivec_probe.cpp -o /tmp/mv && /tmp/mv
clang++ -std=c++17 -O3 -march=native -pthread -Isrc \
  -DA_TYPE=__fp16 -DB_TYPE=__fp16 -DC_TYPE=float -DHUOHUA_FP16_WIDEN \
  -DPRECISION_NAME='"FP16"' tmp/timing.cpp \
  build/half_2_single/gemm.o build/half_2_single/assemble_f16.o -o /tmp/timing
/tmp/timing 15
```

## 后续工作

- **压缩 packing 开销**（优化点 5 完整化）：4096³ 全含时间中 packing 约占 13–20%，是当前
  距实用上限的主要缺口。
- 把 L2 定向 prefetch（优化点 4）接入 FP16 / BF16 内核——目前只在 `src/assemble_f32.s`，
  即只有 FP32 精度路径受益。
- half tile split-K 尾核（优化点 6）。
- INT8 已使用等价的加宽 `smopa za.s, …, z.b, z.b`（8→32）+ kr=4；FP32/FP64 无更窄输入的
  加宽形式，不适用本优化。

## 构建与运行

项目提供五种独立精度构建；各目标使用独立对象目录，类型宏在 C++ 编译阶段传入，避免不同精度错误复用对象：

```bash
make                          # 同时构建以下五个可执行文件
make bench_half_2_single      # FP16 × FP16 → FP32
make bench_bfloat_2_single    # BF16 × BF16 → FP32
make bench_single_2_single    # FP32 × FP32 → FP32
make bench_int8_2_int32       # INT8 × INT8 → INT32
make bench_double_2_double    # FP64 × FP64 → FP64

./bench_half_2_single [file]  # 可传入自定义 "N M K" 列表文件
./bench_bfloat_2_single [file]
./bench_single_2_single [file]
./bench_int8_2_int32 [file]
./bench_double_2_double [file]

make run_half_2_single        # 使用 data/test.in
make run_bfloat_2_single      # 使用 data/test.in
make run_single_2_single      # 使用 data/test.in
make run_int8_2_int32         # 使用 data/test.in
make run_double_2_double      # 使用 data/test.in
```

FP16 / BF16 路径分别生成 **kr=2 lane-pair 的 16-bit k-major 面板**，并使用 `src/assemble_f16.s`
/ `src/assemble_bf16.s` 中的**加宽 32×32 SME 微内核**（`fmopa` / `bfmopa za.s, ..., z.h, z.h`，
即论文的算力内核）。FP32 输入路径则生成 FP32 k-major 面板，使用 `src/assemble_f32.s` 中的
FP32 32×32 SME 微内核。INT8 路径使用 kr=4 的 INT8 packed panel 和 `src/assemble_i8.s` 中独立的 signed `smopa` 32×32 微内核，累加及输出为 INT32；benchmark 输入限制在 `[-8,8]`，正确性按逐元素精确相等判断。FP64 路径生成 FP64 k-major 面板，并使用 `src/assemble_f64.s` 中独立的 FP64 16×16 SME 微内核。FP16/BF16/FP32/INT8 内核使用 `-march=armv9-a+sme2`，FP64 内核额外使用 `+sme-f64f64`；C++ 使用 `-march=native`。