# huohua

在 Apple M4（macOS / Apple clang 17）上复现论文
*《Adaptive FP16 GEMM on Apple M4 Pro SME》* 的矩阵乘法实现。

- 输入：A（N×M）、B（M×K），均为 **FP16**、行主序一维数组；输出 C（N×K）为 FP32。
- 索引约定：`A[i*M + j]`、`B[j*K + k]`、`C[i*K + k]`。
- 算法口径：当前落地的是论文 **FP16→FP32 路径**（FP16 输入、FP32 累加与输出）；
  算术内核当前为 **FP32 FMOPA**（打包时把 FP16 加宽成 FP32 再算），论文的
  FP16 加宽 kr=2 内核尚未切换（见下方“优化点进度”中的优化点 2）。
- 整体思路：C++ 侧做「分块 + 打包 + 调度」编排，真正的算术由 `src/assemble.s` 中的
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
│   ├── assemble_f32.s             # SME FP32 32x32 微内核
│   ├── assemble_i8.s              # SME INT8→INT32 32x32 微内核
│   ├── assemble_f64.s             # SME FP64 16x16 微内核
│   └── bench.cpp                  # main：baseline 对比 + 正确性校验
└── tmp/                           # 临时文件，包含跟踪 SME 探针源码（*.cpp / *.s）
```

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
  布局，使微内核内层 K 循环顺序读流（当前加宽在打包阶段完成，非 FP16 加宽 FMOPA）。
- **L2 列带切分 + 行分块**：按 12 MiB L2 预算（论文公式 5）把 RHS 切成列带（对齐 32），
  N 维按 32 行 m-block 分块，使 B 列带驻留 L2 并在各 m-block 间复用。
- **标量回退**：不满足 32×32 的边缘 tile（`mr/nr<32`）与非 aarch64 平台走标量微内核，
  保证任意规模正确。
- **基准与正确性验证**（`bench.cpp`）：随机生成 FP16 数据，与朴素 ijk 的 `baseline_gemm`
  做容差比较，对齐输出各 shape 的 `N M K 正确性 baseline/optimized(GFLOP/s) speedup`。

验证：`data/test.in` 11 类 shape 全部 `correct=yes`，加速约 18×–166×，峰值约 487 GFLOP/s
（约为本机单 P-cluster SME 峰值 ~2009 GFLOP/s 的 24%）。

具体情况如下
```plain
      N      M      K     correct        baseline       optimized   speedup
                                        (GFLOP/s)       (GFLOP/s)          
   1024   1024   1024         yes           2.612         367.552   140.728
   1024   4096   1024         yes           2.495         124.027    49.703
   4096   4096   4096         yes           1.017         132.347   130.099
    512   4096   4096         yes           1.017         130.590   128.465
    512   4096  11008         yes           1.070         130.301   121.785
   2048   2048     64         yes           2.467         154.766    62.723
     32     32     32         yes           6.929          39.314     5.674
     64     64     64         yes           5.877         142.975    24.327
    100    100    100         yes           4.534          27.413     6.046
    128    128    128         yes           3.754         316.551    84.318
    256    256    256         yes           3.640         490.146   134.640
```

---

### 优化点 2：FP16 加宽 FMOPA 内核（kr=2、lane pair）

- 状态：❌ 未做
- 目标：内核从 FP32 FMOPA 切到 FP16 加宽 `fmopa za0.s, ..., zN.h, zM.h`，一条指令覆盖相邻 2 个 k
  （kr=2、lane pair），算力密度 2×（512→1024 FLOP/条）。
- 改动面：`src/assemble_f32.s`（K 步进、`ld1h` 装载）+ `gemm.cpp` 打包布局（k 交错）
- 验证口径：单 cluster 峰值应翻倍趋近 ~2009 GFLOP/s；正确性保持全绿。
- 备注：FP16 加宽 lane 映射已在本机探针确认（`ZA[i][j] ← A_h[2i]·B_h[2j]`）。

### 优化点 3：微内核软件流水 / 多块展开

- 状态：❌ 未做
- 目标：K 循环内提前装载下一组 A/B、多 ZA tile 轮转，隐藏 load 延迟，把利用率从 ~24% 拉向峰值。
- 改动面：`src/assemble_f32.s`（内层 K 循环展开/流水）
- 验证口径：compute-bound shape 利用率显著上升；正确性保持全绿。

### 优化点 4：L2 定向 prefetch（`pldl2keep`）

- 状态：❌ 未做
- 目标：只对冷流 RHS 做 L2 预取（`locality=2`、前瞻若干向量），decode / 大 K shape 收益。
- 改动面：`src/assemble_f32.s` 或编排层预取；LHS 跨 N 复用不预取。

### 优化点 5：并行 / 重叠 packing

- 状态：⚠️ 部分实现
- 已实现：当存在多个 RHS 列带、M≥64 且估算 packing 时间不超过估算 matmul 时间的 0.85 倍时，后台线程双缓冲打包后续 RHS 列带，与当前列带的 SME 计算重叠；不满足条件时使用同步打包。
- 尚未实现：常驻 worker、A panel 并行打包、首个 RHS 列带与计算并行，以及论文所述的 NEON packing helper / 完整双 P-cluster 计算分工。
- 代码位置：`src/gemm.cpp` 中 `BandPacker`（约第 140–229 行）、overlap 判定（约第 307–313 行）、A panel 同步打包与列带调度（约第 329–385 行）。
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
- 代码位置：`src/gemm.cpp` 中估算常量与模型（约第 21–43 行）、`compute_nc`（约第 65–80 行）、overlap 判定（约第 307–313 行）及统一 tile 遍历（约第 329–369 行）。

### 优化点 8：双 P-cluster 常驻 SME 计算 worker

- 状态：❌ 未实现（当前实现由调用线程执行 SME 计算；硬件适用性取决于具体 M4 型号）
- 说明：论文中的双 P-cluster worker 扩展针对双 P-cluster 配置。当前代码没有双计算 worker；现有后台线程只执行 RHS packing，且每次 GEMM 调用创建并回收（见优化点 5）。本机是否具备双 P-cluster 属于硬件信息，不能仅凭代码判断。

## 实测结果（本机，data/test.in，11 类 shape）

- `data/test.in` 当前包含 11 个 shape；benchmark 对比朴素 ijk baseline，baseline 每个 shape 计时一次，优化路径预热一次后测量三次并取平均（见 `src/bench.cpp`）。
- 这些数据只描述当前测试集及测量流程，不代表复现了论文完整的 shape 集、对照实现或评测方法；不同机器与运行负载也会影响结果。
- 最近一次记录中，全部 shape `correct=yes`；相对朴素 baseline 加速约 18×–166×。
- 峰值约 487 GFLOP/s，约为本机单 P-cluster SME FP32/FP16 峰值（~2009 GFLOP/s）的 24%。
  （优化点 2/3 落地后此项应显著上升。）

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

FP16、BF16 与 FP32 输入路径都会在 packing 阶段生成 FP32 k-major 面板，并共享 `src/assemble_f32.s` 中的 FP32 32×32 SME 微内核；BF16 在 packing 时精确扩展为 FP32，并未使用原生 `bfmopa`。INT8 路径使用 kr=4 的 INT8 packed panel 和 `src/assemble_i8.s` 中独立的 signed `smopa` 32×32 微内核，累加及输出为 INT32；benchmark 输入限制在 `[-8,8]`，正确性按逐元素精确相等判断。FP64 路径生成 FP64 k-major 面板，并使用 `src/assemble_f64.s` 中独立的 FP64 16×16 SME 微内核。FP32/INT8 内核使用 `-march=armv9-a+sme2`，FP64 内核额外使用 `+sme-f64f64`；C++ 使用 `-march=native`。
