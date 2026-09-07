# 自适应 SME Orchestrator 伪代码

> 对应 `plan.md` 阶段三（第 4 章 Design）的 9 项优化。
> 核心思想：一层"调度 + 流水编排"，真正的算术内核（KleidiAI SME2 micro-kernel）不动。

```text
// ============ 顶层：显式 cost model dispatcher ============
function gemm_fp16(M, N, K, A, B, C, out_dtype, ctx):
    # 全含口径：每次调用都做 packing（不缓存 packed operand）
    if M < 64 and est_tmatmul(M,N,K) < SYNC_COST * FALLBACK_RATIO:   # tiny-shape 回退(论文未实现)
        run_single_thread_path(M, N, K, A, B, C, out_dtype)
        return

    Nc = compute_Nc(N, K)                     # 公式(5)：12MiB L2 列带，对齐 32
    n_bands = ceil(N / Nc)

    # 0) 并行打包 LHS 与第 0 个 RHS 列带
    work = { lhs: (A_split[0], A_split[1]),   # LHS 沿 M 切两段，各对齐32
             rhs: band0 }
    both_workers_submit(ctx, work)            # worker0/worker1 并行 pack

    for i in 0 .. n_bands-1:
        band = B[:, i*Nc : (i+1)*Nc]
        if i < n_bands-1:
            # 1) 空闲 P 核上的 NEON helper 后台打包后续列带（与 SME 计算重叠）
            if should_overlap(M, Nc, n_bands, band_bytes):
                spawn_neon_pack_async(band_{i+1})     # ~23 GB/s, 藏进计算
            else:
                both_workers_pack(band_{i+1})         # 回退：纯 SME 并行 pack

        # 2) 等待本次要算的列带打包就绪（第0带在第0步已并行打好）
        wait_rhs_ready(i)

        # 3) 两个 cluster 各算半块（M 或 N 维对半），RHS 列带留在 L2 供 m-block 复用
        dual_cluster_matmul(M, band, A, C[:, band_cols], out_dtype)
    # 与 async NEON helper 同步收尾
    join_neon_helpers()
```

```text
// ============ 双 cluster 常驻 worker（atomic generation counter）============
init:
    worker1 = spawn_resident_spin_worker()   # 常驻，绝不短命重建
    gen = atomic(0);  tasks = ring_buffer[2]

submit(both_workers, job):                    # 主线程与 worker1 各拿一份
    for w in {main, worker1}: tasks[w] = job
    gen.fetch_add(1)                          # 通知
    worker1: while gen < target: pause()      # 自旋等待

spawn_resident_spin_worker():                 # 关键：短命 pthread 会扎堆同一 cluster
    loop:
        g = gen.load(acquire)
        if g != my_last_seen and tasks[me].ready:
            do_pack_or_matmul(tasks[me])
            my_last_seen = g

// 切分策略：M>=64 按 32 行 block 对半切 M；M<64 按 32 列 block 对半切 N
function split_for_two_clusters(M, N, band_N):
    if M >= 64:
        return { w0: {M:0..M/2},  w1: {M:M/2..M} }   # 各按 32 对齐
    else:
        return { w0: {N:0..N/2},  w1: {N:N/2..N} }   # 各按 32 对齐(列带内)
```

```text
// ============ L2 列带宽度（公式5）与 packing 生命周期 ============
function compute_Nc(N, K):
    B_L2  = 12 MiB
    sB    = 2          # FP16 bytes
    Nc    = max(32, floor(B_L2 / (32 * K * sB)))
    return min(N, align_down(Nc, 32))    # 使 band 首次进 L2 后被各 m-block 复用

// KAI packing（复用，不改动）：mr=nr=32, kr=2，相邻2个k交错成 lane pair，RHS 每32列前置 bias
function pack_lhs_tile(A_blk) -> packed        # k-major panel，逐字节等价
function pack_rhs_panel(B_blk) -> packed       # 32 列一 panel, 前置 bias
```

```text
// ============ 单 worker matmul：SME2 FP16→FP16 macro-kernel ============
function sme_macro_kernel(Ap, Bp, Cp, M, N, K, use_prefetch):
    smstart()                                   # 整个 macro-kernel 只切换一次 (~9.4ns 摊销)
    # 2x2 排列的 4 个 ZA32 tile: ZA0..ZA3 覆盖 32x32 输出块
    for m0 in 0..M step 32:                     # m-block
      for n0 in 0..N step 32:                   # n-block
        za_clear(ZA0..ZA3)
        # bias 播种用外积：b0 前置 bias 行的外积初始化，不做额外清零循环
        for k0 in 0..K step kr:                 # kr=2
            a0,a1 = ld4(quad) of Ap[m0, k0]     # SME2: 4条四向量load 替代16条
            b0,b1 = ld4(quad) of Bp[n0, k0]
            if use_prefetch:
                prefetch_l2_next_colband()      # pldl2keep 只用于冷流 RHS
            fma za0 += a0 ⊗ b0                  # 四条加宽 FMOPA, 覆盖相邻2k
            fma za1 += a0 ⊗ b1
            fma za2 += a1 ⊗ b0
            fma za3 += a1 ⊗ b1
        # epilogue: FP32→FP16 窄化 (SME2 变体, 短链)
        store_za_to_c(Cp[m0,n0], za0..za3, narrow_to_fp16, M,N 余量谓词)
    smstop()

// FP16→FP32：同样结构，差异只在 epilogue
function sme_macro_kernel_f32(...):
    ...
    for k0: 四 tile 加宽 FMOPA               # 每 macro-kernel 仍只进一次 streaming
    svst1_hor_za32(pred, Cp[m0,n0], za0..za3)  # ZA 水平谓词化直写 FP32（不经窄化）
```

```text
// ============ prefetch：只进 L2，只预取冷流 RHS ============
function prefetch_l2_next_colband(addr):
    # __builtin_prefetch(addr) → pldl1keep(L1), 实测降 14%
    # 改用 locality=2 → pldl2keep; 只预取 RHS 下一个列带(冷), 前瞻 64 个 FP16 向量
    for v in 0..63:
        __builtin_prefetch(addr + v*64, 0 /*read*/, 2 /*L2*/)
    # LHS 跨 N 复用 → 不预取
```

```text
// ============ 并行 + 重叠 packing 的具体时序 ============
// v2 流水三阶段：pack0 ∥ (pack(i+1) 重叠 compute_i)
function dual_cluster_pipeline(...):
    # 决策表
    if n_bands < 2 or M < 64 or est_neon_pack_time > 0.85 * est_sme_time:
        overlap = FALSE                    # NEON packing 太慢，回退纯 SME 并行 packing
    # NEON pack ~23 GB/s vs SME pack ~67 GB/s；只要藏在 compute 后面就不上关键路径

    # 验证等价性：pack 结果写入毒化缓冲 + 逐字节比对；SME1/SME2 输出逐位一致
```

```text
// ============ half tile split-K 尾核（未并入主驱动，单线程评测用）============
function tail_kernel_16x32(Ap, Bp, Cp, m=16, n=32, k, nsplit):
    # 尾长对齐粒度降到 ZA32 物理下界 16；不改 packing 布局
    # 少 tile 会有流水线气泡 → 空闲 tile 转 split-K partial sum
    za_partial[0..nsplit-1];  za_other[0..] 用于几何 block
    for each 几何位置 p:                       # 16x32 → tile0=tile1 几何分块
        for s in 0..nsplit-1:                  # split K，保持 4 条依赖链轮转
            fma za_partial[s] += a_s ⊗ b_s     # nacc = latency/issue = 4
    C = sum(za_partial)                         # epilogue 合并 partial sums
    # 分派: 16×32, 32×16, 16×16; 尾长 1–16 走本核, 17–31 回退到 32 补齐
```

```text
// ============ 主循环里真正"等对齐 + 等就绪"的那套 ============
// N、M 余量 < 32 时:
//   余 1–16 → half tile split-K 尾核 (16x32/32x16/16x16)
//   余 17–31 → 暂回退 32 补齐 (论文如此, 未做 16 之外粒度)
```

## 9 项优化对照表

| # | 优化 | 对应 plan.md | 伪代码块 |
|---|------|-------------|---------|
| 1 | SME2 变体接入 | 4.1/4.2 | sme_macro_kernel |
| 2 | 12 MiB N 维列带 | 4.2/4.3 | compute_Nc / gemm_fp16 |
| 3 | 双 cluster 常驻 worker | 4.3/4.4 | spawn_resident_spin_worker / split_for_two_clusters |
| 4 | 并行 packing | 4.4/4.5 | both_workers_submit |
| 5 | 混合 v2 流水重叠 | 4.4/4.5 | dual_cluster_pipeline |
| 6 | FP16→FP32 直存内核 | 4.5/4.6 | sme_macro_kernel_f32 |
| 7 | L2 定向 prefetch | 4.5/4.6 | prefetch_l2_next_colband |
| 8 | half tile split-K 尾核 | 4.6/4.7 | tail_kernel_16x32 |
| 9 | 显式 cost model / dispatcher | 4.1 | gemm_fp16 |
