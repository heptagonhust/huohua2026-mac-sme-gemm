#include "gemm.h"

#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <thread>

namespace {

// One SME output micro-tile uses four ZA tiles: 32x32 for FP32 and 16x16
// for FP64 at the 512-bit streaming vector length used by Apple M4.
constexpr bool kUseFp64Kernel = std::is_same<C_TYPE, double>::value;
constexpr bool kUseInt8Kernel = std::is_same<C_TYPE, int32_t>::value;
constexpr std::size_t kTile = kUseFp64Kernel ? 16 : 32;
constexpr std::size_t kReduction = kUseInt8Kernel ? 4 : 1;
using PackType = std::conditional_t<kUseInt8Kernel, int8_t, C_TYPE>;
// L2 budget for one RHS column band (paper formula 5).
constexpr std::size_t kL2Bytes = 12u * 1024u * 1024u;

// Hand-calibrated overlap economics (paper 5.7): the packing side moves data
// at roughly 20-23 GB/s while the SME side retires roughly 3.5 TFLOP/s on the
// shapes where overlap pays off.  These constants only gate whether spawning
// the double-buffered pipeline is worth it at all.
constexpr double kPackBytesPerSecond = 20.0e9;
constexpr double kComputeFlopsPerSecond = 3.5e12;

// Width of the RHS column band that starts at output column `col0`.
std::size_t band_ncols(std::size_t col0, std::size_t Nc, std::size_t K) {
    return (col0 + Nc <= K) ? Nc : (K - col0);
}

// Time to pack one band: read the configured input and write the packed panel.
double est_pack_seconds(std::size_t M, std::size_t ncols) {
    constexpr double kBytesPerElement = sizeof(B_TYPE) + sizeof(PackType);
    return static_cast<double>(M) * static_cast<double>(ncols) *
           kBytesPerElement / kPackBytesPerSecond;
}

// Time for the matrix engines to consume one band: 2*N*M*ncols FLOP.
double est_compute_seconds(std::size_t N, std::size_t M, std::size_t ncols) {
    return 2.0 * static_cast<double>(N) * static_cast<double>(M) *
           static_cast<double>(ncols) / kComputeFlopsPerSecond;
}

std::size_t align_down(std::size_t value, std::size_t alignment) {
    return value - (value % alignment);
}

// Column-band width along the output-column dimension (paper formula 5).
//
// The RHS is M x K in this project's naming, so the banded side is K and the
// inner (reused) side is M. One packed RHS column occupies M C_TYPE elements,
// regardless of the configured source type, so this returns the widest
// tile-aligned band that still fits the L2 budget.
//
// Wider is always better here: the number of bands sets how many times the
// LHS panels are re-packed (n_bands * N * M elements) and how often the RHS
// is streamed in, while the RHS byte volume itself (M * K) is unchanged.
// Hence the largest L2-resident band minimises total packing traffic.
//
// N (the number of output rows) deliberately does not enter the formula: the
// band is reloaded once per band no matter how many row panels consume it, so
// widening it up to the L2 budget is the right trade for every N.
std::size_t compute_nc(std::size_t N, std::size_t M, std::size_t K) {
    (void)N;
    if (K <= kTile) {
        return K;
    }
    // Bytes of a single packed RHS column, including reduction padding.
    const std::size_t groups = (M + kReduction - 1) / kReduction;
    const std::size_t col_bytes = groups * kReduction * sizeof(PackType);
    std::size_t cols = align_down(kL2Bytes / col_bytes, kTile);
    if (cols < kTile) {
        cols = kTile;
    }
    if (cols > K) {
        cols = K;
    }
    return cols;
}

// Pack one A row panel (mr rows x full inner M) into a k-major panel.
// The configured input type is converted to the configured output type.
// A_panel[k * lda + i] = (C_TYPE)A[(row0 + i) * M + k], lda == kTile.
void pack_a_panel(const A_TYPE* A, PackType* A_panel, std::size_t row0,
                  std::size_t mr, std::size_t M, std::size_t lda) {
#if defined(HUOHUA_INT8)
    const std::size_t groups = (M + 3) / 4;
    for (std::size_t group = 0; group < groups; ++group) {
        PackType* dst = A_panel + group * lda;
        for (std::size_t i = 0; i < mr; ++i) {
            for (std::size_t lane = 0; lane < 4; ++lane) {
                const std::size_t k = group * 4 + lane;
                dst[i * 4 + lane] = k < M ? A[(row0 + i) * M + k] : 0;
            }
        }
    }
#else
    for (std::size_t k = 0; k < M; ++k) {
        #pragma unroll(8)
        for (std::size_t i = 0; i < mr; ++i) {
            A_panel[k * lda + i] = static_cast<C_TYPE>(A[(row0 + i) * M + k]);
        }
    }
#endif
}

// Pack one RHS column band into a k-major panel.
// B_panel[k * ldb + j] = (C_TYPE)B[k * K + (col0 + j)].
//
// The body is deliberately unchanged (paper 4.2: packing primitives are
// reused as-is); double buffering lives in the orchestrator below, which just
// points this routine at one of two ping-pong slots.
void pack_b_band(const B_TYPE* B, PackType* B_panel, std::size_t col0,
                 std::size_t ncols, std::size_t M, std::size_t K,
                 std::size_t ldb) {
#if defined(HUOHUA_INT8)
    const std::size_t groups = (M + 3) / 4;
    for (std::size_t group = 0; group < groups; ++group) {
        PackType* dst = B_panel + group * ldb;
        for (std::size_t j = 0; j < ncols; ++j) {
            for (std::size_t lane = 0; lane < 4; ++lane) {
                const std::size_t k = group * 4 + lane;
                dst[j * 4 + lane] = k < M ? B[k * K + col0 + j] : 0;
            }
        }
    }
#else
    for (std::size_t k = 0; k < M; ++k) {
        const B_TYPE* b_row = B + k * K + col0;
        #pragma unroll(8)
        for (std::size_t j = 0; j < ncols; ++j) {
            B_panel[k * ldb + j] = static_cast<C_TYPE>(b_row[j]);
        }
    }
#endif
}

// Double-buffered RHS column-band producer (paper 4.5).
//
// The orchestrator computes band i out of slot_[i & 1] while this background
// thread packs band i+1 into slot_[(i + 1) & 1].  Strict in-order production
// and consumption with only two slots means the producer may start band i
// once band i-2 has been released (band <= consumed_ + 1): that is exactly the
// guarantee that it never overwrites the slot being read.
//
// The lifetime is one gemm call. A resident worker (paper 4.4) would
// avoid the per-call spawn, but that is a separate optimisation.
class BandPacker {
public:
    BandPacker(const B_TYPE* B, std::size_t M, std::size_t K, std::size_t Nc,
               std::size_t n_bands, PackType* slot0, PackType* slot1)
        : B_(B), M_(M), K_(K), Nc_(Nc), n_bands_(n_bands) {
        slot_[0] = slot0;
        slot_[1] = slot1;
    }

    BandPacker(const BandPacker&) = delete;
    BandPacker& operator=(const BandPacker&) = delete;

    ~BandPacker() { join(); }

    void start() { worker_ = std::thread([this] { run(); }); }

    // Blocks until band `band` is packed, then hands back its slot.
    const PackType* acquire(std::size_t band) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] { return produced_ > band; });
        return slot_[band & 1u];
    }

    // Marks band `band` consumed so its slot may be refilled.
    void release(std::size_t band) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            consumed_ = band + 1;
        }
        ready_.notify_all();
    }

    void join() {
        if (!worker_.joinable()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        ready_.notify_all();
        worker_.join();
    }

private:
    void run() {
        for (std::size_t band = 0; band < n_bands_; ++band) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock,
                            [&] { return stopped_ || band <= consumed_ + 1; });
                if (stopped_) {
                    return;
                }
            }

            const std::size_t col0 = band * Nc_;
            pack_b_band(B_, slot_[band & 1u], col0, band_ncols(col0, Nc_, K_),
                        M_, K_, Nc_ * kReduction);

            {
                std::lock_guard<std::mutex> lock(mutex_);
                produced_ = band + 1;
            }
            ready_.notify_all();
        }
    }

    const B_TYPE* B_;
    std::size_t M_;
    std::size_t K_;
    std::size_t Nc_;
    std::size_t n_bands_;
    PackType* slot_[2];
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::size_t produced_ = 0;  // bands fully packed
    std::size_t consumed_ = 0;  // bands released by the consumer
    bool stopped_ = false;
};

// Scalar fallback for edge tiles and for unsupported streaming vector length.
//
// It has the same contract as the SME kernel: the mr x nr tile is
// OVERWRITTEN, not accumulated into.  C may therefore hold garbage (gemm's
// caller only promises the buffer exists) -- zeroing the tile first keeps the
// two paths interchangeable.
void scalar_microkernel(const PackType* A_panel, int lda,
                        const PackType* B_panel, int ldb,
                        C_TYPE* C, int ldc,
                        int mr, int nr, int kc) {
    for (int i = 0; i < mr; ++i) {
        C_TYPE* crow = C + static_cast<std::size_t>(i) * ldc;
        for (int j = 0; j < nr; ++j) {
            crow[j] = 0;
        }
    }
#if defined(HUOHUA_INT8)
    for (int group = 0; group < kc; ++group) {
        const PackType* ap = A_panel + static_cast<std::size_t>(group) * lda;
        const PackType* bp = B_panel + static_cast<std::size_t>(group) * ldb;
        for (int i = 0; i < mr; ++i) {
            C_TYPE* crow = C + static_cast<std::size_t>(i) * ldc;
            for (int j = 0; j < nr; ++j) {
                for (int lane = 0; lane < 4; ++lane) {
                    crow[j] += static_cast<C_TYPE>(ap[i * 4 + lane]) *
                               static_cast<C_TYPE>(bp[j * 4 + lane]);
                }
            }
        }
    }
#else
    for (int k = 0; k < kc; ++k) {
        const PackType* ap = A_panel + static_cast<std::size_t>(k) * lda;
        const PackType* bp = B_panel + static_cast<std::size_t>(k) * ldb;
        for (int i = 0; i < mr; ++i) {
            const C_TYPE a = ap[i];
            C_TYPE* crow = C + static_cast<std::size_t>(i) * ldc;
            for (int j = 0; j < nr; ++j) {
                crow[j] += a * bp[j];
            }
        }
    }
#endif
}

}  // namespace

// ---------------------------------------------------------------------------
// C orchestrator: L2 column bands + row panels + k-major packing, with the
// RHS band double-buffered so that packing overlaps SME compute.
// A: N x M, B: M x K, C: N x K using the configured precision.
// Full tiles use the FP32 32x32 or FP64 16x16 SME micro-kernel; edge tiles
// and machines with an unsupported vector length use scalar_microkernel.
// ---------------------------------------------------------------------------
void gemm(const A_TYPE* A, const B_TYPE* B, C_TYPE* C,
               std::size_t N, std::size_t M, std::size_t K) {
    if (N == 0 || M == 0 || K == 0) {
        return;
    }

    const std::size_t reductions = (M + kReduction - 1) / kReduction;
    const std::size_t Nc = compute_nc(N, M, K);
    const std::size_t lda = kTile * kReduction;
    const std::size_t ldb = Nc * kReduction;
    const int ldc = static_cast<int>(K);
    const std::size_t n_bands = (K + Nc - 1) / Nc;

#if defined(__aarch64__)
    // Both kernels are hard-wired to a 512-bit streaming vector length.
    const bool sme_ok = huohua_sme_svl_bytes() == 64u;
#else
    const bool sme_ok = false;
    (void)sme_ok;
#endif

    // Double-buffered band pipeline (paper 4.5).  Overlapping only pays when
    // there are at least two bands to ping-pong, M is fat enough to amortise
    // the extra thread, and packing is shorter than 0.85x the SME compute it
    // has to hide behind.  Otherwise the synchronous path runs unchanged.
    const bool overlap =
        n_bands >= 2 && M >= 64 &&
        est_pack_seconds(M, Nc) <= 0.85 * est_compute_seconds(N, M, Nc);

    // Buffers: one A panel (kTile x M) plus one or two B bands (M x Nc).
    const std::size_t band_elems = reductions * ldb;
    const std::size_t n_slots = overlap ? 2u : 1u;
    PackType* A_panel = static_cast<PackType*>(
        std::malloc(reductions * lda * sizeof(PackType)));
    PackType* band_mem = static_cast<PackType*>(
        std::malloc(n_slots * band_elems * sizeof(PackType)));
    if (A_panel == nullptr || band_mem == nullptr) {
        std::free(A_panel);
        std::free(band_mem);
        return;
    }
    PackType* slot[2] = {band_mem, overlap ? band_mem + band_elems : nullptr};

    // Consume one packed band: every A row panel against every tile-wide slice.
    auto compute_band = [&](const PackType* band, std::size_t col0,
                            std::size_t ncols) {
        for (std::size_t row0 = 0; row0 < N; row0 += kTile) {
            const std::size_t mr = (row0 + kTile <= N) ? kTile : (N - row0);
            pack_a_panel(A, A_panel, row0, mr, M, lda);

            for (std::size_t j0 = 0; j0 < ncols; j0 += kTile) {
                const std::size_t nr = (j0 + kTile <= ncols) ? kTile : (ncols - j0);
                C_TYPE* ctile = C + row0 * K + col0 + j0;

#if defined(__aarch64__)
                if (sme_ok && mr == kTile && nr == kTile) {
#if defined(HUOHUA_INT8)
                    huohua_sme_microkernel_i8_32x32(
                        A_panel, static_cast<int>(lda),
                        band + j0 * kReduction, static_cast<int>(ldb),
                        ctile, ldc, static_cast<int>(reductions));
#elif defined(HUOHUA_FP64)
                    huohua_sme_microkernel_f64_16x16(
                        A_panel, static_cast<int>(lda),
                        band + j0, static_cast<int>(ldb),
                        ctile, ldc, static_cast<int>(M));
#else
                    huohua_sme_microkernel_f32_32x32(
                        A_panel, static_cast<int>(lda),
                        band + j0, static_cast<int>(ldb),
                        ctile, ldc, static_cast<int>(M));
#endif
                } else
#endif
                {
                    scalar_microkernel(A_panel, static_cast<int>(lda),
                                       band + j0 * kReduction,
                                       static_cast<int>(ldb),
                                       ctile, ldc,
                                       static_cast<int>(mr), static_cast<int>(nr),
                                       static_cast<int>(reductions));
                }
            }
        }
    };

    if (overlap) {
        // The producer fills band i+1 into the other slot while this thread
        // computes band i out of slot[i & 1]; acquire/release keep the slots
        // strictly ping-ponged.  join() before returning, so no work escapes
        // the call.
        BandPacker packer(B, M, K, Nc, n_bands, slot[0], slot[1]);
        packer.start();
        for (std::size_t i = 0; i < n_bands; ++i) {
            const std::size_t col0 = i * Nc;
            const PackType* band = packer.acquire(i);
            compute_band(band, col0, band_ncols(col0, Nc, K));
            packer.release(i);
        }
        packer.join();
    } else {
        for (std::size_t col0 = 0; col0 < K; col0 += Nc) {
            const std::size_t ncols = band_ncols(col0, Nc, K);
            pack_b_band(B, slot[0], col0, ncols, M, K, ldb);
            compute_band(slot[0], col0, ncols);
        }
    }

    std::free(A_panel);
    std::free(band_mem);
}

void gemm_fp16(const A_TYPE* A, const B_TYPE* B, C_TYPE* C,
               std::size_t N, std::size_t M, std::size_t K) {
    gemm(A, B, C, N, M, K);
}
