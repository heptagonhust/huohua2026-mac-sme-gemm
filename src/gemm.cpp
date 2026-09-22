#include "gemm.h"

#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <thread>

namespace {

// One SME output micro-tile is 32 x 32 (four 16x16 ZA32 tiles).
constexpr std::size_t kTile = 32;
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

// Time to pack one band: read the configured input and write the FP32 panel.
double est_pack_seconds(std::size_t M, std::size_t ncols) {
    constexpr double kBytesPerElement = sizeof(B_TYPE) + sizeof(C_TYPE);
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
// inner (reused) side is M. One packed RHS column occupies M FP32 elements,
// regardless of the configured source type, so this returns the widest
// 32-aligned band that still fits the L2 budget.
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
    // Bytes of a single packed RHS column.
    const std::size_t col_bytes = (M > 0 ? M : 1) * sizeof(C_TYPE);
    std::size_t cols = align_down(kL2Bytes / col_bytes, kTile);
    if (cols < kTile) {
        cols = kTile;
    }
    if (cols > K) {
        cols = K;
    }
    return cols;
}

// Pack one A row panel (mr rows x full inner M) into a k-major FP32 panel.
// The configured input type is converted to FP32 for the current kernel.
// A_panel[k * lda + i] = (float)A[(row0 + i) * M + k], lda == kTile.
void pack_a_panel(const A_TYPE* A, C_TYPE* A_panel, std::size_t row0,
                  std::size_t mr, std::size_t M, std::size_t lda) {
    for (std::size_t k = 0; k < M; ++k) {
        #pragma unroll(8)
        for (std::size_t i = 0; i < mr; ++i) {
            A_panel[k * lda + i] = static_cast<C_TYPE>(A[(row0 + i) * M + k]);
        }
    }
}

// Pack one RHS column band (full inner M x ncols columns) into k-major FP32.
// B_panel[k * ldb + j] = (float)B[k * K + (col0 + j)].
//
// The body is deliberately unchanged (paper 4.2: packing primitives are
// reused as-is); double buffering lives in the orchestrator below, which just
// points this routine at one of two ping-pong slots.
void pack_b_band(const B_TYPE* B, C_TYPE* B_panel, std::size_t col0,
                 std::size_t ncols, std::size_t M, std::size_t K,
                 std::size_t ldb) {
    for (std::size_t k = 0; k < M; ++k) {
        const B_TYPE* b_row = B + k * K + col0;
        #pragma unroll(8)
        for (std::size_t j = 0; j < ncols; ++j) {
            B_panel[k * ldb + j] = static_cast<C_TYPE>(b_row[j]);
        }
    }
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
               std::size_t n_bands, C_TYPE* slot0, C_TYPE* slot1)
        : B_(B), M_(M), K_(K), Nc_(Nc), n_bands_(n_bands) {
        slot_[0] = slot0;
        slot_[1] = slot1;
    }

    BandPacker(const BandPacker&) = delete;
    BandPacker& operator=(const BandPacker&) = delete;

    ~BandPacker() { join(); }

    void start() { worker_ = std::thread([this] { run(); }); }

    // Blocks until band `band` is packed, then hands back its slot.
    const C_TYPE* acquire(std::size_t band) {
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
                        M_, K_, Nc_);

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
    C_TYPE* slot_[2];
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::size_t produced_ = 0;  // bands fully packed
    std::size_t consumed_ = 0;  // bands released by the consumer
    bool stopped_ = false;
};

// Scalar fallback for the edge tiles (mr<32 or nr<32), and for builds where
// the SME asm kernel is missing or unusable (SVL != 512 bit).
//
// It has the same contract as the SME kernel: the mr x nr tile is
// OVERWRITTEN, not accumulated into.  C may therefore hold garbage (gemm's
// caller only promises the buffer exists) -- zeroing the tile first keeps the
// two paths interchangeable.
void scalar_microkernel(const C_TYPE* A_panel, int lda,
                        const C_TYPE* B_panel, int ldb,
                        C_TYPE* C, int ldc,
                        int mr, int nr, int kc) {
    for (int i = 0; i < mr; ++i) {
        C_TYPE* crow = C + static_cast<std::size_t>(i) * ldc;
        for (int j = 0; j < nr; ++j) {
            crow[j] = 0.0f;
        }
    }
    for (int k = 0; k < kc; ++k) {
        const C_TYPE* ap = A_panel + static_cast<std::size_t>(k) * lda;
        const C_TYPE* bp = B_panel + static_cast<std::size_t>(k) * ldb;
        for (int i = 0; i < mr; ++i) {
            const C_TYPE a = ap[i];
            C_TYPE* crow = C + static_cast<std::size_t>(i) * ldc;
            for (int j = 0; j < nr; ++j) {
                crow[j] += a * bp[j];
            }
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// C orchestrator: L2 column bands + row panels + k-major packing, with the
// RHS band double-buffered so that packing overlaps SME compute (paper 4.5).
//   A: N x M, B: M x K (configured inputs), C: N x K (FP32 output, row-major)
//
// The arithmetic of every full 32x32 tile is delegated to the SME FMOPA
// micro-kernel in src/assemble.s (huohua_sme_microkernel_32x32); edge tiles
// and machines with an unsupported vector length use scalar_microkernel.
// Both paths overwrite their tile, so every element of C is written exactly
// once here and C's incoming contents are irrelevant.
// ---------------------------------------------------------------------------
void gemm(const A_TYPE* A, const B_TYPE* B, C_TYPE* C,
               std::size_t N, std::size_t M, std::size_t K) {
    if (N == 0 || M == 0 || K == 0) {
        return;
    }

    const std::size_t Nc = compute_nc(N, M, K);
    const std::size_t lda = kTile;               // A panel width (rows per panel)
    const std::size_t ldb = Nc;                  // B band width (columns)
    const int ldc = static_cast<int>(K);
    const std::size_t n_bands = (K + Nc - 1) / Nc;

#if defined(__aarch64__)
    // The micro-kernel is hard-wired to a 512-bit streaming vector length
    // (16 FP32 lanes per Z register).  Only take it when the hardware agrees,
    // otherwise every tile goes through the portable scalar path.
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
    const std::size_t band_elems = M * ldb;
    const std::size_t n_slots = overlap ? 2u : 1u;
    C_TYPE* A_panel = static_cast<C_TYPE*>(std::malloc(M * lda * sizeof(C_TYPE)));
    C_TYPE* band_mem = static_cast<C_TYPE*>(
        std::malloc(n_slots * band_elems * sizeof(C_TYPE)));
    if (A_panel == nullptr || band_mem == nullptr) {
        std::free(A_panel);
        std::free(band_mem);
        return;
    }
    C_TYPE* slot[2] = {band_mem, overlap ? band_mem + band_elems : nullptr};

    // Consume one already-packed band: every A row panel against every 32-wide
    // slice of the band.  Shared by the overlapped and the synchronous paths.
    auto compute_band = [&](const C_TYPE* band, std::size_t col0,
                            std::size_t ncols) {
        for (std::size_t row0 = 0; row0 < N; row0 += kTile) {
            const std::size_t mr = (row0 + kTile <= N) ? kTile : (N - row0);
            pack_a_panel(A, A_panel, row0, mr, M, lda);

            for (std::size_t j0 = 0; j0 < ncols; j0 += kTile) {
                const std::size_t nr = (j0 + kTile <= ncols) ? kTile : (ncols - j0);
                C_TYPE* ctile = C + row0 * K + col0 + j0;

#if defined(__aarch64__)
                if (sme_ok && mr == kTile && nr == kTile) {
                    huohua_sme_microkernel_32x32(
                        A_panel, static_cast<int>(lda),
                        band + j0, static_cast<int>(ldb),
                        ctile, ldc, static_cast<int>(M));
                } else
#endif
                {
                    scalar_microkernel(A_panel, static_cast<int>(lda),
                                       band + j0, static_cast<int>(ldb),
                                       ctile, ldc,
                                       static_cast<int>(mr), static_cast<int>(nr),
                                       static_cast<int>(M));
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
            const C_TYPE* band = packer.acquire(i);
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
