// Temporary diagnostic driver for tmp/abi_audit.s
#include <cstdint>
#include <cstdio>

extern "C" void abi_audit(unsigned long long* out);

int main() {
    unsigned long long out[52] = {0};
    abi_audit(out);

    std::printf("---- low 64 bits of v8..v15 around SMSTART+SMSTOP ----\n");
    const unsigned long long want_byte[8] = {0x11, 0x22, 0x33, 0x44,
                                             0x55, 0x66, 0x77, 0x88};
    for (int i = 0; i < 8; ++i) {
        const unsigned long long expect = want_byte[i] * 0x0101010101010101ULL;
        const unsigned long long lo_b = out[i * 2];
        const unsigned long long lo_a = out[16 + i * 2];
        std::printf("d%-2d  %016llx -> %016llx %s\n", i + 8, lo_b, lo_a,
                    lo_a == expect ? "(kept)" : "<<< LOST");
    }

    std::printf("\n---- x19..x28 around SMSTART+SMSTOP ----\n");
    for (int i = 0; i < 10; ++i) {
        const unsigned long long b = out[32 + i], a = out[42 + i];
        std::printf("x%-2d  %016llx -> %016llx %s\n", 19 + i, b, a,
                    b == a ? "(kept)" : "<<< LOST");
    }
    return 0;
}
