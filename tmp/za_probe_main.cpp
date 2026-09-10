// Temporary diagnostic driver for tmp/za_probe.s
#include <cstdint>
#include <cstdio>

extern "C" void za_probe(unsigned long long* out);

int main() {
    unsigned long long out[27] = {0};
    za_probe(out);

    static const char* steps[9] = {
        "A  patterns installed        ",
        "B  after smstart             ",
        "C  after ptrue p0.s          ",
        "D  after zero {za}           ",
        "E  after 4x ld1w {z0..z3}    ",
        "F  after 4x fmopa            ",
        "G  after 2x mov z4/z5,za..h  ",
        "H  after 2x st1w             ",
        "I  after smstop              "};

    const unsigned long long want[3] = {0x1111111111111111ULL,
                                        0x2222222222222222ULL,
                                        0x3333333333333333ULL};
    int first_bad = -1;
    for (int s = 0; s < 9; ++s) {
        unsigned long long* v = &out[s * 3];
        const bool ok = v[0] == want[0] && v[1] == want[1] && v[2] == want[2];
        if (!ok && first_bad < 0) {
            first_bad = s;
        }
        std::printf("%s : z11=%016llx z12=%016llx z13=%016llx  %s\n", steps[s],
                    v[0], v[1], v[2], ok ? "ok" : "<<< CHANGED");
    }
    std::printf("\nfirst checkpoint where z11/z12/z13 changed: %s\n",
                first_bad < 0 ? "none" : steps[first_bad]);
    return 0;
}
