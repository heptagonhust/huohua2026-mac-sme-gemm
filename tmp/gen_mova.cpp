#include <arm_sme.h>
__arm_locally_streaming __attribute__((noinline))
svfloat32_t probe(svfloat32_t fallback, svbool_t pg) {
    return svread_hor_za32_f32_m(fallback, pg, 0, 5);
}
