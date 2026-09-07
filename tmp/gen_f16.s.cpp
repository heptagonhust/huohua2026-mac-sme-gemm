#include <arm_sme.h>
__arm_locally_streaming __attribute__((noinline))
svfloat32_t probe(svfloat32_t fallback, svbool_t pg) {
    (void)fallback;
    svfloat16_t a = svundef_f16();
    svfloat16_t b = svundef_f16();
    svmopa_za32_f16_m(0, pg, pg, a, b);
    svfloat32_t r = svread_hor_za32_f32_m(svdup_f32(0.0f), pg, 0, 5);
    return r;
}
