#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "vmath.h"

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))
#define ERROR_TOL (0.0001f)

#define RUN_TEST_FLOAT(expr, expected) do { \
    float result = expr; \
    bool satisfied = (result == expected) || (fabsf(result - expected) < ERROR_TOL); \
    if (!satisfied) { \
        printf("%s:%-4d :: FAILED: " #expr " != " #expected ", (" #expr " == %.04f)\n", __FILE__, __LINE__, result); \
        tests_failed += 1; \
    } \
} while(0)

#define RUN_TEST_BOOL(expr, expected) do { \
    bool result = expr; \
    if (result != expected) { \
        printf("%s:%-4d :: FAILED: " #expr " != " #expected ", (" #expr " == %s)\n", __FILE__, __LINE__, result ? "true" : "false"); \
        tests_failed += 1; \
    } \
} while(0)

#define RUN_TEST_VEC2(expr, expected) do { \
    vec2 result = expr; \
    bool satisfied = true; \
    for (size_t ii = 0; ii < ARRAY_SIZE(result.elem); ii++) { \
        satisfied = satisfied && ((result.elem[ii] == expected.elem[ii]) || (fabsf(result.elem[ii] - expected.elem[ii]) < ERROR_TOL)); \
    } \
    if (!satisfied) { \
        printf("%s:%-4d :: FAILED: " #expr " != " #expected ", (" #expr " == " VEC2_FMT ")\n", __FILE__, __LINE__, VEC2_ARG(result)); \
        tests_failed += 1; \
    } \
} while(0)

#define RUN_TEST_VEC3(expr, expected) do { \
    vec3 result = expr; \
    bool satisfied = true; \
    for (size_t ii = 0; ii < ARRAY_SIZE(result.elem); ii++) { \
        satisfied = satisfied && ((result.elem[ii] == expected.elem[ii]) || (fabsf(result.elem[ii] - expected.elem[ii]) < ERROR_TOL)); \
    } \
    if (!satisfied) { \
        printf("%s:%-4d :: FAILED: " #expr " != " #expected ", (" #expr " == " VEC3_FMT ")\n", __FILE__, __LINE__, VEC3_ARG(result)); \
        tests_failed += 1; \
    } \
} while(0)

#define RUN_TEST_VEC4(expr, expected) do { \
    vec4 result = expr; \
    bool satisfied = true; \
    for (size_t ii = 0; ii < ARRAY_SIZE(result.elem); ii++) { \
        satisfied = satisfied && ((result.elem[ii] == expected.elem[ii]) || (fabsf(result.elem[ii] - expected.elem[ii]) < ERROR_TOL)); \
    } \
    if (!satisfied) { \
        printf("%s:%-4d :: FAILED: " #expr " != " #expected ", (" #expr " == " VEC4_FMT ")\n", __FILE__, __LINE__, VEC4_ARG(result)); \
        tests_failed += 1; \
    } \
} while(0)

#define RUN_TEST_MAT2X2(expr, expected) do { \
    mat2x2 result = expr; \
    bool satisfied = true; \
    for (size_t ii = 0; ii < ARRAY_SIZE(result.all); ii++) { \
        satisfied = satisfied && ((result.all[ii] == expected.all[ii]) || (fabsf(result.all[ii] - expected.all[ii]) < ERROR_TOL)); \
    } \
    if (!satisfied) { \
        printf("%s:%-4d :: FAILED: " #expr " != " #expected ", (" #expr " == " MAT2X2_FMT ")\n", __FILE__, __LINE__, MAT2X2_ARG(result)); \
        tests_failed += 1; \
    } \
} while(0)

#define RUN_TEST_MAT3X3(expr, expected) do { \
    mat3x3 result = expr; \
    bool satisfied = true; \
    for (size_t ii = 0; ii < ARRAY_SIZE(result.all); ii++) { \
        satisfied = satisfied && ((result.all[ii] == expected.all[ii]) || (fabsf(result.all[ii] - expected.all[ii]) < ERROR_TOL)); \
    } \
    if (!satisfied) { \
        printf("%s:%-4d :: FAILED: " #expr " != " #expected ", (" #expr " == " MAT3X3_FMT ")\n", __FILE__, __LINE__, MAT3X3_ARG(result)); \
        tests_failed += 1; \
    } \
} while(0)

#define RUN_TEST_MAT4X4(expr, expected) do { \
    mat4x4 result = expr; \
    bool satisfied = true; \
    for (size_t ii = 0; ii < ARRAY_SIZE(result.all); ii++) { \
        satisfied = satisfied && ((result.all[ii] == expected.all[ii]) || (fabsf(result.all[ii] - expected.all[ii]) < ERROR_TOL)); \
    } \
    if (!satisfied) { \
        printf("%s:%-4d :: FAILED: " #expr " != " #expected ", (" #expr " == " MAT4X4_FMT ")\n", __FILE__, __LINE__, MAT4X4_ARG(result)); \
        tests_failed += 1; \
    } \
} while(0)

int test_scalar(void)
{
    int tests_failed = 0;

    // vadd
    RUN_TEST_FLOAT(vadd(0.0f, 1.0f), 1.0f);
    RUN_TEST_FLOAT(vadd(-1.0f, 1.0f), 0.0f);
    RUN_TEST_FLOAT(vadd(1.0f, -1.0f), 0.0f);
    RUN_TEST_FLOAT(vadd(2.0f, 2.0f), 4.0f);
    RUN_TEST_FLOAT(vadd(0.0f, 0.0f), 0.0f);

    // vsub
    RUN_TEST_FLOAT(vsub(0.0f, 1.0f), -1.0f);
    RUN_TEST_FLOAT(vsub(-1.0f, 1.0f), -2.0f);
    RUN_TEST_FLOAT(vsub(1.0f, -1.0f), 2.0f);
    RUN_TEST_FLOAT(vsub(2.0f, 2.0f), 0.0f);
    RUN_TEST_FLOAT(vsub(0.0f, 0.0f), 0.0f);

    // vmul
    RUN_TEST_FLOAT(vmul(0.0f, 1.0f), 0.0f);
    RUN_TEST_FLOAT(vmul(1.0f, 0.0f), 0.0f);
    RUN_TEST_FLOAT(vmul(1.0f, 1.0f), 1.0f);
    RUN_TEST_FLOAT(vmul(2.0f, 2.0f), 4.0f);
    RUN_TEST_FLOAT(vmul(-2.0f, 2.0f), -4.0f);
    RUN_TEST_FLOAT(vmul(-2.0f, -2.0f), 4.0f);

    // vdiv
    RUN_TEST_FLOAT(vdiv(0.0f, 1.0f), 0.0f);
    RUN_TEST_FLOAT(vdiv(2.0f, 1.0f), 2.0f);
    RUN_TEST_FLOAT(vdiv(2.0f, 2.0f), 1.0f);
    RUN_TEST_FLOAT(vdiv(8.0f, 2.0f), 4.0f);
    RUN_TEST_FLOAT(vdiv(-8.0f, 2.0f), -4.0f);
    RUN_TEST_FLOAT(vdiv(1.0f, 0.0f), INF32);
    RUN_TEST_FLOAT(vdiv(-1.0f, 0.0f), -INF32);
    RUN_TEST_FLOAT(vdiv(-1.0f, INF32), -0.0f);
    RUN_TEST_FLOAT(vdiv(1.0f, -INF32), -0.0f);
    RUN_TEST_FLOAT(vdiv(1.0f, INF32), 0.0f);

    // vmag
    RUN_TEST_FLOAT(vmag(1.0f), 1.0f);
    RUN_TEST_FLOAT(vmag(2.0f), 2.0f);
    RUN_TEST_FLOAT(vmag(-2.0f), 2.0f);
    RUN_TEST_FLOAT(vmag(0.0f), 0.0f);
    RUN_TEST_FLOAT(vmag(INF32), INF32);
    RUN_TEST_FLOAT(vmag(-INF32), INF32);

    // vmag2
    RUN_TEST_FLOAT(vmag2(1.0f), 1.0f);
    RUN_TEST_FLOAT(vmag2(2.0f), 4.0f);
    RUN_TEST_FLOAT(vmag2(-2.0f), 4.0f);
    RUN_TEST_FLOAT(vmag2(0.0f), 0.0f);
    RUN_TEST_FLOAT(vmag2(INF32), INF32);
    RUN_TEST_FLOAT(vmag2(-INF32), INF32);

    // vlerp
    RUN_TEST_FLOAT(vlerp(0.0f, 1.0f, 0.0f), 0.0f);
    RUN_TEST_FLOAT(vlerp(0.0f, 1.0f, 1.0f), 1.0f);
    RUN_TEST_FLOAT(vlerp(0.0f, 1.0f, 0.5f), 0.5f);
    RUN_TEST_FLOAT(vlerp(-1.0f, 1.0f, 0.5f), 0.0f);
    RUN_TEST_FLOAT(vlerp(1.0f, -1.0f, 0.5f), 0.0f);

    // vequ
    RUN_TEST_BOOL(vequ(0.0f, 0.0f), true);
    RUN_TEST_BOOL(vequ(0.0f, 1.0f), false);
    RUN_TEST_BOOL(vequ(-1.0f, 1.0f), false);
    RUN_TEST_BOOL(vequ(1.0f, 1.0f + EPSILON32), true);
    RUN_TEST_BOOL(vequ(1.0f, 1.0f + 2 * EPSILON32), false);
    RUN_TEST_BOOL(vequ(3.0f, 1.0f + 2.0f), true);

    // vmin
    RUN_TEST_FLOAT(vmin(0.0f, 1.0f), 0.0f);
    RUN_TEST_FLOAT(vmin(1.0f, 0.0f), 0.0f);
    RUN_TEST_FLOAT(vmin(-1.0f, 0.0f), -1.0f);
    RUN_TEST_FLOAT(vmin(-INF32, INF32), -INF32);

    // vmax
    RUN_TEST_FLOAT(vmax(0.0f, 1.0f), 1.0f);
    RUN_TEST_FLOAT(vmax(1.0f, 0.0f), 1.0f);
    RUN_TEST_FLOAT(vmax(-1.0f, 0.0f), 0.0f);
    RUN_TEST_FLOAT(vmax(-INF32, INF32), INF32);

    // vsqrt
    RUN_TEST_FLOAT(vsqrt(0.0f), 0.0f);
    RUN_TEST_FLOAT(vsqrt(0.25f), 0.5f);
    RUN_TEST_FLOAT(vsqrt(1.0f), 1.0f);
    RUN_TEST_FLOAT(vsqrt(4.0f), 2.0f);

    // vrsqrt
    RUN_TEST_FLOAT(vrsqrt(0.0f), INF32);
    RUN_TEST_FLOAT(vrsqrt(0.25f), 2.0f);
    RUN_TEST_FLOAT(vrsqrt(1.0f), 1.0f);
    RUN_TEST_FLOAT(vrsqrt(4.0f), 0.5f);

    // vrcp
    RUN_TEST_FLOAT(vrcp(0.0f), INF32);
    RUN_TEST_FLOAT(vrcp(1.0f), 1.0f);
    RUN_TEST_FLOAT(vrcp(2.0f), 0.5f);
    RUN_TEST_FLOAT(vrcp(4.0f), 0.25f);
    RUN_TEST_FLOAT(vrcp(10.0f), 0.1f);

    // vclamp
    RUN_TEST_FLOAT(vclamp(0.0f, -1.0f, 1.0f), 0.0f);
    RUN_TEST_FLOAT(vclamp(0.0f, 1.0f, 2.0f), 1.0f);
    RUN_TEST_FLOAT(vclamp(3.0f, 1.0f, 2.0f), 2.0f);
    RUN_TEST_FLOAT(vclamp(-1.0f, -INF32, INF32), -1.0f);

    // vneg
    RUN_TEST_FLOAT(vneg(0.0f), 0.0f);
    RUN_TEST_FLOAT(vneg(1.0f), -1.0f);
    RUN_TEST_FLOAT(vneg(2.0f), -2.0f);
    RUN_TEST_FLOAT(vneg(-2.0f), 2.0f);
    RUN_TEST_FLOAT(vneg(INF32), -INF32);
    RUN_TEST_FLOAT(vneg(-INF32), INF32);

    // vdist
    RUN_TEST_FLOAT(vdist(0, 0), 0.0f);
    RUN_TEST_FLOAT(vdist(0, 1), 1.0f);
    RUN_TEST_FLOAT(vdist(1, 0), 1.0f);
    RUN_TEST_FLOAT(vdist(0, -1), 1.0f);
    RUN_TEST_FLOAT(vdist(-1, 0), 1.0f);
    RUN_TEST_FLOAT(vdist(-1, 1), 2.0f);
    RUN_TEST_FLOAT(vdist(2, 9), 7.0f);

    // vsum
    RUN_TEST_FLOAT(vsum(7.0f, 2.0f), 9.0f);
    RUN_TEST_FLOAT(vsum(1.0f, 2.0f, 3.0f), 6.0f);
    RUN_TEST_FLOAT(vsum(0.0f, 0.0f, 1.0f), 1.0f);
    RUN_TEST_FLOAT(vsum(-1.0f, 4.0f, 1.0f), 4.0f);
    RUN_TEST_FLOAT(vsum(1.0f, 2.0f, 3.0f, 4.0f), 10.0f);
    RUN_TEST_FLOAT(vsum(1.0f, 2.0f, 3.0f, 4.0f, 5.0f), 15.0f);
    RUN_TEST_FLOAT(vsum(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f), 21.0f);

    // vprod
    RUN_TEST_FLOAT(vprod(1.0f, 2.0f), 2.0f);
    RUN_TEST_FLOAT(vprod(1.0f, 2.0f, 4.0f), 8.0f);
    RUN_TEST_FLOAT(vprod(-1.0f, 2.0f, 4.0f), -8.0f);
    RUN_TEST_FLOAT(vprod(-2.0f, -2.0f, -2.0f), -8.0f);
    RUN_TEST_FLOAT(vprod(1.0f, 2.0f, 3.0f, 4.0f), 24.0f);
    RUN_TEST_FLOAT(vprod(1.0f, 2.0f, 3.0f, 4.0f, 5.0f), 120.0f);

    return tests_failed;
}

int test_vec2(void)
{
    int tests_failed = 0;

    // vadd
    RUN_TEST_VEC2(vadd(vec2(0, 0), vec2(0, 0)), vec2(0, 0));
    RUN_TEST_VEC2(vadd(vec2(1, 2), vec2(3, 4)), vec2(4, 6));
    RUN_TEST_VEC2(vadd(vec2(-3, -1), vec2(3, 1)), vec2(0, 0));
    RUN_TEST_VEC2(vadd(vec2(1, 1), vec2(0, 0)), vec2(1, 1));

    // vsub
    RUN_TEST_VEC2(vsub(vec2(0, 0), vec2(0, 0)), vec2(0, 0));
    RUN_TEST_VEC2(vsub(vec2(1, 2), vec2(3, 4)), vec2(-2, -2));
    RUN_TEST_VEC2(vsub(vec2(-3, -1), vec2(3, 1)), vec2(-6, -2));
    RUN_TEST_VEC2(vsub(vec2(1, 1), vec2(0, 0)), vec2(1, 1));
    RUN_TEST_VEC2(vsub(vec2(2, 4), vec2(2, 4)), vec2(0, 0));

    // vmul (scalar)
    RUN_TEST_VEC2(vmul(2, vec2(0, 0)), vec2(0, 0));
    RUN_TEST_VEC2(vmul(0, vec2(2, 2)), vec2(0, 0));
    RUN_TEST_VEC2(vmul(2, vec2(-2, 3)), vec2(-4, 6));
    RUN_TEST_VEC2(vmul(1, vec2(-2, 3)), vec2(-2, 3));
    RUN_TEST_VEC2(vmul(-1, vec2(-2, 3)), vec2(2, -3));
    RUN_TEST_VEC2(vmul(vec2(0, 0), 2), vec2(0, 0));
    RUN_TEST_VEC2(vmul(vec2(2, 2), 0), vec2(0, 0));
    RUN_TEST_VEC2(vmul(vec2(-2, 3), 2), vec2(-4, 6));
    RUN_TEST_VEC2(vmul(vec2(-2, 3), 1), vec2(-2, 3));
    RUN_TEST_VEC2(vmul(vec2(-2, 3), -1), vec2(2, -3));
    RUN_TEST_VEC2(vmul(vec2(-1, 1), INF32), vec2(-INF32, INF32));

    // vmul (vec2)
    RUN_TEST_VEC2(vmul(vec2(0, 0), vec2(2, 0)), vec2(0, 0));
    RUN_TEST_VEC2(vmul(vec2(2, 0), vec2(0, 2)), vec2(0, 0));
    RUN_TEST_VEC2(vmul(vec2(2, 4), vec2(1, 3)), vec2(2, 12));
    RUN_TEST_VEC2(vmul(vec2(-1, 2), vec2(4, -1)), vec2(-4, -2));
    RUN_TEST_VEC2(vmul(vec2(-1, 2), vec2(-4, 1)), vec2(4, 2));
    RUN_TEST_VEC2(vmul(vec2(-1, 1), vec2(INF32, INF32)), vec2(-INF32, INF32));

    // vdiv (scalar)
    RUN_TEST_VEC2(vdiv(vec2(0, 0), 1), vec2(0, 0));
    RUN_TEST_VEC2(vdiv(vec2(2, 4), 1), vec2(2, 4));
    RUN_TEST_VEC2(vdiv(vec2(2, 4), 2), vec2(1, 2));
    RUN_TEST_VEC2(vdiv(vec2(9, 24), 3), vec2(3, 8));
    RUN_TEST_VEC2(vdiv(vec2(2, 4), -2), vec2(-1, -2));
    RUN_TEST_VEC2(vdiv(vec2(-2, 4), -1), vec2(2, -4));
    RUN_TEST_VEC2(vdiv(vec2(1, -1), 0), vec2(INF32, -INF32));
    RUN_TEST_VEC2(vdiv(vec2(1, -1), INF32), vec2(0.0f, -0.0f));

    // vdiv (vec2)
    RUN_TEST_VEC2(vdiv(vec2(0, 0), vec2(1, 2)), vec2(0, 0));
    RUN_TEST_VEC2(vdiv(vec2(2, 4), vec2(-1, 2)), vec2(-2, 2));
    RUN_TEST_VEC2(vdiv(vec2(9, 24), vec2(3, 8)), vec2(3, 3));
    RUN_TEST_VEC2(vdiv(vec2(-10, 20), vec2(1, -10)), vec2(-10, -2));
    RUN_TEST_VEC2(vdiv(vec2(1, -1), vec2(0, 0)), vec2(INF32, -INF32));
    RUN_TEST_VEC2(vdiv(vec2(1, -1), vec2(INF32, INF32)), vec2(0.0f, -0.0f));

    // vdot
    RUN_TEST_FLOAT(vdot(vec2(0, 0), vec2(0, 0)), 0.0f);
    RUN_TEST_FLOAT(vdot(vec2(1, 1), vec2(0, 0)), 0.0f);
    RUN_TEST_FLOAT(vdot(vec2(1, 1), vec2(1, 1)), 2.0f);
    RUN_TEST_FLOAT(vdot(vec2(2, 4), vec2(-3, 1)), -2.0f);
    RUN_TEST_FLOAT(vdot(vec2(0, 1), vec2(1, 0)), 0.0f);
    RUN_TEST_FLOAT(vdot(vec2(0, -1), vec2(0, 1)), -1.0f);

    // TODO: more tests
    // vcross
    RUN_TEST_VEC3(vcross(vec2(1, 0), vec2(0, 1)), vec3(0, 0, 1));
    RUN_TEST_VEC3(vcross(vec2(2, 0), vec2(0, 2)), vec3(0, 0, 4));

    // vmag
    RUN_TEST_FLOAT(vmag(vec2(0, 0)), 0.0f);
    RUN_TEST_FLOAT(vmag(vec2(1, 0)), 1.0f);
    RUN_TEST_FLOAT(vmag(vec2(0, 1)), 1.0f);
    RUN_TEST_FLOAT(vmag(vec2(-1, 0)), 1.0f);
    RUN_TEST_FLOAT(vmag(vec2(3, 4)), 5.0f);
    RUN_TEST_FLOAT(vmag(vec2(-3, -4)), 5.0f);

    // vmag2
    RUN_TEST_FLOAT(vmag2(vec2(0, 0)), 0.0f);
    RUN_TEST_FLOAT(vmag2(vec2(1, 0)), 1.0f);
    RUN_TEST_FLOAT(vmag2(vec2(0, 1)), 1.0f);
    RUN_TEST_FLOAT(vmag2(vec2(-1, 0)), 1.0f);
    RUN_TEST_FLOAT(vmag2(vec2(3, 4)), 25.0f);
    RUN_TEST_FLOAT(vmag2(vec2(-3, -4)), 25.0f);

    // vnorm
    RUN_TEST_VEC2(vnorm(vec2(1, 0)), vec2(1, 0));
    RUN_TEST_VEC2(vnorm(vec2(0, 1)), vec2(0, 1));
    RUN_TEST_VEC2(vnorm(vec2(-1, 0)), vec2(-1, 0));
    RUN_TEST_VEC2(vnorm(vec2(4, 0)), vec2(1, 0));
    RUN_TEST_VEC2(vnorm(vec2(0, -4)), vec2(0, -1));
    RUN_TEST_VEC2(vnorm(vec2(-1, 1)), vec2(-1.0f / sqrtf(2.0f), 1.0f / sqrtf(2.0f)));

    // vlerp
    RUN_TEST_VEC2(vlerp(vec2(0, 0), vec2(0, 0), 0.0f), vec2(0, 0));
    RUN_TEST_VEC2(vlerp(vec2(0, 1), vec2(1, 0), 0.0f), vec2(0, 1));
    RUN_TEST_VEC2(vlerp(vec2(0, 1), vec2(1, 0), 1.0f), vec2(1, 0));
    RUN_TEST_VEC2(vlerp(vec2(0, 1), vec2(1, 0), 0.5f), vec2(0.5f, 0.5f));
    RUN_TEST_VEC2(vlerp(vec2(0, -1), vec2(1, 0), 0.5f), vec2(0.5f, -0.5f));

    // vequ
    RUN_TEST_BOOL(vequ(vec2(0, 0), vec2(0, 0)), true);
    RUN_TEST_BOOL(vequ(vec2(0, 1), vec2(0, 0)), false);
    RUN_TEST_BOOL(vequ(vec2(1, 0), vec2(0, 0)), false);
    RUN_TEST_BOOL(vequ(vec2(1, -1), vec2(1, -1)), true);
    RUN_TEST_BOOL(vequ(vec2(1, -1), vec2(-1, 1)), false);
    RUN_TEST_BOOL(vequ(vec2(1, 1), vec2(1 + EPSILON32, 1 + EPSILON32)), true);
    RUN_TEST_BOOL(vequ(vec2(-1, -1), vec2(-1, -1)), true);

    // vmax
    RUN_TEST_VEC2(vmax(vec2(0, 0), vec2(0, 0)), vec2(0, 0));
    RUN_TEST_VEC2(vmax(vec2(1, -1), vec2(0, 0)), vec2(1, 0));
    RUN_TEST_VEC2(vmax(vec2(7, 7), vec2(1, 1)), vec2(7, 7));
    RUN_TEST_VEC2(vmax(vec2(-8, 2), vec2(-2, 4)), vec2(-2, 4));
    RUN_TEST_VEC2(vmax(vec2(INF32, -INF32), vec2(0, 0)), vec2(INF32, 0));
    RUN_TEST_VEC2(vmax(vec2(-INF32, INF32), vec2(INF32, -INF32)), vec2(INF32, INF32));

    // vmin
    RUN_TEST_VEC2(vmin(vec2(0, 0), vec2(0, 0)), vec2(0, 0));
    RUN_TEST_VEC2(vmin(vec2(1, -1), vec2(0, 0)), vec2(0, -1));
    RUN_TEST_VEC2(vmin(vec2(7, 7), vec2(1, 1)), vec2(1, 1));
    RUN_TEST_VEC2(vmin(vec2(-8, 2), vec2(-2, 4)), vec2(-8, 2));
    RUN_TEST_VEC2(vmin(vec2(INF32, -INF32), vec2(0, 0)), vec2(0, -INF32));
    RUN_TEST_VEC2(vmin(vec2(-INF32, INF32), vec2(INF32, -INF32)), vec2(-INF32, -INF32));

    // vsqrt
    RUN_TEST_VEC2(vsqrt(vec2(0, 0)), vec2(0, 0));
    RUN_TEST_VEC2(vsqrt(vec2(0.25f, 0.25f)), vec2(0.5f, 0.5f));
    RUN_TEST_VEC2(vsqrt(vec2(1, 1)), vec2(1, 1));
    RUN_TEST_VEC2(vsqrt(vec2(0, 1)), vec2(0, 1));
    RUN_TEST_VEC2(vsqrt(vec2(4, 16)), vec2(2, 4));
    RUN_TEST_VEC2(vsqrt(vec2(0, 100)), vec2(0, 10));

    // vrsqrt
    RUN_TEST_VEC2(vrsqrt(vec2(0, 0)), vec2(INF32, INF32));
    RUN_TEST_VEC2(vrsqrt(vec2(0.25f, 0.25f)), vec2(2.0f, 2.0f));
    RUN_TEST_VEC2(vrsqrt(vec2(1, 1)), vec2(1, 1));
    RUN_TEST_VEC2(vrsqrt(vec2(0, 1)), vec2(INF32, 1));
    RUN_TEST_VEC2(vrsqrt(vec2(4, 16)), vec2(0.5f, 0.25f));
    RUN_TEST_VEC2(vrsqrt(vec2(0, 100)), vec2(INF32, 0.1f));

    // vrcp
    RUN_TEST_VEC2(vrcp(vec2(0, 0)), vec2(INF32, INF32));
    RUN_TEST_VEC2(vrcp(vec2(0.25f, 0.25f)), vec2(4.0f, 4.0f));
    RUN_TEST_VEC2(vrcp(vec2(1, 1)), vec2(1, 1));
    RUN_TEST_VEC2(vrcp(vec2(0, 1)), vec2(INF32, 1));
    RUN_TEST_VEC2(vrcp(vec2(4, 2)), vec2(0.25f, 0.5f));
    RUN_TEST_VEC2(vrcp(vec2(0, -10)), vec2(INF32, -0.1f));

    // vclamp
    RUN_TEST_VEC2(vclamp(vec2(0, 0), vec2(0, 0), vec2(0, 0)), vec2(0, 0));
    RUN_TEST_VEC2(vclamp(vec2(-1, 1), vec2(0, 0), vec2(1, 1)), vec2(0, 1));
    RUN_TEST_VEC2(vclamp(vec2(-1, 1), vec2(-1, -1), vec2(0, 0)), vec2(-1, 0));
    RUN_TEST_VEC2(vclamp(vec2(-1, 1), vec2(-1, -1), vec2(1, 1)), vec2(-1, 1));
    RUN_TEST_VEC2(vclamp(vec2(4, -10), vec2(-10, -10), vec2(10, 10)), vec2(4, -10));
    RUN_TEST_VEC2(vclamp(vec2(3, -9), vec2(-INF32, -INF32), vec2(INF32, INF32)), vec2(3, -9));

    // vneg
    RUN_TEST_VEC2(vneg(vec2(0, 0)), vec2(0, 0));
    RUN_TEST_VEC2(vneg(vec2(-1, 1)), vec2(1, -1));
    RUN_TEST_VEC2(vneg(vec2(4, -16)), vec2(-4, 16));
    RUN_TEST_VEC2(vneg(vec2(INF32, -INF32)), vec2(-INF32, INF32));

    // vdist
    RUN_TEST_FLOAT(vdist(vec2(0, 0), vec2(0, 0)), 0.0f);
    RUN_TEST_FLOAT(vdist(vec2(0, 0), vec2(0, 1)), 1.0f);
    RUN_TEST_FLOAT(vdist(vec2(0, 1), vec2(0, 0)), 1.0f);
    RUN_TEST_FLOAT(vdist(vec2(-1, 1), vec2(1, -1)), 2.0f * sqrtf(2.0f));
    RUN_TEST_FLOAT(vdist(vec2(0, 0), vec2(3, 4)), 5.0f);

    // vsum
    RUN_TEST_VEC2(vsum(vec2(0, 0), vec2(1, -1)), vec2(1, -1));
    RUN_TEST_VEC2(vsum(vec2(0, 0), vec2(1, -1), vec2(2, -2)), vec2(3, -3));
    RUN_TEST_VEC2(vsum(vec2(0, 0), vec2(1, -1), vec2(2, -2), vec2(3, -3)), vec2(6, -6));

    // vprod
    RUN_TEST_VEC2(vprod(vec2(1, -1), vec2(2, -2)), vec2(2, 2));
    RUN_TEST_VEC2(vprod(vec2(1, -1), vec2(2, -2), vec2(3, -3)), vec2(6, -6));
    RUN_TEST_VEC2(vprod(vec2(1, -1), vec2(2, -2), vec2(3, -3), vec2(4, -4)), vec2(24, 24));

    return tests_failed;
}

int test_vec3(void)
{
    int tests_failed = 0;
    return tests_failed;
}

int test_vec4(void)
{
    int tests_failed = 0;
    return tests_failed;
}

int test_mat2x2(void)
{
    int tests_failed = 0;
    return tests_failed;
}

int test_mat3x3(void)
{
    int tests_failed = 0;
    return tests_failed;
}

int test_mat4x4(void)
{
    int tests_failed = 0;
    return tests_failed;
}

int test_misc(void)
{
    int tests_failed = 0;
    return tests_failed;
}

int main(void) {
    printf("Running vmath tests :: VMATH_SIMD = %d\n", VMATH_SIMD);

    int total_tests_failed = 0;

    int scalar_tests_failed = test_scalar();
    total_tests_failed += scalar_tests_failed;
    printf("Scalar Tests Failed: %d\n", scalar_tests_failed);

    int vec2_tests_failed = test_vec2();
    total_tests_failed += vec2_tests_failed;
    printf("Vec2 Tests Failed: %d\n", vec2_tests_failed);

    int vec3_tests_failed = test_vec3();
    total_tests_failed += vec3_tests_failed;
    printf("Vec3 Tests Failed: %d\n", vec3_tests_failed);

    int vec4_tests_failed = test_vec4();
    total_tests_failed += vec4_tests_failed;
    printf("Vec4 Tests Failed: %d\n", vec4_tests_failed);

    int mat2x2_tests_failed = test_mat2x2();
    total_tests_failed += mat2x2_tests_failed;
    printf("Mat2x2 Tests Failed: %d\n", mat2x2_tests_failed);

    int mat3x3_tests_failed = test_mat3x3();
    total_tests_failed += mat3x3_tests_failed;
    printf("Mat3x3 Tests Failed: %d\n", mat3x3_tests_failed);

    int mat4x4_tests_failed = test_mat4x4();
    total_tests_failed += mat4x4_tests_failed;
    printf("Mat4x4 Tests Failed: %d\n", mat4x4_tests_failed);

    int misc_tests_failed = test_misc();
    total_tests_failed += misc_tests_failed;
    printf("Misc Tests Failed: %d\n", misc_tests_failed);

    printf("Total Tests Failed: %d\n", total_tests_failed);

    return 0;
}
