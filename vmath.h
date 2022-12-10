#pragma once

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/* clang-format off */

#define VMATH_SIMD_NONE 0
#define VMATH_SIMD_SSE  1
#define VMATH_SIMD_AVX2 2

#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
#   define VMATH_ALIGN16     __attribute__((aligned(16)))
#   define VMATH_ALIGN32     __attribute__((aligned(32)))
#   define VMATH_UNLIKELY(x) __builtin_expect(!!(x), 0)
#   define VMATH_LIKELY(x)   __builtin_expect(!!(x), 1)
#   if !defined(VMATH_SIMD)
#       if defined(__AVX2__)
#           define VMATH_SIMD VMATH_SIMD_AVX2
#       elif defined(__SSE__)
#           define VMATH_SIMD VMATH_SIMD_SSE
#       else
#           define VMATH_SIMD VMATH_SIMD_NONE
#       endif
#   endif
#elif defined(_MSC_VER)
#   define VMATH_ALIGN16     __declspec(align(16))
#   define VMATH_ALIGN32     __declspec(align(32))
#   define VMATH_UNLIKELY(x) (x)
#   define VMATH_LIKELY(x)   (x)
#   if !defined(VMATH_SIMD)
#       if defined(__AVX2__)
#           define VMATH_SIMD VMATH_SIMD_AVX2
#       elif _M_IX86_FP >= 1
#           define VMATH_SIMD VMATH_SIMD_SSE
#       else
#           define VMATH_SIMD VMATH_SIMD_NONE
#       endif
#   endif
#endif

#if VMATH_SIMD > 0
#   include <immintrin.h>
#endif

/* ---- Macros ---- */

#define PI32 (3.1415926535897932384626f)
#define EPSILON32 (FLT_EPSILON)

// TODO: make these compatible with different compilers and add some fallback for ISO C
#ifndef INF32
#define INF32 (__builtin_inff())
#endif

#ifndef NAN32
#define NAN32 (__builtin_nanf(""))
#endif

// TODO: this doesn't appear to be faster than powf on -Ofast + unsafe math
// needs more testing, but this is interesting
#define POWF_2(base) ((base) * (base))
#define POWF_3(base) (POWF_2((base)) * (base))
#define POWF_4(base) (POWF_2((base)) * POWF_2((base)))
#define POWF_5(base) (POWF_4((base)) * (base))

#define POWF_(base, exp) (POWF_##exp((base)))
#if 0
#define POWF(base, exp) (POWF_((base), exp))
#else
#define POWF(base, exp) (powf((base), (exp)))
#endif

#define VMATH_BLEND_MASK(x, y, z, w) (((w) << 3) | ((z) << 2) | ((y) << 1) | ((x) << 0))

#define VMATH_PRECISION ".02"

#define VEC2_FMT "<%" VMATH_PRECISION "f, %" VMATH_PRECISION "f>"
#define VEC2_ARG(vec) vec.x, vec.y

#define VEC3_FMT "<%" VMATH_PRECISION "f, %" VMATH_PRECISION "f, %" VMATH_PRECISION "f>"
#define VEC3_ARG(vec) vec.x, vec.y, vec.z

#define VEC4_FMT "<%" VMATH_PRECISION "f, %" VMATH_PRECISION "f, %" VMATH_PRECISION "f, %" VMATH_PRECISION "f>"
#define VEC4_ARG(vec) vec.x, vec.y, vec.z, vec.w

// TODO: padding to get the alignment correct

#define MAT2X2_FMT "\n" \
                   "| %.02f %.02f |\n" \
                   "| %.02f %.02f |\n"
#define MAT2X2_ARG(mat) mat.x.x, mat.y.x, \
                        mat.x.y, mat.y.y

#define MAT3X3_FMT "\n" \
                   "| %.02f %.02f %.02f |\n" \
                   "| %.02f %.02f %.02f |\n" \
                   "| %.02f %.02f %.02f |\n"
#define MAT3X3_ARG(mat) mat.x.x, mat.y.x, mat.z.x, \
                        mat.x.y, mat.y.y, mat.z.y, \
                        mat.x.z, mat.y.z, mat.z.z

#define MAT4X4_FMT "\n" \
                   "| %.02f %.02f %.02f %.02f |\n" \
                   "| %.02f %.02f %.02f %.02f |\n" \
                   "| %.02f %.02f %.02f %.02f |\n" \
                   "| %.02f %.02f %.02f %.02f |\n"
#define MAT4X4_ARG(mat) mat.x.x, mat.y.x, mat.z.x, mat.w.x, \
                        mat.x.y, mat.y.y, mat.z.y, mat.w.y, \
                        mat.x.z, mat.y.z, mat.z.z, mat.w.z, \
                        mat.x.w, mat.y.w, mat.z.w, mat.w.w

#define AXIS_FMT "%s"
#define AXIS_ARG(axis) (((const char*[]) {"x-axis", "y-axis", "z-axis", "w-axis"})[axis])

#define VMAP _Generic
#define VBIND(type, func) \
    type:                 \
    func

#define vadd(x, y)                 \
    VMAP((x),                      \
        VBIND(vec2, vec2_Add),     \
        VBIND(vec3, vec3_Add),     \
        VBIND(vec4, vec4_Add),     \
        VBIND(mat2x2, mat2x2_Add), \
        VBIND(mat3x3, mat3x3_Add), \
        VBIND(mat4x4, mat4x4_Add), \
        VBIND(default, scalar_Add) \
    )((x), (y))

#define vsub(x, y)                      \
    VMAP((x),                           \
        VBIND(vec2, vec2_Subtract),     \
        VBIND(vec3, vec3_Subtract),     \
        VBIND(vec4, vec4_Subtract),     \
        VBIND(mat2x2, mat2x2_Subtract), \
        VBIND(mat3x3, mat3x3_Subtract), \
        VBIND(mat4x4, mat4x4_Subtract), \
        VBIND(default, scalar_Subtract) \
    )((x), (y))

#define vmul(x, y)                                       \
    VMAP((x),                                            \
        VBIND(vec2,                                      \
            VMAP((y),                                    \
                VBIND(vec2, vec2_MultiplyComponents),    \
                VBIND(default, vec2_MultiplyScalar))),   \
        VBIND(vec3,                                      \
            VMAP((y),                                    \
                VBIND(vec3, vec3_MultiplyComponents),    \
                VBIND(default, vec3_MultiplyScalar))),   \
        VBIND(vec4,                                      \
            VMAP((y),                                    \
                VBIND(vec4, vec4_MultiplyComponents),    \
                VBIND(default, vec4_MultiplyScalar))),   \
        VBIND(mat2x2,                                    \
            VMAP((y),                                    \
                VBIND(vec2, mat2x2_MultiplyVector),      \
                VBIND(mat2x2, mat2x2_MultiplyMatrix),    \
                VBIND(default, mat2x2_MultiplyScalar))), \
        VBIND(mat3x3,                                    \
            VMAP((y),                                    \
                VBIND(vec3, mat3x3_MultiplyVector),      \
                VBIND(mat3x3, mat3x3_MultiplyMatrix),    \
                VBIND(default, mat3x3_MultiplyScalar))), \
        VBIND(mat4x4,                                    \
            VMAP((y),                                    \
                VBIND(vec4, mat4x4_MultiplyVector),      \
                VBIND(mat4x4, mat4x4_MultiplyMatrix),    \
                VBIND(default, mat4x4_MultiplyScalar))), \
        VBIND(default,                                   \
            VMAP((y),                                    \
                VBIND(vec2, vec2_MultiplyScalarR),       \
                VBIND(vec3, vec3_MultiplyScalarR),       \
                VBIND(vec4, vec4_MultiplyScalarR),       \
                VBIND(mat2x2, mat2x2_MultiplyScalarR),   \
                VBIND(mat3x3, mat3x3_MultiplyScalarR),   \
                VBIND(mat4x4, mat4x4_MultiplyScalarR),   \
                VBIND(default, scalar_Multiply)))        \
    )((x), (y))

#define vdiv(x, y)                                    \
    VMAP((x),                                         \
        VBIND(vec2,                                   \
            VMAP((y),                                 \
                VBIND(vec2, vec2_DivideComponents),   \
                VBIND(default, vec2_DivideScalar))),  \
        VBIND(vec3,                                   \
            VMAP((y),                                 \
                VBIND(vec3, vec3_DivideComponents),   \
                VBIND(default, vec3_DivideScalar))),  \
        VBIND(vec4,                                   \
            VMAP((y),                                 \
                VBIND(vec4, vec4_DivideComponents),   \
                VBIND(default, vec4_DivideScalar))),  \
        VBIND(mat2x2, mat2x2_DivideScalar),           \
        VBIND(mat3x3, mat3x3_DivideScalar),           \
        VBIND(mat4x4, mat4x4_DivideScalar),           \
        VBIND(default, scalar_Divide)                 \
    )((x), (y))

#define vdot(x, y)                    \
    VMAP((x),                         \
        VBIND(vec2, vec2_DotProduct), \
        VBIND(vec3, vec3_DotProduct), \
        VBIND(vec4, vec4_DotProduct)  \
    )((x), (y))

#define vcross(x, y)                    \
    VMAP((x),                           \
        VBIND(vec2, vec2_CrossProduct), \
        VBIND(vec3, vec3_CrossProduct)  \
    )((x), (y))

#define vmag(x)                          \
    VMAP((x),                            \
        VBIND(vec2, vec2_Magnitude),     \
        VBIND(vec3, vec3_Magnitude),     \
        VBIND(vec4, vec4_Magnitude),     \
        VBIND(default, scalar_Magnitude) \
    )((x))

#define vmag2(x)                                \
    VMAP((x),                                   \
        VBIND(vec2, vec2_MagnitudeSquared),     \
        VBIND(vec3, vec3_MagnitudeSquared),     \
        VBIND(vec4, vec4_MagnitudeSquared),     \
        VBIND(default, scalar_MagnitudeSquared) \
    )((x))

#define vnorm(x)                     \
    VMAP((x),                        \
        VBIND(vec2, vec2_Normalize), \
        VBIND(vec3, vec3_Normalize), \
        VBIND(vec4, vec4_Normalize)  \
    )((x))

#define vlerp(a, b, t)              \
    VMAP((a),                       \
        VBIND(vec2, vec2_Lerp),     \
        VBIND(vec3, vec3_Lerp),     \
        VBIND(vec4, vec4_Lerp),     \
        VBIND(default, scalar_Lerp) \
    )((a), (b), (t))

#define vequ(x, y)                       \
    VMAP((x),                            \
        VBIND(vec2, vec2_AlmostTheSame), \
        VBIND(vec3, vec3_AlmostTheSame), \
        VBIND(vec4, vec4_AlmostTheSame), \
        VBIND(default, scalar_Equal)     \
    )((x), (y))

#define vtrans(x)                        \
    VMAP((x),                            \
        VBIND(mat2x2, mat2x2_Transpose), \
        VBIND(mat3x3, mat3x3_Transpose), \
        VBIND(mat4x4, mat4x4_Transpose)  \
    )((x))

// TODO: for all types
#define vmax(x, y)                 \
    VMAP((x),                      \
        VBIND(default, scalar_Max) \
    )((x), (y))

#define vmin(x, y)                 \
    VMAP((x),                      \
        VBIND(default, scalar_Min) \
    )((x), (y))

// TODO: vsqrt, vrsqrt

#define vclamp(t, t_min, t_max) vmax(vmin(t, t_max), t_min)
#define vneg(x) (vmul(-1.0f, (x)))

#define VMATH_OVERLOAD(IGNORE1, IGNORE2, IGNORE3, INGORE4, IGNORE5, IGNORE6, IGNORE7, NAME, ...) NAME

#define VSUM_2(v1, v2)                         vadd((v1), (v2))
#define VSUM_3(v1, v2, v3)                     VSUM_2(VSUM_2((v1), (v2)), (v3))
#define VSUM_4(v1, v2, v3, v4)                 VSUM_2(VSUM_2((v1), (v2)), VSUM_2((v3), (v4)))
#define VSUM_5(v1, v2, v3, v4, v5)             VSUM_2(VSUM_4((v1), (v2), (v3), (v4)), (v5))
#define VSUM_6(v1, v2, v3, v4, v5, v6)         VSUM_2(VSUM_4((v1), (v2), (v3), (v4)), VSUM_2((v5), (v6)))
#define VSUM_7(v1, v2, v3, v4, v5, v6, v7)     VSUM_2(VSUM_4((v1), (v2), (v3), (v4)), VSUM_2((v5), (v6)), (v7))
#define VSUM_8(v1, v2, v3, v4, v5, v6, v7, v8) VSUM_2(VSUM_4((v1), (v2), (v3), (v4)), VSUM_4((v5), (v6), (v7), (v8)))

#define vsum(v1, ...)                                                                      \
    VMATH_OVERLOAD(__VA_ARGS__, VSUM_8, VSUM_7, VSUM_6, VSUM_5, VSUM_4, VSUM_3, VSUM_2, _) \
    ((v1), __VA_ARGS__)

#define VPROD_2(v1, v2)                         vmul((v1), (v2))
#define VPROD_3(v1, v2, v3)                     VPROD_2(VPROD_2((v1), (v2)), (v3))
#define VPROD_4(v1, v2, v3, v4)                 VPROD_2(VPROD_2((v1), (v2)), VPROD_2((v3), (v4)))
#define VPROD_5(v1, v2, v3, v4, v5)             VPROD_2(VPROD_4((v1), (v2), (v3), (v4)), (v5))
#define VPROD_6(v1, v2, v3, v4, v5, v6)         VPROD_2(VPROD_4((v1), (v2), (v3), (v4)), VPROD_2((v5), (v6)))
#define VPROD_7(v1, v2, v3, v4, v5, v6, v7)     VPROD_2(VPROD_4((v1), (v2), (v3), (v4)), VPROD_2((v5), (v6)), (v7))
#define VPROD_8(v1, v2, v3, v4, v5, v6, v7, v8) VPROD_2(VPROD_4((v1), (v2), (v3), (v4)), VPROD_4((v5), (v6), (v7), (v8)))

#define vprod(v1, ...)                                                                            \
    VMATH_OVERLOAD(__VA_ARGS__, VPROD_8, VPROD_7, VPROD_6, VPROD_5, VPROD_4, VPROD_3, VPROD_2, _) \
    ((v1), __VA_ARGS__)

#define VMATH_VEC2_XY(x_, y_) ((vec2){.x = x_, .y = y_})
#define VMATH_VEC2_SET(t_)    ((vec2){.x = t_, .y = t_})
#define vec2(x, ...) \
    VMATH_OVERLOAD(__VA_ARGS__, _, _, _, _, _, _, VMATH_VEC2_XY, VMATH_VEC2_SET)((x), __VA_ARGS__)

#define VMATH_VEC3_XYZ(x_, y_, z_) ((vec3){.x = x_, .y = y_, .z = z_})
#define VMATH_VEC3_SET(t_)         ((vec3){.x = t_, .y = t_, .z = t_})
#define vec3(x, ...) \
    VMATH_OVERLOAD(__VA_ARGS__, _, _, _, _, _, VMATH_VEC3_XYZ, _, VMATH_VEC2_SET)((x), __VA_ARGS__)

#define VMATH_VEC4_XYZW(x_, y_, z_, w_) ((vec4){.x = x_, .y = y_, .z = z_, .w = w_})
#define VMATH_VEC4_SET(t_)              ((vec4){.x = t_, .y = t_, .z = t_, .w = t_})
#define vec4(x, ...) \
    VMATH_OVERLOAD(__VA_ARGS__, _, _, _, _, VMATH_VEC4_XYZW, _, _, VMATH_VEC2_SET)((x), __VA_ARGS__)

#define mat2x2(m11, m12, m21, m22) \
    ((mat2x2) {                    \
        .X.x = m11, .Y.x = m12,    \
        .X.y = m21, .Y.y = m22,    \
    })

#define mat3x3(m11, m12, m13, m21, m22, m23, m31, m32, m33) \
    ((mat3x3) {                                             \
        .X.x = m11, .Y.x = m12, .Z.x = m13,                 \
        .X.y = m21, .Y.y = m22, .Z.y = m23,                 \
        .X.z = m31, .Y.z = m32, .Z.z = m33,                 \
    })

#define mat4x4(m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34, m41, m42, m43, m44) \
    ((mat4x4) { \
        .X.x = m11, .Y.x = m12, .Z.x = m13, .W.x = m14, \
        .X.y = m21, .Y.y = m22, .Z.y = m23, .W.y = m24, \
        .X.z = m31, .Y.z = m32, .Z.z = m33, .W.z = m34, \
        .X.w = m41, .Y.w = m42, .Z.w = m43, .W.w = m44, \
    })

/* ---- Types ---- */

typedef enum {
    AXIS_X = 0,
    AXIS_U = AXIS_X,
    AXIS_R = AXIS_X,

    AXIS_Y = 1,
    AXIS_V = AXIS_Y,
    AXIS_THETA = AXIS_Y,

    AXIS_Z = 2,
    AXIS_PHI = AXIS_Z,

    AXIS_W = 3,
    AXIS_LAST
} Axis;

typedef union {
    struct {
        float x, y;
    };

    struct {
        float u, v;
    };

    struct {
        float r, theta;
    };

    struct {
        float width, height;
    };

    float elem[2];
} vec2;

typedef union {
    struct {
        float x, y, z;
    };

    struct {
        float u, v, w;
    };

    struct {
        float r, g, b;
    };

    struct {
        float rho, theta, phi;
    };

    struct {
        vec2 xy;
        uint32_t : 32;
    };

    struct {
        uint32_t : 32;
        vec2 yz;
    };

    struct {
        vec2 uv;
        uint32_t : 32;
    };

    struct {
        uint32_t : 32;
        vec2 vw;
    };

    struct {
        float length, width, height;
    };

    float elem[3];
} vec3;

typedef union {
    struct {
        float x, y, z, w;
    };

    struct {
        vec2 xy, zw;
    };

    struct {
        uint32_t : 32;
        vec2 yz;
        uint32_t : 32;
    };

    struct {
        vec3 xyz;
        uint32_t : 32;
    };

    struct {
        uint32_t : 32;
        vec3 yzw;
    };

    struct {
        float r, g, b, a;
    };

    struct {
        vec3 rgb;
        uint32_t : 32;
    };

    float elems[4];

#if VMATH_SIMD > VMATH_SIMD_NONE
    __m128 m128;
#endif
} VMATH_ALIGN16 vec4;

typedef vec2 point2;
typedef vec3 point3;
typedef vec4 point4;

typedef union {
    struct {
        vec2 X, Y;
    };

    float elems[2][2];

    vec2 cols[2];
} mat2x2;

typedef union {
    struct {
        vec3 X, Y, Z;
    };

    // elems[col][row]
    float elems[3][3];

    vec3 cols[3];
} mat3x3;

typedef union {
    struct {
        vec4 X, Y, Z;
        union { vec4 W; vec4 T; };
    };

    // elems[col][row]
    float elems[4][4];

    vec4 cols[4];

#if VMATH_SIMD > VMATH_SIMD_NONE
    __m128 m128[4];
#endif

#if VMATH_SIMD > VMATH_SIMD_SSE
    __m256 m256[2];
#endif
} VMATH_ALIGN32 mat4x4;

/* ---- Functions ---- */

/* --- scalar --- */

static inline float scalar_Min(float x, float y);
static inline float scalar_Max(float x, float y);
static inline float scalar_Clamp(float x, float min, float max);
static inline float scalar_Radians(float degrees);
static inline float scalar_Degrees(float radians);
static inline float scalar_Lerp(float a, float b, float t);
static inline bool  scalar_Equal(float a, float b);
static inline float scalar_Multiply(float x, float y);
static inline float scalar_Divide(float x, float y);
static inline float scalar_Add(float x, float y);
static inline float scalar_Subtract(float x, float y);
static inline float scalar_Magnitude(float t);
static inline float scalar_MagnitudeSquared(float t);

/* --- vec2 --- */

static inline vec2  vec2_Add(vec2 v1, vec2 v2);
static inline vec2  vec2_Subtract(vec2 v1, vec2 v2);
static inline vec2  vec2_MultiplyComponents(vec2 v1, vec2 v2);
static inline vec2  vec2_MultiplyScalar(vec2 vec, float scalar);
static inline vec2  vec2_MultiplyScalarR(float scalar, vec2 vec);
static inline vec2  vec2_DivideComponents(vec2 vdividend, vec2 vdivisor);
static inline vec2  vec2_DivideScalar(vec2 vec, float scalar);
static inline float vec2_DotProduct(vec2 v1, vec2 v2);
static inline vec3  vec2_CrossProduct(vec2 v1, vec2 v2);
static inline float vec2_Magnitude(vec2 vec);
static inline float vec2_MagnitudeSquared(vec2 vec);
static inline vec2  vec2_Normalize(vec2 vec);
static inline vec2  vec2_Lerp(vec2 v1, vec2 v2, float t);
static inline bool  vec2_AlmostTheSame(vec2 v1, vec2 v2);
static inline vec2  vec2_CartesianToPolar(vec2 cartesian);
static inline vec2  vec2_PolarToCartesian(vec2 polar);

/* --- vec3 --- */

static inline vec3  vec3_Add(vec3 v1, vec3 v2);
static inline vec3  vec3_Subtract(vec3 v1, vec3 v2);
static inline vec3  vec3_MultiplyComponents(vec3 v1, vec3 v2);
static inline vec3  vec3_MultiplyScalar(vec3 vec, float scalar);
static inline vec3  vec3_MultiplyScalarR(float scalar, vec3 vec);
static inline vec3  vec3_DivideComponents(vec3 vdividend, vec3 vdivisor);
static inline vec3  vec3_DivideScalar(vec3 vec, float scalar);
static inline float vec3_DotProduct(vec3 v1, vec3 v2);
static inline vec3  vec3_CrossProduct(vec3 v1, vec3 v2);
static inline float vec3_Magnitude(vec3 vec);
static inline float vec3_MagnitudeSquared(vec3 vec);
static inline vec3  vec3_Normalize(vec3 vec);
static inline vec3  vec3_Lerp(vec3 v1, vec3 v2, float t);
static inline bool  vec3_AlmostTheSame(vec3 v1, vec3 v2);
static inline vec3  vec3_Reflect(vec3 vec, vec3 normal);
static inline vec3  vec3_Refract(vec3 vec, vec3 normal, float refractRatio);
static inline vec3  vec3_CartesianToSpherical(vec3 cartesian);
static inline vec3  vec3_SphericalToCartesian(vec3 spherical);

/* --- vec4 --- */

static inline vec4  vec4_Add(vec4 v1, vec4 v2);
static inline vec4  vec4_Subtract(vec4 v1, vec4 v2);
static inline vec4  vec4_MultiplyComponents(vec4 v1, vec4 v2);
static inline vec4  vec4_MultiplyScalar(vec4 vec, float scalar);
static inline vec4  vec4_MultiplyScalarR(float scalar, vec4 vec);
static inline vec4  vec4_DivideComponents(vec4 vdividend, vec4 vdivisor);
static inline vec4  vec4_DivideScalar(vec4 vec, float scalar);
static inline float vec4_DotProduct(vec4 v1, vec4 v2);
static inline vec4  vec4_CrossProduct(vec4 v1, vec4 v2);
static inline float vec4_Magnitude(vec4 vec);
static inline float vec4_MagnitudeSquared(vec4 vec);
static inline vec4  vec4_Normalize(vec4 vec);
static inline vec4  vec4_Lerp(vec4 v1, vec4 v2, float t);
static inline bool  vec4_AlmostTheSame(vec4 v1, vec4 v2);

/* --- mat2x2 --- */

static inline vec2   mat2x2_MultiplyVector(mat2x2 M, vec2 V);
static inline mat2x2 mat2x2_MultiplyMatrix(mat2x2 left, mat2x2 right);
static inline mat2x2 mat2x2_MultiplyScalar(mat2x2 M, float t);
static inline mat2x2 mat2x2_MultiplyScalarR(float t, mat2x2 M);
static inline mat2x2 mat2x2_Add(mat2x2 M1, mat2x2 M2);
static inline mat2x2 mat2x2_Subtract(mat2x2 M1, mat2x2 M2);
static inline mat2x2 mat2x2_DivideScalar(mat2x2 M, float t);
static inline mat2x2 mat2x2_Transpose(mat2x2 M);

/* --- mat3x3 --- */

static inline vec3   mat3x3_MultiplyVector(mat3x3 M, vec3 V);
static inline mat3x3 mat3x3_MultiplyMatrix(mat3x3 left, mat3x3 right);
static inline mat3x3 mat3x3_MultiplyScalar(mat3x3 M, float t);
static inline mat3x3 mat3x3_MultiplyScalarR(float t, mat3x3 M);
static inline mat3x3 mat3x3_Add(mat3x3 M1, mat3x3 M2);
static inline mat3x3 mat3x3_Subtract(mat3x3 M1, mat3x3 M2);
static inline mat3x3 mat3x3_DivideScalar(mat3x3 M, float t);
static inline mat3x3 mat3x3_Transpose(mat3x3 M);

/* --- mat4x4 --- */

static inline vec4   mat4x4_MultiplyVector(mat4x4 M, vec4 V);
static inline mat4x4 mat4x4_MultiplyMatrix(mat4x4 left, mat4x4 right);
static inline mat4x4 mat4x4_MultiplyScalar(mat4x4 M, float t);
static inline mat4x4 mat4x4_MultiplyScalarR(float t, mat4x4 M);
static inline mat4x4 mat4x4_Add(mat4x4 M1, mat4x4 M2);
static inline mat4x4 mat4x4_Subtract(mat4x4 M1, mat4x4 M2);
static inline mat4x4 mat4x4_DivideScalar(mat4x4 M, float t);
static inline mat4x4 mat4x4_Transpose(mat4x4 M);

/* ---- Scalar Functions ---- */

static inline float scalar_Min(float x, float y)
{
    return x < y ? x : y;
}

static inline float scalar_Max(float x, float y)
{
    return x > y ? x : y;
}

static inline float scalar_Clamp(float x, float min, float max)
{
    return vmax(vmin(x, max), min);
}

static inline float scalar_Add(float x, float y)
{
    return x + y;
}

static inline float scalar_Subtract(float x, float y)
{
    return x - y;
}

static inline float scalar_Multiply(float x, float y)
{
    return x * y;
}

static inline float scalar_Divide(float x, float y)
{
     return x / y;
}

static inline float scalar_Lerp(float a, float b, float t)
{
    return a + t * (b - a);
}

static inline bool scalar_Equal(float a, float b)
{
    float absA = vmag(a);
    float absB = vmag(b);
    float diff = vmag(a - b);

    if (a == b) { // shortcut, handles infinities
        return true;
    } else if (a == 0.0f || b == 0.0f || (absA + absB < FLT_MIN)) {
        // a or b is zero or both are extremely close to it
        // relative error is less meaningful here
        return diff < (EPSILON32 * FLT_MIN);
    } else { // use relative error
        return (diff / vmin((absA + absB), FLT_MAX)) < EPSILON32;
    }
}

static inline float scalar_Magnitude(float t)
{
    return t > 0.0f ? t : -t;
}

static inline float scalar_MagnitudeSquared(float t)
{
    return t * t;
}

/* --- Vec2 Functions --- */

static inline vec2 vec2_Add(vec2 v1, vec2 v2)
{
    return (vec2) {
        .x = v1.x + v2.x,
        .y = v1.y + v2.y,
    };
}

static inline vec2 vec2_Subtract(vec2 v1, vec2 v2)
{
    return (vec2) {
        .x = v1.x - v2.x,
        .y = v1.y - v2.y,
    };
}

static inline vec2 vec2_MultiplyScalar(vec2 vec, float scalar)
{
    return (vec2) {
        .x = scalar * vec.x,
        .y = scalar * vec.y,
    };
}

static inline vec2 vec2_MultiplyScalarR(float scalar, vec2 vec)
{
    return vec2_MultiplyScalar(vec, scalar);
}

static inline vec2 vec2_DivideScalar(vec2 vec, float scalar)
{
    return vec2_MultiplyScalar(vec, 1.0f / scalar);
}

static inline vec2 vec2_MultiplyComponents(vec2 v1, vec2 v2)
{
    return (vec2) {
        .x = v1.x * v2.x,
        .y = v1.y * v2.y,
    };
}

static inline vec2 vec2_DivideComponents(vec2 vdividend, vec2 vdivisor)
{
    return (vec2) {
        .x = vdividend.x / vdivisor.x,
        .y = vdividend.y / vdivisor.y,
    };
}

static inline float vec2_DotProduct(vec2 v1, vec2 v2)
{
    return v1.x * v2.x + v1.y * v2.y;
}

static inline vec3 vec2_CrossProduct(vec2 v1, vec2 v2)
{
    return (vec3) {
        .z = (v1.x * v2.y - v1.y * v2.x),
    };
}

static inline float vec2_MagnitudeSquared(vec2 vec)
{
    return vec2_DotProduct(vec, vec);
}

static inline float vec2_Magnitude(vec2 vec)
{
    return sqrtf(vec2_MagnitudeSquared(vec));
}

static inline vec2 vec2_Normalize(vec2 vec)
{
    return vec2_DivideScalar(vec, vec2_Magnitude(vec));
}

static inline vec2 vec2_Lerp(vec2 v1, vec2 v2, float t)
{
    return vec2_Add(v1, vec2_MultiplyScalar(vec2_Subtract(v2, v1), t));
}

static inline bool vec2_AlmostTheSame(vec2 v1, vec2 v2)
{
    return vequ(v1.x, v2.x) && vequ(v1.y, v2.y);
}

static inline vec2 vec2_PolarToCartesian(vec2 polar)
{
    return (vec2) {
        .x = polar.r * cosf(polar.theta),
        .y = polar.r * sinf(polar.theta),
    };
}

static inline vec2 vec2_CartesianToPolar(vec2 cartesian)
{
    return (vec2) {
        .r = vec2_Magnitude(cartesian),
        .theta = atan2f(cartesian.y, cartesian.x),
    };
}

/* --- Vec3 Functions --- */

static inline vec3 vec3_Add(vec3 v1, vec3 v2)
{
    return (vec3) {
        .x = v1.x + v2.x,
        .y = v1.y + v2.y,
        .z = v1.z + v2.z,
    };
}

static inline vec3 vec3_Subtract(vec3 v1, vec3 v2)
{
    return (vec3) {
        .x = v1.x - v2.x,
        .y = v1.y - v2.y,
        .z = v1.z - v2.z,
    };
}

static inline vec3 vec3_MultiplyScalar(vec3 vec, float scalar)
{
    return (vec3) {
        .x = scalar * vec.x,
        .y = scalar * vec.y,
        .z = scalar * vec.z,
    };
}

static inline vec3 vec3_MultiplyComponents(vec3 v1, vec3 v2)
{
    return (vec3) {
        .x = v1.x * v2.x,
        .y = v1.y * v2.y,
        .z = v1.z * v2.z,
    };
}

static inline vec3 vec3_DivideComponents(vec3 vdividend, vec3 vdivisor)
{
    return (vec3) {
        .x = vdividend.x / vdivisor.x,
        .y = vdividend.y / vdivisor.y,
        .z = vdividend.z / vdivisor.z,
    };
}

static inline float vec3_DotProduct(vec3 v1, vec3 v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

static inline vec3 vec3_CrossProduct(vec3 v1, vec3 v2)
{
    return (vec3) {
        .x = (v1.y * v2.z - v1.z * v2.y),
        .y = (v1.z * v2.x - v1.x * v2.z),
        .z = (v1.x * v2.y - v1.y * v2.x),
    };
}

static inline vec3 vec3_MultiplyScalarR(float scalar, vec3 vec)
{
    return vec3_MultiplyScalar(vec, scalar);
}

static inline vec3 vec3_DivideScalar(vec3 vec, float scalar)
{
    return vec3_MultiplyScalar(vec, 1.0f / scalar);
}

static inline vec3 vec3_Lerp(vec3 v1, vec3 v2, float t)
{
    return vec3_Add(v1, vec3_MultiplyScalar(vec3_Subtract(v2, v1), t));
}

static inline float vec3_MagnitudeSquared(vec3 vec)
{
    return vec3_DotProduct(vec, vec);
}

static inline float vec3_Magnitude(vec3 vec)
{
    return sqrtf(vec3_MagnitudeSquared(vec));
}

static inline vec3 vec3_Normalize(vec3 vec)
{
    return vec3_DivideScalar(vec, vec3_Magnitude(vec));
}

static inline bool vec3_AlmostTheSame(vec3 v1, vec3 v2)
{
    return vequ(v1.x, v2.x) && vequ(v1.y, v2.y) && vequ(v1.z, v2.z);
}

static inline vec3 vec3_SphericalToCartesian(vec3 spherical)
{
    return (vec3) {
        .x = spherical.rho * sinf(spherical.theta) * cosf(spherical.phi),
        .y = spherical.rho * sinf(spherical.theta) * sinf(spherical.phi),
        .z = spherical.rho * cosf(spherical.theta),
    };
}

static inline vec3 vec3_CartesianToSpherical(vec3 cartesian)
{
    return (vec3) {
        .rho   = vec3_Magnitude(cartesian),
        .theta = acosf(cartesian.z / vec3_Magnitude(cartesian)),
        .phi   = atan2f(cartesian.y, cartesian.x),
    };
}

/* --- Vec4 Functions --- */

static inline vec4 vec4_Add(vec4 v1, vec4 v2)
{
    return (vec4) {
        .x = v1.x + v2.x,
        .y = v1.y + v2.y,
        .z = v1.z + v2.z,
        .w = v1.w + v2.w,
    };
}

static inline vec4 vec4_Subtract(vec4 v1, vec4 v2)
{
    return (vec4) {
        .x = v1.x - v2.x,
        .y = v1.y - v2.y,
        .z = v1.z - v2.z,
        .w = v1.w - v2.w,
    };
}

static inline vec4 vec4_MultiplyScalar(vec4 vec, float scalar)
{
    return (vec4) {
        .x = scalar * vec.x,
        .y = scalar * vec.y,
        .z = scalar * vec.z,
        .w = scalar * vec.w,
    };
}

static inline vec4 vec4_MultiplyScalarR(float scalar, vec4 vec)
{
    return vec4_MultiplyScalar(vec, scalar);
}

static inline vec4 vec4_MultiplyComponents(vec4 v1, vec4 v2)
{
    return (vec4) {
        .x = v1.x * v2.x,
        .y = v1.y * v2.y,
        .z = v1.z * v2.z,
        .w = v1.w * v2.w,
    };
}

static inline vec4 vec4_DivideScalar(vec4 vec, float scalar)
{
    return vec4_MultiplyScalar(vec, 1.0f / scalar);
}

static inline vec4 vec4_DivideComponents(vec4 vdividend, vec4 vdivisor)
{
    return (vec4) {
        .x = vdividend.x / vdivisor.x,
        .y = vdividend.y / vdivisor.y,
        .z = vdividend.z / vdivisor.z,
        .w = vdividend.w / vdivisor.w,
    };
}

static inline vec4 vec4_Lerp(vec4 v1, vec4 v2, float t)
{
    return vec4_Add(v1, vec4_MultiplyScalar(vec4_Subtract(v2, v1), t));
}

static inline float vec4_DotProduct(vec4 v1, vec4 v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

static inline float vec4_MagnitudeSquared(vec4 vec)
{
    return vec4_DotProduct(vec, vec);
}

static inline float vec4_Magnitude(vec4 vec)
{
    return sqrtf(vec4_MagnitudeSquared(vec));
}

static inline vec4 vec4_Normalize(vec4 vec)
{
    return vec4_DivideScalar(vec, vec4_Magnitude(vec));
}

static inline bool vec4_AlmostTheSame(vec4 v1, vec4 v2)
{
    return vequ(v1.x, v2.x) && vequ(v1.y, v2.y) && vequ(v1.z, v2.z) && vequ(v1.w, v2.w);
}

/* ---- 2x2 Matrix Functions ---- */

static inline vec2 mat2x2_MultiplyVector(mat2x2 M, vec2 V)
{
    return vec2(
        M.X.x * V.x + M.Y.x * V.y,
        M.X.y * V.x + M.Y.y * V.y
    );
}

static inline mat2x2 mat2x2_MultiplyMatrix(mat2x2 left, mat2x2 right)
{
    return (mat2x2){
        .X = mat2x2_MultiplyVector(left, right.X),
        .Y = mat2x2_MultiplyVector(left, right.Y),
    };
}

static inline mat2x2 mat2x2_Transpose(mat2x2 M)
{
    return mat2x2(
        M.X.x, M.X.y,
        M.Y.x, M.Y.y
    );
}

static inline mat2x2 mat2x2_MultiplyScalar(mat2x2 M, float t)
{
    return (mat2x2){
        .X = vec2_MultiplyScalar(M.X, t),
        .Y = vec2_MultiplyScalar(M.Y, t),
    };
}

static inline mat2x2 mat2x2_MultiplyScalarR(float t, mat2x2 m)
{
    return mat2x2_MultiplyScalar(m, t);
}

static inline mat2x2 mat2x2_Add(mat2x2 M1, mat2x2 M2)
{
    return (mat2x2){
        .X = vec2_Add(M1.X, M2.X),
        .Y = vec2_Add(M1.Y, M2.Y),
    };
}

static inline mat2x2 mat2x2_Subtract(mat2x2 M1, mat2x2 M2)
{
    return (mat2x2){
        .X = vec2_Subtract(M1.X, M2.X),
        .Y = vec2_Subtract(M1.Y, M2.Y),
    };
}

static inline mat2x2 mat2x2_DivideScalar(mat2x2 M, float t)
{
    return mat2x2_MultiplyScalar(M, 1.0f / t);
}

/* ---- 3x3 Matrix Functions ---- */

static inline vec3 mat3x3_MultiplyVector(mat3x3 M, vec3 V)
{
    return vec3(
        M.X.x * V.x + M.Y.x * V.y + M.Z.x * V.z,
        M.X.y * V.x + M.Y.y * V.y + M.Z.y * V.z,
        M.X.z * V.x + M.Y.z * V.y + M.Z.z * V.z
    );
}

static inline mat3x3 mat3x3_MultiplyMatrix(mat3x3 left, mat3x3 right)
{
    return (mat3x3){
        .X = mat3x3_MultiplyVector(left, right.X),
        .Y = mat3x3_MultiplyVector(left, right.Y),
        .Z = mat3x3_MultiplyVector(left, right.Z),
    };
}

static inline mat3x3 mat3x3_Transpose(mat3x3 M)
{
    return mat3x3(
        M.X.x, M.X.y, M.X.z,
        M.Y.x, M.Y.y, M.Y.z,
        M.Z.x, M.Z.y, M.Z.z
    );
}

static inline mat3x3 mat3x3_MultiplyScalar(mat3x3 M, float t)
{
    return (mat3x3){
        .X = vec3_MultiplyScalar(M.X, t),
        .Y = vec3_MultiplyScalar(M.Y, t),
        .Z = vec3_MultiplyScalar(M.Z, t),
    };
}

static inline mat3x3 mat3x3_MultiplyScalarR(float t, mat3x3 M)
{
    return mat3x3_MultiplyScalar(M, t);
}

static inline mat3x3 mat3x3_Add(mat3x3 M1, mat3x3 M2)
{
    return (mat3x3){
        .X = vec3_Add(M1.X, M2.X),
        .Y = vec3_Add(M1.Y, M2.Y),
        .Z = vec3_Add(M1.Z, M2.Z),
    };
}

static inline mat3x3 mat3x3_Subtract(mat3x3 M1, mat3x3 M2)
{
    return (mat3x3){
        .X = vec3_Subtract(M1.X, M2.X),
        .Y = vec3_Subtract(M1.Y, M2.Y),
        .Z = vec3_Subtract(M1.Z, M2.Z),
    };
}

static inline mat3x3 mat3x3_DivideScalar(mat3x3 M, float t)
{
    return mat3x3_MultiplyScalar(M, 1.0f / t);
}

/* ---- 4x4 Matrix Functions ---- */

#if VMATH_SIMD == VMATH_SIMD_AVX2
static inline vec4 mat4x4_MultiplyVector(mat4x4 M, vec4 V)
{
    __m128 xx, yy, zz, ww;
    __m128 tx, txy, txyz, txyzw;

    xx = _mm_set1_ps(V.x);
    tx = _mm_mul_ps(M.X.m128, xx);

    yy = _mm_set1_ps(V.y);
    txy = _mm_fmadd_ps(M.Y.m128, yy, tx);

    zz = _mm_set1_ps(V.z);
    txyz = _mm_fmadd_ps(M.Z.m128, zz, txy);

    ww = _mm_set1_ps(V.w);
    txyzw = _mm_fmadd_ps(M.W.m128, ww, txyz);

    return (vec4){.m128 = txyzw};
}
#elif VMATH_SIMD == VMATH_SIMD_SSE
static inline vec4 mat4x4_MultiplyVector(mat4x4 M, vec4 V)
{
    __m128 xx, yy, zz, ww;
    __m128 tx, ty, tz, tw;
    __m128 txyzw;

    xx = _mm_set1_ps(V.x);
    tx = _mm_mul_ps(M.X.m128, xx);

    yy = _mm_set1_ps(V.y);
    ty = _mm_mul_ps(M.Y.m128, yy);

    zz = _mm_set1_ps(V.z);
    tz = _mm_mul_ps(M.Z.m128, zz);

    ww = _mm_set1_ps(V.w);
    tw = _mm_mul_ps(M.W..m128, ww);

    txyzw = _mm_add_ps(_mm_add_ps(tx, ty), _mm_add_ps(tz, tw));

    return (vec4){.m128 = txyzw};
}
#elif VMATH_SIMD == VMATH_SIMD_NONE
static inline vec4 mat4x4_MultiplyVector(mat4x4 M, vec4 V)
{
    return vec4(
        M.X.x * V.x + M.Y.x * V.y + M.Z.x * V.z + M.W.x * V.w,
        M.X.y * V.x + M.Y.y * V.y + M.Z.y * V.z + M.W.y * V.w,
        M.X.z * V.x + M.Y.z * V.y + M.Z.z * V.z + M.W.z * V.w,
        M.X.w * V.x + M.Y.w * V.y + M.Z.w * V.z + M.W.w * V.w
    );
}
#endif

static inline mat4x4 mat4x4_MultiplyMatrix(mat4x4 left, mat4x4 right)
{
    return (mat4x4){
        .X = mat4x4_MultiplyVector(left, right.X),
        .Y = mat4x4_MultiplyVector(left, right.Y),
        .Z = mat4x4_MultiplyVector(left, right.Z),
        .W = mat4x4_MultiplyVector(left, right.W),
    };
}

#if VMATH_SIMD == VMATH_SIMD_AVX2
static inline mat4x4 mat4x4_Transpose(mat4x4 M)
{
    __m256i indices = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    __m256 xy = M.m256[0];
    __m256 zw = M.m256[1];

    __m256 xyzw_xy = _mm256_unpacklo_ps(xy, zw);
    __m256 xyzw_zw = _mm256_unpackhi_ps(xy, zw);

    return (mat4x4){
        .m256 = {
            _mm256_permutevar8x32_ps(xyzw_xy, indices),
            _mm256_permutevar8x32_ps(xyzw_zw, indices),
        },
    };
}
#elif VMATH_SIMD == VMATH_SIMD_SSE
static inline mat4x4 mat4x4_Transpose(mat4x4 M)
{
    __m128 xy_xy = _mm_unpacklo_ps(M.X.m128, M.Y.m128);
    __m128 xy_zw = _mm_unpackhi_ps(M.X.m128, M.Y.m128);

    __m128 zw_xy = _mm_unpacklo_ps(M.Z.m128, M.W.m128);
    __m128 zw_zw = _mm_unpackhi_ps(M.Z.m128, M.W.m128);

    return (mat4x4){
        .X.m128 = _mm_shuffle_ps(xy_xy, zw_xy, _MM_SHUFFLE(1, 0, 1, 0)),
        .Y.m128 = _mm_shuffle_ps(xy_xy, zw_xy, _MM_SHUFFLE(3, 2, 3, 2)),
        .Z.m128 = _mm_shuffle_ps(xy_zw, zw_zw, _MM_SHUFFLE(1, 0, 1, 0)),
        .W.m128 = _mm_shuffle_ps(xy_zw, zw_zw, _MM_SHUFFLE(3, 2, 3, 2)),
    };
}
#else
static inline mat4x4 mat4x4_Transpose(mat4x4 M)
{
    return mat4x4(
        M.X.x, M.X.y, M.X.z, M.X.w,
        M.Y.x, M.Y.y, M.Y.z, M.Y.w,
        M.Z.x, M.Z.y, M.Z.z, M.Z.w,
        M.W.x, M.W.y, M.W.z, M.W.w
    );
}
#endif

#if VMATH_SIMD == VMATH_SIMD_AVX2
static inline mat4x4 mat4x4_MultiplyScalar(mat4x4 M, float t)
{
    __m256 vt = _mm256_set1_ps(t);

    return (mat4x4){
        .m256 = {
            _mm256_mul_ps(vt, M.m256[0]),
            _mm256_mul_ps(vt, M.m256[1]),
        },
    };
}
#else
static inline mat4x4 mat4x4_MultiplyScalar(mat4x4 M, float t)
{
    return (mat4x4){
        .X = vec4_MultiplyScalar(M.x, t),
        .Y = vec4_MultiplyScalar(M.y, t),
        .Z = vec4_MultiplyScalar(M.z, t),
        .W = vec4_MultiplyScalar(M.w, t),
    };
}
#endif

static inline mat4x4 mat4x4_MultiplyScalarR(float t, mat4x4 M)
{
    return mat4x4_MultiplyScalar(M, t);
}

#if VMATH_SIMD == VMATH_SIMD_AVX2
static inline mat4x4 mat4x4_Add(mat4x4 M1, mat4x4 M2)
{
    return (mat4x4){
        .m256 = {
            _mm256_add_ps(M1.m256[0], M2.m256[0]),
            _mm256_add_ps(M1.m256[1], M2.m256[1]),
        },
    };
}
#else
static inline mat4x4 mat4x4_Add(mat4x4 M1, mat4x4 M2)
{
    return (mat4x4){
        .X = vec4_Add(M1.X, M2.X),
        .Y = vec4_Add(M1.Y, M2.Y),
        .Z = vec4_Add(M1.Z, M2.Z),
        .W = vec4_Add(M1.W, M2.W),
    };
}
#endif

#if VMATH_SIMD == VMATH_SIMD_AVX2
static inline mat4x4 mat4x4_Subtract(mat4x4 M1, mat4x4 M2)
{
    return (mat4x4){
        .m256 = {
            _mm256_sub_ps(M1.m256[0], M2.m256[0]),
            _mm256_sub_ps(M1.m256[1], M2.m256[1]),
        },
    };
}
#else
static inline mat4x4 mat4x4_Subtract(mat4x4 M1, mat4x4 M2)
{
    return (mat4x4){
        .x = vec4_Subtract(M1.x, M2.X),
        .y = vec4_Subtract(M1.y, M2.Y),
        .z = vec4_Subtract(M1.z, M2.Z),
        .w = vec4_Subtract(M1.w, M2.W),
    };
}
#endif

static inline mat4x4 mat4x4_DivideScalar(mat4x4 M, float t)
{
    return mat4x4_MultiplyScalar(M, 1.0f / t);
}

/* --- Misc Functions ---- */

static inline float Radians(float degrees)
{
    return degrees * PI32 / 180.0f;
}

static inline float Degrees(float radians)
{
    return radians * 180.0f / PI32;
}

static inline vec3 Reflect(vec3 V_in, vec3 normal)
{
    return vsub(V_in, vmul(normal, 2.0f * vdot(V_in, normal)));
}

// eta = IOR (outer) / IOR (inner)
static inline vec3 Refract(vec3 V_in, vec3 V_normal, float eta)
{
    float cos_theta = vmin(vdot(vmul(V_in, -1), V_normal), 1.0f);
    vec3 V_perp = vmul(vadd(V_in, vmul(V_normal, cos_theta)), eta);
    vec3 V_para = vmul(V_normal, -sqrtf(vmag(1.0f - vdot(V_perp, V_perp))));
    return vadd(V_perp, V_para);
}

// Creates an orthonormal basis with the argument as the Z basis vector
// See: https://backend.orbit.dtu.dk/ws/portalfiles/portal/126824972/onb_frisvad_jgt2012_v2.pdf
// WARNING: normal_in_z must be normalized
static inline mat3x3 OrthonormalBasis(vec3 normal_in_z)
{
    vec3 B_z = normal_in_z;
    vec3 B_x, B_y;
    if (VMATH_UNLIKELY(B_z.z < -0.9999999f)) {
        B_y = vec3(0.0f, -1.0f, 0.0f);
        B_x = vec3(-1.0f, 0.0f, 0.0f);
    } else {
        float a = 1.0f / (1.0f + B_z.z);
        float b = -B_z.x * B_z.y * a;

        B_y = vec3(-1.0f + B_z.x * B_z.x * a, -b, B_z.x);
        B_x = vec3(-b, -1.0f + B_z.y * B_z.y * a, B_z.y);
    }

    return (mat3x3) {
        .X = B_x,
        .Y = B_y,
        .Z = B_z,
    };
}

// computes the inverse matrix of a matrix of the form
// | 1 0 0 Tx |
// | 0 1 0 Ty |
// | 0 0 1 Tz |
// | 0 0 0  1 |
#if VMATH_SIMD > VMATH_SIMD_NONE
static inline mat4x4 InverseAffine_Translation(mat4x4 M) {
    return (mat4x4){
        .X = M.X,
        .Y = M.Y,
        .Z = M.Z,
        .T.m128 = _mm_sub_ps(_mm_setr_ps(0.0f, 0.0f, 0.0f, 2.0f), M.T.m128),
    };
}
#elif VMATH_SIMD == VMATH_SIMD_NONE
static inline mat4x4 InverseAffine_Translation(mat4x4 M) {
    return mat4x4(
        1, 0, 0, -M.T.x,
        0, 1, 0, -M.T.y,
        0, 0, 1, -M.T.z,
        0, 0, 0,      1
    );
}
#endif

// computes the inverse matrix of a matrix of the form
// | Xx Yx Zx Tx |
// | Xy Yy Zy Ty |
// | Xz Yz Zz Tz |
// |  0  0  0  1 |
// where |X| = |Y| = |Z| = 1
// and X.Y = 0, X.Z = 0, Y.Z = 0
#if VMATH_SIMD > VMATH_SIMD_NONE
static inline mat4x4 InverseAffine_TranslationRotation(mat4x4 M) {
    __m128 XY_xy = _mm_shuffle_ps(M.X.m128, M.Y.m128, _MM_SHUFFLE(1, 0, 1, 0)); // X.x X.y Y.x Y.y
    __m128 XY_zw = _mm_shuffle_ps(M.X.m128, M.Y.m128, _MM_SHUFFLE(3, 2, 3, 2)); // X.z X.w Y.z Y.w

    __m128 X_tr = _mm_shuffle_ps(XY_xy, M.Z.m128, _MM_SHUFFLE(3, 0, 2, 0)); // X.x Y.x Z.x 0
    __m128 Y_tr = _mm_shuffle_ps(XY_xy, M.Z.m128, _MM_SHUFFLE(3, 1, 3, 1)); // X.y Y.y Z.y 0
    __m128 Z_tr = _mm_shuffle_ps(XY_zw, M.Z.m128, _MM_SHUFFLE(3, 2, 2, 0)); // X.z Y.z Z.z 0

    __m128 T_x = _mm_set1_ps(M.T.x);
    __m128 T_y = _mm_set1_ps(M.T.y);
    __m128 T_z = _mm_set1_ps(M.T.z);

#if VMATH_SIMD == VMATH_SIMD_AVX2
    __m128 T = _mm_mul_ps(T_x, X_tr);
    T = _mm_fmadd_ps(T_y, Y_tr, T);
    T = _mm_fmadd_ps(T_z, Z_tr, T);
#elif VMATH_SIMD == VMATH_SIMD_SSE
    __m128 T = _mm_mul_ps(T_x, X_tr);
    T = _mm_add_ps(_mm_mul_ps(T_y, Y_tr), T);
    T = _mm_add_ps(_mm_mul_ps(T_z, Z_tr), T);
#endif

    T = _mm_sub_ps(_mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f), T);

    return (mat4x4){
        .X.m128 = X_tr,
        .Y.m128 = Y_tr,
        .Z.m128 = Z_tr,
        .T.m128 = T,
    };
}
#elif VMATH_SIMD == VMATH_SIMD_NONE
static inline mat4x4 InverseAffine_TranslationRotation(mat4x4 M) {
    return mat4x4(
        M.X.x, M.X.y, M.X.z, -vdot(M.T.xyz, M.X.xyz),
        M.Y.x, M.Y.y, M.Y.z, -vdot(M.T.xyz, M.Y.xyz),
        M.Z.x, M.Z.y, M.Z.z, -vdot(M.T.xyz, M.Z.xyz),
            0,     0,     0,                       1
    );
}
#endif

// computes the inverse matrix of a matrix of the form
// | a 0 0 Tx |
// | 0 b 0 Ty |
// | 0 0 c Tz |
// | 0 0 0  1 |
// where |X| = a, |Y| = b, |Z| = c
// and a != 0, b != 0, c != 0
#if VMATH_SIMD > VMATH_SIMD_NONE
static inline mat4x4 InverseAffine_TranslationScale(mat4x4 M) {
    __m128 X_tr = M.X.m128;
    __m128 Y_tr = M.Y.m128;
    __m128 Z_tr = M.Z.m128;

    // mag2 = X.x^2 Y.y^2 Z.z^2 1
    __m128 ones = _mm_set1_ps(1.0f);
    __m128 mag2 = _mm_mul_ps(X_tr, X_tr);
    mag2 = _mm_add_ps(mag2, _mm_mul_ps(Y_tr, Y_tr));
    mag2 = _mm_add_ps(mag2, _mm_mul_ps(Z_tr, Z_tr));
    mag2 = _mm_blend_ps(ones, mag2, VMATH_BLEND_MASK(1, 1, 1, 0));

    // inv_mag2 = 1/X.x^2 1/Y.y^2 1/Z.z^2 1
    __m128 inv_mag2 = _mm_div_ps(ones, mag2);

    X_tr = _mm_mul_ps(inv_mag2, X_tr);
    Y_tr = _mm_mul_ps(inv_mag2, Y_tr);
    Z_tr = _mm_mul_ps(inv_mag2, Z_tr);

    __m128 T = _mm_add_ps(X_tr, _mm_add_ps(Y_tr, Z_tr));
    T = _mm_mul_ps(T, M.T.m128);
    T = _mm_sub_ps(_mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f), T);

    return (mat4x4){
        .X.m128 = X_tr,
        .Y.m128 = Y_tr,
        .Z.m128 = Z_tr,
        .T.m128 = T,
    };
}
#elif VMATH_SIMD == VMATH_SIMD_NONE
static inline mat4x4 InverseAffine_TranslationScale(mat4x4 M) {
    float inv_mx2 = 1.0f / (M.X.x * M.X.x);
    float inv_my2 = 1.0f / (M.Y.y * M.Y.y);
    float inv_mz2 = 1.0f / (M.Z.z * M.Z.z);

    return mat4x4(
        inv_mx2 * M.X.x,           0,               0, inv_mx2 * -(M.T.x * M.X.x),
        0,           inv_my2 * M.Y.y,               0, inv_my2 * -(M.T.y * M.Y.y),
        0,                         0, inv_mz2 * M.Z.z, inv_mz2 * -(M.T.z * M.Z.z),
        0,                         0,               0,                          1
    );
}
#endif

// computes the inverse matrix of a matrix of the form
// | Xx Yx Zx Tx |
// | Xy Yy Zy Ty |
// | Xz Yz Zz Tz |
// |  0  0  0  1 |
// where |X| = a, |Y| = b, |Z| = c
// and a != 0, b != 0, c != 0
#if VMATH_SIMD > VMATH_SIMD_NONE
static inline mat4x4 InverseAffine_TranslationRotationScale(mat4x4 M) {
    // transpose
    __m128 XY_xy = _mm_shuffle_ps(M.X.m128, M.Y.m128, _MM_SHUFFLE(1, 0, 1, 0)); // X.x X.y Y.x Y.y
    __m128 XY_zw = _mm_shuffle_ps(M.X.m128, M.Y.m128, _MM_SHUFFLE(3, 2, 3, 2)); // X.z X.w Y.z Y.w

    __m128 X_tr = _mm_shuffle_ps(XY_xy, M.Z.m128, _MM_SHUFFLE(3, 0, 2, 0)); // X.x Y.x Z.x 0
    __m128 Y_tr = _mm_shuffle_ps(XY_xy, M.Z.m128, _MM_SHUFFLE(3, 1, 3, 1)); // X.y Y.y Z.y 0
    __m128 Z_tr = _mm_shuffle_ps(XY_zw, M.Z.m128, _MM_SHUFFLE(3, 2, 2, 0)); // X.z Y.z Z.z 0

    // mag2 = X.x^2 Y.y^2 Z.z^2 1
    __m128 ones = _mm_set1_ps(1.0f);
    __m128 mag2 = _mm_mul_ps(X_tr, X_tr);
    mag2 = _mm_add_ps(mag2, _mm_mul_ps(Y_tr, Y_tr));
    mag2 = _mm_add_ps(mag2, _mm_mul_ps(Z_tr, Z_tr));
    mag2 = _mm_blend_ps(ones, mag2, VMATH_BLEND_MASK(1, 1, 1, 0));

    // inv_mag2 = 1/X.x^2 1/Y.y^2 1/Z.z^2 1
    __m128 inv_mag2 = _mm_div_ps(ones, mag2);

    X_tr = _mm_mul_ps(inv_mag2, X_tr);
    Y_tr = _mm_mul_ps(inv_mag2, Y_tr);
    Z_tr = _mm_mul_ps(inv_mag2, Z_tr);

    __m128 T_x = _mm_set1_ps(M.T.x);
    __m128 T_y = _mm_set1_ps(M.T.y);
    __m128 T_z = _mm_set1_ps(M.T.z);

#if VMATH_SIMD == VMATH_SIMD_AVX2
    __m128 T = _mm_mul_ps(T_x, X_tr);
    T = _mm_fmadd_ps(T_y, Y_tr, T);
    T = _mm_fmadd_ps(T_z, Z_tr, T);
#elif VMATH_SIMD == VMATH_SIMD_SSE
    __m128 T = _mm_mul_ps(T_x, X_tr);
    T = _mm_add_ps(_mm_mul_ps(T_y, Y_tr), T);
    T = _mm_add_ps(_mm_mul_ps(T_z, Z_tr), T);
#endif

    T = _mm_sub_ps(_mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f), T);

    return (mat4x4){
        .X.m128 = X_tr,
        .Y.m128 = Y_tr,
        .Z.m128 = Z_tr,
        .T.m128 = T,
    };
}
#elif VMATH_SIMD == VMATH_SIMD_NONE
static inline mat4x4 InverseAffine_TranslationRotationScale(mat4x4 M) {
    float inv_mx2 = 1.0f / vmag2(M.X.xyz);
    float inv_my2 = 1.0f / vmag2(M.Y.xyz);
    float inv_mz2 = 1.0f / vmag2(M.Z.xyz);

    return mat4x4(
        inv_mx2 * M.X.x, inv_mx2 * M.X.y, inv_mx2 * M.X.z, inv_mx2 * -vdot(M.T.xyz, M.X.xyz),
        inv_my2 * M.Y.x, inv_my2 * M.Y.y, inv_my2 * M.Y.z, inv_my2 * -vdot(M.T.xyz, M.Y.xyz),
        inv_mz2 * M.Z.x, inv_mz2 * M.Z.y, inv_mz2 * M.Z.z, inv_mz2 * -vdot(M.T.xyz, M.Z.xyz),
                      0,               0,               0,                                 1
    );
}
#endif

// Converts a 3x3 matrix to a 4x4 matrix of the form
static inline mat4x4 HomogeneousMatrix(mat3x3 M)
{
    return (mat4x4){
        .X = {.xyz = M.X, .w = 0},
        .Y = {.xyz = M.Y, .w = 0},
        .Z = {.xyz = M.Z, .w = 0},
        .T = vec4(0, 0, 0, 1),
    };
}
