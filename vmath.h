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
#   if defined(__AVX2__)
#       define VMATH_SIMD VMATH_SIMD_AVX2
#   elif defined(__SSE__)
#       define VMATH_SIMD VMATH_SIMD_SSE
#   else
#       define VMATH_SIMD VMATH_SIMD_NONE
#   endif
#elif defined(_MSC_VER)
#   define VMATH_ALIGN16     __declspec(align(16))
#   define VMATH_ALIGN32     __declspec(align(32))
#   define VMATH_UNLIKELY(x) (x)
#   define VMATH_LIKELY(x)   (x)
#   if defined(__AVX2__)
#       define VMATH_SIMD VMATH_SIMD_AVX2
#   elif _M_IX86_FP >= 1
#       define VMATH_SIMD VMATH_SIMD_SSE
#   else
#       define VMATH_SIMD VMATH_SIMD_NONE
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

#define vmax(x, y)                 \
    VMAP((x),                      \
        VBIND(default, scalar_Max) \
    )((x), (y))

#define vmin(x, y)                 \
    VMAP((x),                      \
        VBIND(default, scalar_Min) \
    )((x), (y))

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
        .x.x = m11, .y.x = m12,    \
        .x.y = m21, .y.y = m22,    \
    })

#define mat3x3(m11, m12, m13, m21, m22, m23, m31, m32, m33) \
    ((mat3x3) {                                             \
        .x.x = m11, .y.x = m12, .z.x = m13,                 \
        .x.y = m21, .y.y = m22, .z.y = m23,                 \
        .x.z = m31, .y.z = m32, .z.z = m33,                 \
    })

#define mat4x4(m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34, m41, m42, m43, m44) \
    ((mat4x4) { \
        .x.x = m11, .y.x = m12, .z.x = m13, .w.x = m14, \
        .x.y = m21, .y.y = m22, .z.y = m23, .w.y = m24, \
        .x.z = m31, .y.z = m32, .z.z = m33, .w.z = m34, \
        .x.w = m41, .y.w = m42, .z.w = m43, .w.w = m44, \
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
        vec2 x, y;
    };

    vec2 cols[2];
} mat2x2;

typedef union {
    struct {
        vec3 x, y, z;
    };

    vec3 cols[3];
} mat3x3;

typedef union {
    struct {
        vec4 x, y, z, w;
    };

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

static inline vec2   mat2x2_MultiplyVector(mat2x2 m, vec2 v);
static inline mat2x2 mat2x2_MultiplyMatrix(mat2x2 left, mat2x2 right);
static inline mat2x2 mat2x2_MultiplyScalar(mat2x2 m, float t);
static inline mat2x2 mat2x2_MultiplyScalarR(float t, mat2x2 m);
static inline mat2x2 mat2x2_Add(mat2x2 m1, mat2x2 m2);
static inline mat2x2 mat2x2_Subtract(mat2x2 m1, mat2x2 m2);
static inline mat2x2 mat2x2_DivideScalar(mat2x2 m, float t);
static inline mat2x2 mat2x2_Transpose(mat2x2 m);

/* --- mat3x3 --- */

static inline vec3   mat3x3_MultiplyVector(mat3x3 m, vec3 v);
static inline mat3x3 mat3x3_MultiplyMatrix(mat3x3 left, mat3x3 right);
static inline mat3x3 mat3x3_MultiplyScalar(mat3x3 m, float t);
static inline mat3x3 mat3x3_MultiplyScalarR(float t, mat3x3 m);
static inline mat3x3 mat3x3_Add(mat3x3 m1, mat3x3 m2);
static inline mat3x3 mat3x3_Subtract(mat3x3 m1, mat3x3 m2);
static inline mat3x3 mat3x3_DivideScalar(mat3x3 m, float t);
static inline mat3x3 mat3x3_Transpose(mat3x3 m);

/* --- mat4x4 --- */

static inline vec4   mat4x4_MultiplyVector(mat4x4 m, vec4 v);
static inline mat4x4 mat4x4_MultiplyMatrix(mat4x4 left, mat4x4 right);
static inline mat4x4 mat4x4_MultiplyScalar(mat4x4 m, float t);
static inline mat4x4 mat4x4_MultiplyScalarR(float t, mat4x4 m);
static inline mat4x4 mat4x4_Add(mat4x4 m1, mat4x4 m2);
static inline mat4x4 mat4x4_Subtract(mat4x4 m1, mat4x4 m2);
static inline mat4x4 mat4x4_DivideScalar(mat4x4 m, float t);
static inline mat4x4 mat4x4_Transpose(mat4x4 m);

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

static inline bool vec4_CompareMagnitudeEqual(vec4 v1, float mag)
{
    float v1mag = vec4_MagnitudeSquared(v1);
    return vequ(v1mag, mag * mag);
}

static inline bool vec4_CompareMagnitudeEqualR(float mag, vec4 v1)
{
    return vec4_CompareMagnitudeEqual(v1, mag);
}

static inline bool vec4_CompareMagnitudeGreaterThan(vec4 v1, float mag)
{
    float v1mag = vec4_MagnitudeSquared(v1);
    return v1mag > mag;
}

static inline bool vec4_CompareMagnitudeGreaterThanR(float mag, vec4 v1)
{
    return vec4_CompareMagnitudeGreaterThan(v1, mag);
}

static inline bool vec4_AlmostTheSame(vec4 v1, vec4 v2)
{
    return vequ(v1.x, v2.x) && vequ(v1.y, v2.y) && vequ(v1.z, v2.z) && vequ(v1.w, v2.w);
}

/* ---- 2x2 Matrix Functions ---- */

static inline vec2 mat2x2_MultiplyVector(mat2x2 m, vec2 v)
{
    return vec2(
        m.x.x * v.x + m.y.x * v.y,
        m.x.y * v.x + m.y.y * v.y
    );
}

static inline mat2x2 mat2x2_MultiplyMatrix(mat2x2 left, mat2x2 right)
{
    return (mat2x2){
        .x = mat2x2_MultiplyVector(left, right.x),
        .y = mat2x2_MultiplyVector(left, right.y),
    };
}

static inline mat2x2 mat2x2_Transpose(mat2x2 m)
{
    return mat2x2(
        m.x.x, m.x.y,
        m.y.x, m.y.y
    );
}

static inline mat2x2 mat2x2_MultiplyScalar(mat2x2 m, float t)
{
    return (mat2x2){
        .x = vec2_MultiplyScalar(m.x, t),
        .y = vec2_MultiplyScalar(m.y, t),
    };
}

static inline mat2x2 mat2x2_MultiplyScalarR(float t, mat2x2 m)
{
    return mat2x2_MultiplyScalar(m, t);
}

static inline mat2x2 mat2x2_Add(mat2x2 m1, mat2x2 m2)
{
    return (mat2x2){
        .x = vec2_Add(m1.x, m2.x),
        .y = vec2_Add(m1.y, m2.y),
    };
}

static inline mat2x2 mat2x2_Subtract(mat2x2 m1, mat2x2 m2)
{
    return (mat2x2){
        .x = vec2_Subtract(m1.x, m2.x),
        .y = vec2_Subtract(m1.y, m2.y),
    };
}

static inline mat2x2 mat2x2_DivideScalar(mat2x2 m, float t)
{
    return mat2x2_MultiplyScalar(m, 1.0f / t);
}

/* ---- 3x3 Matrix Functions ---- */

static inline vec3 mat3x3_MultiplyVector(mat3x3 m, vec3 v)
{
    return vec3(
        m.x.x * v.x + m.y.x * v.y + m.z.x * v.z,
        m.x.y * v.x + m.y.y * v.y + m.z.y * v.z,
        m.x.z * v.x + m.y.z * v.y + m.z.z * v.z
    );
}

static inline mat3x3 mat3x3_MultiplyMatrix(mat3x3 left, mat3x3 right)
{
    return (mat3x3){
        .x = mat3x3_MultiplyVector(left, right.x),
        .y = mat3x3_MultiplyVector(left, right.y),
        .z = mat3x3_MultiplyVector(left, right.z),
    };
}

static inline mat3x3 mat3x3_Transpose(mat3x3 m)
{
    return mat3x3(
        m.x.x, m.x.y, m.x.z,
        m.y.x, m.y.y, m.y.z,
        m.z.x, m.z.y, m.z.z
    );
}

static inline mat3x3 mat3x3_MultiplyScalar(mat3x3 m, float t)
{
    return (mat3x3){
        .x = vec3_MultiplyScalar(m.x, t),
        .y = vec3_MultiplyScalar(m.y, t),
        .z = vec3_MultiplyScalar(m.z, t),
    };
}

static inline mat3x3 mat3x3_MultiplyScalarR(float t, mat3x3 m)
{
    return mat3x3_MultiplyScalar(m, t);
}

static inline mat3x3 mat3x3_Add(mat3x3 m1, mat3x3 m2)
{
    return (mat3x3){
        .x = vec3_Add(m1.x, m2.x),
        .y = vec3_Add(m1.y, m2.y),
        .z = vec3_Add(m1.z, m2.z),
    };
}

static inline mat3x3 mat3x3_Subtract(mat3x3 m1, mat3x3 m2)
{
    return (mat3x3){
        .x = vec3_Subtract(m1.x, m2.x),
        .y = vec3_Subtract(m1.y, m2.y),
        .z = vec3_Subtract(m1.z, m2.z),
    };
}

static inline mat3x3 mat3x3_DivideScalar(mat3x3 m, float t)
{
    return mat3x3_MultiplyScalar(m, 1.0f / t);
}

/* ---- 4x4 Matrix Functions ---- */

#if VMATH_SIMD == VMATH_SIMD_AVX2
static inline vec4 mat4x4_MultiplyVector(mat4x4 m, vec4 v)
{
    __m128 xx, yy, zz, ww;
    __m128 tx, txy, txyz, txyzw;

    xx = _mm_set1_ps(v.x);
    tx = _mm_mul_ps(m.x.m128, xx);

    yy = _mm_set1_ps(v.y);
    txy = _mm_fmadd_ps(m.y.m128, yy, tx);

    zz = _mm_set1_ps(v.z);
    txyz = _mm_fmadd_ps(m.z.m128, zz, txy);

    ww = _mm_set1_ps(v.w);
    txyzw = _mm_fmadd_ps(m.w.m128, ww, txyz);

    return (vec4){.m128 = txyzw};
}
#elif VMATH_SIMD == VMATH_SIMD_SSE
static inline vec4 mat4x4_MultiplyVector(mat4x4 m, vec4 v)
{
    __m128 xx, yy, zz, ww;
    __m128 tx, ty, tz, tw;
    __m128 txyzw;

    xx = _mm_set1_ps(v.x);
    tx = _mm_mul_ps(m.x.m128, xx);

    yy = _mm_set1_ps(v.y);
    ty = _mm_mul_ps(m.y.m128, yy);

    zz = _mm_set1_ps(v.z);
    tz = _mm_mul_ps(m.z.m128, zz);

    ww = _mm_set1_ps(v.w);
    tw = _mm_mul_ps(m.w.m128, ww);

    txyzw = _mm_add_ps(_mm_add_ps(tx, ty), _mm_add_ps(tz, tw));

    return (vec4){.m128 = txyzw};
}
#elif VMATH_SIMD == VMATH_SIMD_NONE
static inline vec4 mat4x4_MultiplyVector(mat4x4 m, vec4 v)
{
    return vec4(
        m.x.x * v.x + m.y.x * v.y + m.z.x * v.z + m.w.x * v.w,
        m.x.y * v.x + m.y.y * v.y + m.z.y * v.z + m.w.y * v.w,
        m.x.z * v.x + m.y.z * v.y + m.z.z * v.z + m.w.z * v.w,
        m.x.w * v.x + m.y.w * v.y + m.z.w * v.z + m.w.w * v.w
    );
}
#endif

static inline mat4x4 mat4x4_MultiplyMatrix(mat4x4 left, mat4x4 right)
{
    return (mat4x4){
        .x = mat4x4_MultiplyVector(left, right.x),
        .y = mat4x4_MultiplyVector(left, right.y),
        .z = mat4x4_MultiplyVector(left, right.z),
        .w = mat4x4_MultiplyVector(left, right.w),
    };
}

#if VMATH_SIMD == VMATH_SIMD_AVX2
static inline mat4x4 mat4x4_Transpose(mat4x4 m)
{
    __m256i indices = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    __m256 xy = m.m256[0];
    __m256 zw = m.m256[1];

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
static inline mat4x4 mat4x4_Transpose(mat4x4 m)
{
    __m128 xy_xy = _mm_unpacklo_ps(m.x.m128, m.y.m128);
    __m128 xy_zw = _mm_unpackhi_ps(m.x.m128, m.y.m128);

    __m128 zw_xy = _mm_unpacklo_ps(m.z.m128, m.w.m128);
    __m128 zw_zw = _mm_unpackhi_ps(m.z.m128, m.w.m128);

    return (mat4x4){
        .x.m128 = _mm_shuffle_ps(xy_xy, zw_xy, _MM_SHUFFLE(1, 0, 1, 0)),
        .y.m128 = _mm_shuffle_ps(xy_xy, zw_xy, _MM_SHUFFLE(3, 2, 3, 2)),
        .z.m128 = _mm_shuffle_ps(xy_zw, zw_zw, _MM_SHUFFLE(1, 0, 1, 0)),
        .w.m128 = _mm_shuffle_ps(xy_zw, zw_zw, _MM_SHUFFLE(3, 2, 3, 2)),
    };
}
#else
static inline mat4x4 mat4x4_Transpose(mat4x4 m)
{
    return mat4x4(
        m.x.x, m.x.y, m.x.z, m.x.w,
        m.y.x, m.y.y, m.y.z, m.y.w,
        m.z.x, m.z.y, m.z.z, m.z.w,
        m.w.x, m.w.y, m.w.z, m.w.w
    );
}
#endif

#if VMATH_SIMD == VMATH_SIMD_AVX2
static inline mat4x4 mat4x4_MultiplyScalar(mat4x4 m, float t)
{
    __m256 vt = _mm256_set1_ps(t);

    return (mat4x4){
        .m256 = {
            _mm256_mul_ps(vt, m.m256[0]),
            _mm256_mul_ps(vt, m.m256[1]),
        },
    };
}
#else
static inline mat4x4 mat4x4_MultiplyScalar(mat4x4 m, float t)
{
    return (mat4x4){
        .x = vec4_MultiplyScalar(m.x, t),
        .y = vec4_MultiplyScalar(m.y, t),
        .z = vec4_MultiplyScalar(m.z, t),
        .w = vec4_MultiplyScalar(m.w, t),
    };
}
#endif

static inline mat4x4 mat4x4_MultiplyScalarR(float t, mat4x4 m)
{
    return mat4x4_MultiplyScalar(m, t);
}

#if VMATH_SIMD == VMATH_SIMD_AVX2
static inline mat4x4 mat4x4_Add(mat4x4 m1, mat4x4 m2)
{
    return (mat4x4){
        .m256 = {
            _mm256_add_ps(m1.m256[0], m2.m256[0]),
            _mm256_add_ps(m1.m256[1], m2.m256[1]),
        },
    };
}
#else
static inline mat4x4 mat4x4_Add(mat4x4 m1, mat4x4 m2)
{
    return (mat4x4){
        .x = vec4_Add(m1.x, m2.x),
        .y = vec4_Add(m1.y, m2.y),
        .z = vec4_Add(m1.z, m2.z),
        .w = vec4_Add(m1.w, m2.w),
    };
}
#endif

#if VMATH_SIMD == VMATH_SIMD_AVX2
static inline mat4x4 mat4x4_Subtract(mat4x4 m1, mat4x4 m2)
{
    return (mat4x4){
        .m256 = {
            _mm256_sub_ps(m1.m256[0], m2.m256[0]),
            _mm256_sub_ps(m1.m256[1], m2.m256[1]),
        },
    };
}
#else
static inline mat4x4 mat4x4_Subtract(mat4x4 m1, mat4x4 m2)
{
    return (mat4x4){
        .x = vec4_Subtract(m1.x, m2.x),
        .y = vec4_Subtract(m1.y, m2.y),
        .z = vec4_Subtract(m1.z, m2.z),
        .w = vec4_Subtract(m1.w, m2.w),
    };
}
#endif

static inline mat4x4 mat4x4_DivideScalar(mat4x4 m, float t)
{
    return mat4x4_MultiplyScalar(m, 1.0f / t);
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

static inline vec3 Reflect(vec3 vec, vec3 normal)
{
    return vsub(vec, vmul(normal, 2.0f * vdot(vec, normal)));
}

static inline vec3 Refract(vec3 vec, vec3 normal, float refractRatio)
{
    float cosTheta = vmin(vdot(vmul(vec, -1), normal), 1.0f);
    vec3 vecOutPerp = vmul(vadd(vec, vmul(normal, cosTheta)), refractRatio);
    vec3 vecOutPara = vmul(normal, -sqrtf(vmag(1.0f - vdot(vecOutPerp, vecOutPerp))));
    return vadd(vecOutPerp, vecOutPara);
}

// TODO: figure out what to name this function
// Creates an orthonormal basis with a vector bz as the Z basis vector
// See: https://backend.orbit.dtu.dk/ws/portalfiles/portal/126824972/onb_frisvad_jgt2012_v2.pdf
// WARNING: normal_in_z must be normalized
static inline mat3x3 OrthonormalBasis(vec3 normal_in_z)
{
    vec3 bz = normal_in_z;
    vec3 bx, by;
    if (VMATH_UNLIKELY(bz.z < -0.9999999f)) {
        by = vec3(0.0f, -1.0f, 0.0f);
        bx = vec3(-1.0f, 0.0f, 0.0f);
    } else {
        float a = 1.0f / (1.0f + bz.z);
        float b = -bz.x * bz.y * a;

        by = vec3(-1.0f + bz.x * bz.x * a, -b, bz.x);
        bx = vec3(-b, -1.0f + bz.y * bz.y * a, bz.y);
    }

    return (mat3x3) {
        .x = bx,
        .y = by,
        .z = bz,
    };
}

// static inline mat4x4 InverseAffine(mat4x4 m){}
//
