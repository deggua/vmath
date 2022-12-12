# vmath (wip)
Header only vector and matrix math C11 library with SIMD acceleration, supporting clang, GCC, and MSVC

This library is originally derived from the vector functions I needed in [my raytracer](https://github.com/deggua/raytracer)

# Features
Provides primitives for 3D graphics programming such as:
* vec2, vec3, vec4
* mat2, mat3, mat4 (column major)
* reflect, refract
* cartesian, polar, spherical transformations
* orthonormal basis creation from a normal vector
* inverse of various affine matrices (translation, rotation, scale, and their combos)
* lerp
* misc scalar functions

Functions are macro overloaded via _Generic to simplify usage:
```C
vec2 v1 = vec2(1, 2);
vec2 v2 = vec2(4, 2);
float dp2 = vdot(v1, v2);

vec3 v1 = vec3(1, 2, 3);
vec3 v2 = vec3(4, 2, 1);
float dp3 = vdot(v1, v2);

vec4 v1 = vec4(1, 2, 3, 4);
vec4 v2 = vec4(4, 2, 1, 0);
float dp4 = vdot(v1, v2);
```

Some macro functions are provided for concise expressions:
```C
vec3 v1 = vec3(0, 2, 4);
vec3 v2 = vec3(1, 1, 1);
vec3 v3 = vec3(5, 1, 5);
vec3 v4 = vec3(7, 7, 7);
vec3 sum = vsum(v1, v2, v3, v4);
// sum = v1 + v2 + v3 + v4
```

Accelerated versions of some functions are provided if SSE or AVX2 is detected to be available at compile time. Approximate versions are used if compiled with in a fast floating point mode (`-Ofast` or `/fp:fast`)

Printf format specifiers are provided for easily printing vectors and matrices
```C
vec3 v1 = vec3(1, 2, 3);
printf("v1 = " VEC3_FMT "\n", VEC3_ARG(v1));
// v1 = <1.00, 2.00, 3.00>
```

# Functions
| Name                        | Operation                   | Description                         |
| --------------------------- | --------------------------- | ----------------------------------- |
| `vadd(a, b)`                | `a + b`                     | Addition                            |
| `vsub(a, b)`                | `a - b`                     | Subtraction                         |
| `vmul(a, b)`                | `a * b`                     | Multiplication                      |
| `vdiv(a, b)`                | `a / b`                     | Division                            |
| `vdot(a, b)`                | `a . b`                     | Dot Product                         |
| `vcross(a, b)`              | `a x b`                     | Cross Product                       |
| `vmag(a)`                   | `\|\|a\|\|`                 | Magnitude                           |
| `vmag2(a)`                  | `\|\|a\|\|^2`               | Magnitude Squared                   |
| `vnorm(a)`                  | `a / \|\|a\|\|`             | Normalize                           |
| `vlerp(a, b, t)`            | `a + (b - a) * t`           | Linear Interpolate                  |
| `vequ(a, b)`                | `a == b`                    | Test Equality                       |
| `vtrans(M)`                 | `M^T`                       | Transpose                           |
| `vmax(a, b)`                | `max(a, b)`                 | Maximum                             |
| `vmin(a, b)`                | `min(a, b)`                 | Minimum                             |
| `vsqrt(a)`                  | `sqrt(a)`                   | Square Root                         |
| `vrsqrt(a)`                 | `1 / sqrt(a)`               | Reciprocal Square Root              |
| `vrcp(a)`                   | `1 / a`                     | Reciprocal                          |
| `vclamp(a, a_min, a_max)`   | `max(min(a, a_max), a_min)` | Clamp to Range                      |
| `vneg(a)`                   | `-a`                        | Negate                              |
| `vdist(a, b)`               | `\|\|a - b\|\|`             | Distance                            |
| `vsum(v_1, v_2, ..., v_n)`  | `v_1 + v_2 + ... + v_n`     | Sum of N Items                      |
| `vprod(v_1, v_2, ..., v_n)` | `v_1 * v_2 * ... * v_n`     | Product of N Items                  |
| `vec2(t)`                   | `<t, t>`                    | `vec2` with all elements set to `t` |
| `vec2(x, y)`                | `<x, y>`                    | `vec2` specifying each element      |
| `vec3(t)`                   | `<t, t, t>`                 | `vec3` with all elements set to `t` |
| `vec3(x, y, z)`             | `<x, y, z>`                 | `vec3` specifying each element      |
| `vec4(t)`                   | `<t, t, t, t>`              | `vec4` with all elements set to `t` |
| `vec4(x, y, z, w)`          | `<x, y, z, w>`              | `vec4` specifying each element      |
| `mat2(`<br>`m11, m12,`<br>`m21, m22)`  | `\| m11 m12 \|`<br>`\| m21 m22 \|` | `mat2` specified left-right, top-down |
| `mat3(`<br>`m11, m12, m13,`<br>`m21, m22, m23,`<br>`m31, m32, m33)`  | `\| m11 m12 m13 \|`<br>`\| m21 m22 m23 \|`<br>`\| m31 m32 m33 \|` | `mat3` specified left-right, top-down |
| `mat4(`<br>`m11, m12, m13, m14,`<br>`m21, m22, m23, m24,`<br>`m31, m32, m33, m34,`<br>`m41, m42, m43, m44)`  | `\| m11 m12 m13 m14 \|`<br>`\| m21 m22 m23 m24 \|`<br>`\| m31 m32 m33 m34 \|`<br>`\| m41 m42 m43 m44 \|` | `mat4` specified left-right, top-down |
| `Radians(deg)` | `π * deg / 180` | Convert degrees to radians |
| `Degrees(rad)` | `180 * rad / π` | Convert radians to degrees |
| `Reflect(v_in, v_norm)` | `...` | Compute the outgoing reflection of an incoming vector `v_in` about `v_norm` (points against `v_in`) |
| `Refract(v_in, v_norm, eta)` | `...` | Compute the refraction of an incoming vector `v_in` about `v_norm` (points against `v_in`) with `eta = n_out / n_in` |
| `OrthonormalBasis(norm_in_z)` | `...` | Compute a matrix which transforms `<0, 0, 1>` into `norm_in_z` where the basis vectors are orthonormal |
| `InverseAffineMatrix_T(M)` | `M^-1` | Compute `M^-1` where `M` is an affine translation matrix |
| `InverseAffineMatrix_TR(M)` | `M^-1` | Compute `M^-1` where `M` is an affine translation-rotation matrix |
| `InverseAffineMatrix_TS(M)` | `M^-1` | Compute `M^-1` where `M` is an affine translation-scale matrix |
| `InverseAffineMatrix_TRS(M)` | `M^-1` | Compute `M^-1` where `M` is an affine translation-rotation-scale matrix |
| `AffineMatrix(M)` | `...` | Convert a `mat3` into a `mat4` in affine coordinates with no translation component |
| `PolarToCartesian(v)` | `...` | Convert a vector in polar coordinates to a vector in cartesian coordinates |
| `CartesianToPolar(v)` | `...` | Convert a vector in cartesian coordinates to a vector in polar coordinates |
| `SphericalToCartesian(v)` | `...` | Convert a vector in spherical coordinates to a vector in cartesian coordiantes |
| `CartesianToSpherical(v)` | `...` | Convert a vector in cartesian coordinates to a vector in spherical coordinates |

# TODO:
* SIMD implementations for 2d and 3d vectors and matrices
* Tests (accuracy, performance, functionality)
* Left handed multiplication (row vectors) (?)
* General inverse matrix functions
* Misc helper functions (slerp, log, pow, trig functions)
* Broader support for overloaded functions (min/max/equ for matrices, some other stuff)
* Quaternions
