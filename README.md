# vmath
Header only vector and matrix math C11 library with SIMD acceleration, supporting clang, GCC, and MSVC

This library is originally derived from the vector functions I needed in [my raytracer](https://github.com/deggua/raytracer)

# Features
Provides primitives for 3D graphics programming such as:
* vec2, vec3, vec4
* mat2x2, mat3x3, mat4x4 (column major)
* reflect, refract
* cartesian, polar, spherical transformations
* orthonormal basis creation from a normal vector
* lerp
* scalar functions

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

Some macro functions are provided for simplicity:
```C
vec3 v1 = vec3(0, 2, 4);
vec3 v2 = vec3(1, 1, 1);
vec3 v3 = vec3(5, 1, 5);
vec3 v4 = vec3(7, 7, 7);
vec3 sum = vsum(v1, v2, v3, v4);
// sum = v1 + v2 + v3 + v4
```

Accelerated versions of some functions are provided if SSE or AVX2 is detected to be available

Printf format specifiers are provided for efficiently printing vectors and matrices
```C
vec3 v1 = vec3(1, 2, 3);
printf("v1 = " VEC3_FMT "\n", VEC3_ARG(v1));
// v1 = <1.00, 2.00, 3.00>
```

# TODO:
* Affine matrix inverse
* More helper functions (`vneg`, `vdist`, etc)
* SIMD implementations for 2d and 3d vectors and matrices
* Tests
* Row vectors (?)
* General inverse matrix functions
* Misc helper functions
* Quaternions
