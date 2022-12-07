#include <stdio.h>

#include "vmath.h"

int main(void) {
    mat4x4 m1 = mat4x4(
        0,  1,  2, 3,
        -4, 5,  6, 7,
        8,  9,  0, 1,
        2,  3, -4, 5
    );

    mat4x4 m2 = mat4x4(
         1,     2,     -4,    8,
        -16,   -32,    64,    128,
         256,   512,   1024, -2048,
         4096, -8192,  16384, 32768
    );

    vec4 v = vec4(0, 1, 2, 3);

    vec4 r = vmul(m1, v);
    mat4x4 mr = vtrans(m1);

    printf("r = " VEC4_FMT "\n", VEC4_ARG(r));
    printf("mr = " MAT4X4_FMT "\n", MAT4X4_ARG(mr));

    return 0;
}
