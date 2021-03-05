#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int main() {
    int size = 5;
    double *A = randHerm(size);
    //double *L = cholDecomp(A, size);
    //double *LT = transpose(L, size);
    //double *LLT = matMul(L,LT, size);
    LD_pair LD = cholDecomp_LD(A, size);
    double *LxD = matMulDiag(LD.L, LD.D, size);
    double *LT = transpose(LD.L, size);
    double *LxDxLT = matMul(LxD,LT, size);

    printf("A =\n");
    printMatrix(A, size);
    putchar('\n');

    printf("L =\n");
    printMatrix(LD.L, size);
    putchar('\n');

    printf("D = \n");
    printArray(LD.D, size);

    printf("L*D*LT =\n");
    printMatrix(LxDxLT, size);

    if (matEqual(A, LxDxLT, size, 1e-12)) {
        printf("A = L*D*LT :D\n");
    } else { printf("A != L*D*LT :(\n"); }

    free(A);
    free(LD.L);
    free(LD.D);
    free(LT);
    free(LxD);
    free(LxDxLT);

    return 0;
}