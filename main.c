#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"

int main() {
    double time1, time2;

    int size = 2000;
    int blockSize = 32;
    if (size%blockSize != 0) {
        printf("[ERROR] Matrix size %dx%d not divisible by block size %dx%d!\n", size,size, blockSize,blockSize);
        exit(1);
    }

    double *A = randHerm(size);
    //printf("A =\n");
    //printMatrix(A, size);
    //putchar('\n');

    LD_pair LD = cholDecomp_LD(A, size);
    double *LxD = matMulDiag(LD.L, LD.D, size);
    time1 = get_wall_seconds();
    double *LT = transpose_blocks(LD.L, size, blockSize);
    //double *LT = transpose(LD.L, size);
    time2 = get_wall_seconds();
    double *LxDxLT = matMul(LxD,LT, size);

    //printf("L =\n");
    //printMatrix(LD.L, size);
    //putchar('\n');

    //printf("D = \n");
    //printArray(LD.D, size);

    //printf("L*D*LT =\n");
    //printMatrix(LxDxLT, size);

    if (matEqual(A, LxDxLT, size, 1e-12)) {
        printf("A = L*D*LT :D\n");
    } else { printf("A != L*D*LT :(\n"); }
    printf("[TIME] Matrix transpose took %lf seconds.\n", time2-time1);

    free(A);
    free(LD.L);
    free(LD.D);
    free(LT);
    free(LxD);
    free(LxDxLT);

    //int size = 5;
    //double *A = randHerm(size);
    //double *L = cholDecomp(A, size);
    //double *LT = transpose(L, size);
    //double *LLT = matMul(L,LT, size);

    //printf("A =\n");
    //printMatrix(A, size);
    //putchar('\n');

    //printf("L =\n");
    //printMatrix(L, size);
    //putchar('\n');

    //printf("L*LT =\n");
    //printMatrix(LLT, size);

    //if (matEqual(A, LLT, size, 1e-12)) {
    //    printf("A = L*LT :D\n");
    //} else { printf("A != L*LT :(\n"); }

    //free(A);
    //free(L);
    //free(LT);
    //free(LLT);

    return 0;
}