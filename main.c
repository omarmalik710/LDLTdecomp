#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"

int main(int argc, char **argv) {
    double time1, time2;

    int size;
    int blockSize;
    switch (argc) {
        case 2:
            size = atoi(argv[1]);
            blockSize = 16;
            break;
        case 3:
            size = atoi(argv[1]);
            blockSize = atoi(argv[2]);
            break;
        default:
            size = 2e3;
            blockSize = 16;
    }
    if (size%blockSize != 0) {
        printf("[ERROR] Matrix size %dx%d not divisible by block size %dx%d!\n", size,size, blockSize,blockSize);
        exit(1);
    }

    //double* restrict A __attribute__((aligned (XMM_ALIGNMENT_BYTES)));

    double *A = randHerm(size);
    time1 = get_wall_seconds();
    //LD_pair LD = LDLTdecomp(A, size);
    LD_pair LD = LDLTdecomp_blocks(A, size, blockSize);
    time2 = get_wall_seconds();
    printf("%lf\n", time2-time1);
    //double *LxD = matMulDiag(LD.L, LD.D, size);
    //double *LT = transpose(LD.L, size);
    //double *LxDxLT = matMul(LxD,LT, size);
    //double *LxD = matMulDiag_blocks(LD.L, LD.D, size, blockSize);
    //double *LT = transpose_blocks(LD.L, size, blockSize);
    //double *LxDxLT = matMul_blocks(LxD,LT, size, blockSize);

    //printf("A =\n");
    //printMatrix(A, size);
    //putchar('\n');

    //printf("L =\n");
    //printMatrix(LD.L, size);
    //putchar('\n');

    //printf("LT = \n");
    //printMatrix(LT, size);
    //putchar('\n');

    //printf("D = \n");
    //printArray(LD.D, size);

    //printf("L*D*LT =\n");
    //printMatrix(LxDxLT, size);

    //if (matEqual(A, LxDxLT, size, 1e-12)) {
    //    printf("A = L*D*LT :D\n");
    //} else { printf("A != L*D*LT :(\n"); }

    free(A);
    free(LD.L);
    free(LD.D);
    //free(LT);
    //free(LxD);
    //free(LxDxLT);

    //int size = 5;
    //double *A = randHerm(size);
    //double *L = LDLTdecomp(A, size);
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