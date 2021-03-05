#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int main() {
    int size = 5;
    double **A = randHerm(size);
    double **L = cholDecomp(A, size);
    double **LT = transpose(L, size);
    double **LLT = matMul(L,LT, size);

    printf("A =\n");
    printMatrix(A, size);
    putchar('\n');

    printf("L =\n");
    printMatrix(L, size);
    putchar('\n');

    printf("L*LT =\n");
    printMatrix(LLT, size);

    if (matEqual(A, LLT, size, 1e-12)) {
        printf("A = L*LT :D\n");
    } else { printf("A != L*LT :(\n"); }

    deleteMatrix(A, size);
    deleteMatrix(L, size);
    deleteMatrix(LT, size);
    deleteMatrix(LLT, size);

    return 0;
}