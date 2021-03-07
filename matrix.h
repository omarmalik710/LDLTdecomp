#ifndef MATRIX_H
#define MATRIX_H
typedef struct LD_pair {
    double *L;
    double *D;
} LD_pair;

#define ELEMS_PER_REG 2
#define UNROLL_FACT 5

LD_pair LDLTdecomp(double* restrict A, const int size);

LD_pair LDLTdecomp_blocks(double* restrict A, const int size, const int blockSize);

double *transpose(double* restrict A, const int size);

double *transpose_blocks(double* restrict A, const int size, const int blockSize);

double *randHerm(const int size);

int isHerm(double* restrict Matrix, const int size);

double *matMul(double* restrict A, double* restrict B, const int size);

double *matMul_blocks(double* restrict A, double* restrict B, const int size, const int blockSize);

double *matMulDiag(double* restrict A, double *D, const int size);

double *matMulDiag_blocks(double* restrict A, double *D, const int size, const int blockSize);

int matEqual(double* restrict A, double* restrict B, const int size, const double tol);

void printMatrix(double* restrict Matrix, const int size);

void printArray(double* restrict array, const int size);

double get_wall_seconds();

#endif