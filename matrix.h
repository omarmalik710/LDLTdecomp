#ifndef MATRIX_H
#define MATRIX_H
typedef struct LD_pair {
    double *L;
    double *D;
} LD_pair;

typedef struct thread_args {
    double *L, *D, *A;
    int j;
    int N;
    int i1, i2;
    int thrID;
} thrArgs;

#define ELEMS_PER_REG 2
#define REGS_PER_iITER 2
#define ELEMS_PER_iITER 4
#define UNROLL_FACT 4

void *calcLij_thread(void *myArgs);

LD_pair LDLTdecomp(double* restrict A, const int N);

LD_pair LDLTdecomp_blocks(double* restrict A, const int N, const int blockSize);

double *transpose(double* restrict A, const int N);

double *transpose_blocks(double* restrict A, const int N, const int blockSize);

double *randHerm(const int N);

int isHerm(double* restrict Matrix, const int N);

double *matMul(double* restrict A, double* restrict B, const int N);

double *matMul_blocks(double* restrict A, double* restrict B, const int N, const int blockSize);

double *matMulDiag(double* restrict A, double *D, const int N);

double *matMulDiag_blocks(double* restrict A, double *D, const int N, const int blockSize);

int matEqual(double* restrict A, double* restrict B, const int N, const double tol);

void printMatrix(double* restrict Matrix, const int N);

void printArray(double* restrict array, const int N);

double get_wall_seconds();

#endif