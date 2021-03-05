#ifndef MATRIX_H
#define MATRIX_H
typedef struct LD_pair {
    double *L;
    double *D;
} LD_pair;

LD_pair cholDecomp_LD(double *A, int size);

double *transpose(double *A, int size);

double *randHerm(int size);

int isHerm(double *Matrix, int size);

double *matMul(double *A, double *B, int size);

double *matMulDiag(double *A, double *D, int size);

int matEqual(double *A, double *B, int size, double tol);

void printMatrix(double *Matrix, int size);

void printArray(double *array, int size);

double get_wall_seconds();

#endif