#ifndef MATRIX_H
#define MATRIX_H

typedef struct LD_pair {
    double **L;
    double **D;
} LD_pair;

double **cholDecomp(double **A, int size);

double **transpose(double **A, int size);

double **randHerm(int size);

int isHerm(double **Matrix, int size);

double **matMul(double **A, double **B, int size);

int matEqual(double **A, double **B, int size, double tol);

double **allocMatrix(int size);

void printMatrix(double **Matrix, int size);

void deleteMatrix(double **Matrix, int size);

#endif