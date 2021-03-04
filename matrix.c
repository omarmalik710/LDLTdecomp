#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "matrix.h"

double **cholDecomp(double **A, int size) {
    double **L = allocMatrix(size);

    double Ljj;
    double Lij;
    for (int j=0; j<size; j++) {
        Ljj = A[j][j];
        for (int i=j+1; i<size; i++) {
            Lij = A[i][j];
            for (int k=0; k<j; k++) {
                Ljj -= L[j][k]*L[j][k];
                printf("L[j][j] = %lf\n", Ljj);
                Lij -= L[i][k]*L[j][k];
            }
            L[i][j] = Lij/sqrt(Ljj);
        }
        L[j][j] = sqrt(Ljj);
    }

    return L;
}

//double **cholDecomp(double **A, int size) {
//    double **L = allocMatrix(size);
//
//    double Ljj;
//    double Lij;
//    for (int j=0; j<size; j++) {
//        Ljj = A[j][j];
//        for (int i=j+1; i<size; i++) {
//            Lij = A[i][j];
//            for (int k=0; k<j; k++) {
//                Ljj -= L[j][k]*L[j][k];
//                printf("L[j][j] = %lf\n", Ljj);
//                Lij -= L[i][k]*L[j][k];
//            }
//            L[i][j] = Lij/sqrt(Ljj);
//        }
//        L[j][j] = sqrt(Ljj);
//    }
//
//    return L;
//}

double **transpose(double **A, int size) {
    double **AT = allocMatrix(size);

    for (int i=0; i<size; i++) {
        for (int j=0; j<size; j++) {
            AT[i][j] = A[j][i];
        }
    }

    return AT;
}

double **randHerm(int size) {
    double **Matrix = allocMatrix(size);

    time_t t;
    srand((unsigned) time(&t));
    int i,j;
    for (i=0; i<size; i++) {
        Matrix[i][i] = rand()%20;
        for (j=i+1; j<size; j++) {
            Matrix[i][j] = rand()%5;
            Matrix[j][i] = Matrix[i][j];
        }
    }

    return Matrix;
}

int isHerm(double **Matrix, int size) {
    for (int i=0; i<size; i++) {
        for (int j=0; j<i; j++) {
            if (Matrix[i][j] != Matrix[j][i]) { return 0; }
        }
    }

    return 1;
}

double **matMul(double **A, double **B, int size) {
    double **C = allocMatrix(size);
    double C_ij;

    for (int i=0; i<size; i++) {
        for (int j=0; j<size; j++) {
            C_ij = 0.0;
            for (int k=0; k<size; k++) {
                C_ij += A[i][k]*B[k][j];
            }
            C[i][j] = C_ij;
        }
    }

    return C;
}

int matEqual(double **A, double **B, int size, double tol) {
    for (int i=0; i<size; i++) {
        for (int j=0; j<size; j++) {
            if ( abs(A[i][j]-B[i][j]) > tol ) {
                return 0;
            }
        }
    }

    return 1;
}

double **allocMatrix(int size) {
    double **Matrix = (double **)malloc(size*sizeof(double *));
    for (int i=0; i<size; i++) {
        Matrix[i] = (double *)malloc(size*sizeof(double));
    }

    return Matrix;
}

void printMatrix(double **Matrix, int size) {
    for (int i=0; i<size; i++) {
        for (int j=0; j<size; j++) {
            printf("%10.6lf ", Matrix[i][j]);
        }
        putchar('\n');
    }
}

void deleteMatrix(double **Matrix, int size) {
    for (int i=0; i<size; i++) {
        free(Matrix[i]);
        Matrix[i] = NULL;
    }
    free(Matrix);
    Matrix = NULL;
}
