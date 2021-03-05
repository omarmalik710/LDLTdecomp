#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include "matrix.h"

LD_pair cholDecomp_LD(double *A, int size) {

    double *L = (double *)calloc(size*size,sizeof(double));
    double *D = (double *)malloc(size*sizeof(double));

    double time1, timeDj, timeLij;
    double Dj;
    double factor;
    int i,j,k;
    for (j=0; j<size; j++) {

        time1 = get_wall_seconds();
        Dj = A[j+size*j];
        for (k=0; k<j; k++) {
            Dj -= L[j+size*k]*L[j+size*k]*D[k];
        }
        D[j] = Dj;
        timeDj = get_wall_seconds() - time1;

        time1 = get_wall_seconds();
        L[j+size*j] = 1.0;
        for (k=0; k<j; k++) {
            factor = L[j+size*k]*D[k];
            for (i=j+1; i<size; i++) {
                L[i+size*j] -= L[i+size*k]*factor;
            }
        }
        for (i=j+1; i<size; i++) {
            L[i+size*j] = (L[i+size*j] + A[i+size*j])/Dj;
        }
        timeLij = get_wall_seconds() - time1;

        //printf("[TIME] One Dj took %lf seconds.\n", timeDj);
        //printf("[TIME] One Lij took %lf seconds.\n", timeLij);
    }

    LD_pair LD;
    LD.L = L;
    LD.D = D;
    return LD;
}

double *transpose(double *A, int size) {
    double *AT = (double *)calloc(size*size,sizeof(double));

    for (int i=0; i<size; i++) {
        for (int j=0; j<size; j++) {
            AT[i+size*j] = A[j+size*i];
        }
    }

    return AT;
}

double *transpose_blocks(double *A, int size, int blockSize) {

    double *AT = (double *)calloc(size*size,sizeof(double));
    int numBlocks = size/blockSize;
    int istart;
    for (int iBlock=0; iBlock<numBlocks; iBlock++) {
        istart = iBlock*blockSize;
        for (int j=0; j<size; j++) {
            for (int i=istart; i<(istart+blockSize); i++) {
                AT[i+size*j] = A[j+size*i];
            }
        }
    }

    return AT;
}

double *randHerm(int size) {
    double *Matrix = (double *)calloc(size*size,sizeof(double));

    time_t t;
    srand((unsigned) time(&t));
    int i,j;
    for (i=0; i<size; i++) {
        Matrix[i+size*i] = rand()%20;
        for (j=i+1; j<size; j++) {
            Matrix[i+size*j] = rand()%5;
            Matrix[j+size*i] = Matrix[i+size*j];
        }
    }

    return Matrix;
}

int isHerm(double *Matrix, int size) {
    for (int i=0; i<size; i++) {
        for (int j=0; j<i; j++) {
            if (Matrix[i+size*j] != Matrix[j+size*i]) { return 0; }
        }
    }

    return 1;
}

double *matMul(double *A, double *B, int size) {
    double *C = (double *)calloc(size*size,sizeof(double));
    double factor;

    for (int j=0; j<size; j++) {
        for (int k=0; k<size; k++) {
            factor = B[k+size*j];
            for (int i=0; i<size; i++) {
                C[i+size*j] += A[i+size*k]*factor;
            }
        }
    }

    return C;
}

double *matMulDiag(double *A, double *D, int size) {
    double *C = (double *)calloc(size*size,sizeof(double));
    double Dj;

    for (int j=0; j<size; j++) {
        Dj = D[j];
        for (int i=0; i<size; i++) {
            C[i+size*j] = A[i+size*j]*Dj;
        }
    }

    return C;
}

int matEqual(double *A, double *B, int size, double tol) {
    for (int i=0; i<size; i++) {
        for (int j=0; j<size; j++) {
            if ( abs(A[i+size*j]-B[i+size*j]) > tol ) {
                return 0;
            }
        }
    }

    return 1;
}

void printMatrix(double *Matrix, int size) {
    for (int i=0; i<size; i++) {
        for (int j=0; j<size; j++) {
            printf("%10.6lf ", Matrix[i+size*j]);
        }
        putchar('\n');
    }
}

void printArray(double *array, int size) {
    for (int i=0; i<size; i++) {
        printf("%10.6lf ", array[i]);
    }
    putchar('\n');
}

double get_wall_seconds() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  double seconds = tv.tv_sec + (double)tv.tv_usec / 1000000;
  return seconds;
}
