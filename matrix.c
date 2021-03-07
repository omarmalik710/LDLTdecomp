#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pmmintrin.h>
#include "matrix.h"

LD_pair cholDecomp_LD(double *A, int size) {

    double *L = (double *)calloc(size*size,sizeof(double));
    double *D = (double *)malloc(size*sizeof(double));

    double time1, timeDj, timeLij;
    double Dj;
    double factor;
    int i,j,k;
    int unrollFact = 4;
    int iRemain;
    int kRemain;
    for (j=0; j<size; j++) {

        time1 = get_wall_seconds();
        kRemain = j%unrollFact;
        Dj = A[j+size*j];
        for (k=0; k<kRemain; k++) {
            Dj -= L[j+size*k]*L[j+size*k]*D[k];
        }
        for (k; k<j; k+=unrollFact) {
            Dj -= L[j+size*k]*L[j+size*k]*D[k];
            Dj -= L[j+size*(k+1)]*L[j+size*(k+1)]*D[k+1];
            Dj -= L[j+size*(k+2)]*L[j+size*(k+2)]*D[k+2];
            Dj -= L[j+size*(k+3)]*L[j+size*(k+3)]*D[k+3];
        }
        D[j] = Dj;
        timeDj = get_wall_seconds() - time1;

        time1 = get_wall_seconds();
        iRemain = (size-(j+1))%unrollFact;
        L[j+size*j] = 1.0;
        for (k=0; k<j; k++) {
            factor = L[j+size*k]*D[k];
            for (i=j+1; i<(iRemain+(j+1)); i++) {
                L[i+size*j] -= L[i+size*k]*factor;
            }
            for (i; i<size; i+=unrollFact) {
                L[i+size*j] -= L[i+size*k]*factor;
                L[i+1+size*j] -= L[i+1+size*k]*factor;
                L[i+2+size*j] -= L[i+2+size*k]*factor;
                L[i+3+size*j] -= L[i+3+size*k]*factor;
            }
        }
        for (i=j+1; i<(iRemain+(j+1)); i++) {
            L[i+size*j] = (L[i+size*j] + A[i+size*j])/Dj;
        }
        for (i; i<size; i+=unrollFact) {
            L[i+size*j] = (L[i+size*j] + A[i+size*j])/Dj;
            L[(i+1)+size*j] = (L[(i+1)+size*j] + A[(i+1)+size*j])/Dj;
            L[(i+2)+size*j] = (L[(i+2)+size*j] + A[(i+2)+size*j])/Dj;
            L[(i+3)+size*j] = (L[(i+3)+size*j] + A[(i+3)+size*j])/Dj;
        }
        timeLij = get_wall_seconds() - time1;
    }

    LD_pair LD;
    LD.L = L;
    LD.D = D;
    return LD;
}

LD_pair cholDecomp_LD_blocks(double *A, int size, int blockSize) {

    int unrollFact = 4;
    if (blockSize%unrollFact != 0) {
        printf("[ERROR] Size of cache block (%d) not divisible"
               "by the loop unroll factor (%d)!\n", blockSize, unrollFact);
        exit(1);
    }
    double *L = (double *)calloc(size*size,sizeof(double));
    double *D = (double *)malloc(size*sizeof(double));

    double time1, timeDj, timeLij;
    double Dj, Lij;
    double factor;
    int numBlocks, blockRemain;
    int iBlock, iStart;
    int i,j,k;
    int iRemain, kRemain;
    for (j=0; j<size; j++) {

        time1 = get_wall_seconds();
        kRemain = j%unrollFact;
        Dj = A[j+size*j];
        for (k=0; k<kRemain; k++) {
            Dj -= L[j+size*k]*L[j+size*k]*D[k];
        }
        for (k; k<j; k+=unrollFact) {
            Dj -= L[j+size*k]*L[j+size*k]*D[k];
            Dj -= L[j+size*(k+1)]*L[j+size*(k+1)]*D[k+1];
            Dj -= L[j+size*(k+2)]*L[j+size*(k+2)]*D[k+2];
            Dj -= L[j+size*(k+3)]*L[j+size*(k+3)]*D[k+3];
        }
        D[j] = Dj;
        timeDj = get_wall_seconds() - time1;

        time1 = get_wall_seconds();
        L[j+size*j] = 1.0;
        numBlocks = (size-(j+1))/blockSize;
        blockRemain = (size-(j+1))%blockSize;

        for (k=0; k<j; k++) {
            factor = L[j+size*k]*D[k];

            // Loop over the block remainder first. If block
            // remainder is 0, then this nested loop is skipped.
            for (i=j+1; i<((j+1)+blockRemain); i++) {
                L[i+size*j] -= L[i+size*k]*factor;
            }

            // Loop over the blocks from "(j+1) + block remainder" to
            // the last row. If block remainder is 0, then this nested
            // loop starts from j+1 instead.
            for (iBlock=0; iBlock<numBlocks; iBlock++) {
                iStart=iBlock*blockSize + ((j+1)+blockRemain);
                for (i=iStart; i<(iStart+blockSize); i+=unrollFact) {
                    L[i+size*j] -= L[i+size*k]*factor;
                    L[(i+1)+size*j] -= L[(i+1)+size*k]*factor;
                    L[(i+2)+size*j] -= L[(i+2)+size*k]*factor;
                    L[(i+3)+size*j] -= L[(i+3)+size*k]*factor;
                }
            }
        }

        iRemain = (size-(j+1))%unrollFact;
        for (i=j+1; i<(iRemain+(j+1)); i++) {
            L[i+size*j] = (L[i+size*j] + A[i+size*j])/Dj;
        }
        for (i; i<size; i+=unrollFact) {
            L[i+size*j] = (L[i+size*j] + A[i+size*j])/Dj;
            L[(i+1)+size*j] = (L[(i+1)+size*j] + A[(i+1)+size*j])/Dj;
            L[(i+2)+size*j] = (L[(i+2)+size*j] + A[(i+2)+size*j])/Dj;
            L[(i+3)+size*j] = (L[(i+3)+size*j] + A[(i+3)+size*j])/Dj;
        }
        timeLij = get_wall_seconds() - time1;
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
    int iStart;
    for (int iBlock=0; iBlock<numBlocks; iBlock++) {
        iStart = iBlock*blockSize;
        for (int j=0; j<size; j++) {
            for (int i=iStart; i<(iStart+blockSize); i++) {
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

double *matMul_blocks(double *A, double *B, int size, int blockSize) {
    double *C = (double *)calloc(size*size,sizeof(double));

    int numBlocks = size/blockSize;
    int iBlock, jBlock, kBlock;
    int iStart, jStart, kStart;
    int i,j,k;
    double factor;
    for (jBlock=0; jBlock<numBlocks; jBlock++) {
        jStart = jBlock*blockSize;
        for (kBlock=0; kBlock<numBlocks; kBlock++) {
            kStart = kBlock*blockSize;
            for (iBlock=0; iBlock<numBlocks; iBlock++) {
                iStart = iBlock*blockSize;
                for (j=jStart; j<(jStart+blockSize); j++) {
                    for (k=kStart; k<(kStart+blockSize); k++) {
                        factor = B[k+size*j];
                        for (i=iStart; i<(iStart+blockSize); i++) {
                            C[i+size*j] += A[i+size*k]*factor;
                        }
                    }
                }
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

double *matMulDiag_blocks(double *A, double *D, int size, int blockSize) {
    double *C = (double *)calloc(size*size,sizeof(double));

    int numBlocks = size/blockSize;
    int iBlock, jBlock;
    int iStart, jStart;
    int i,j;
    double Dj;
    for (jBlock=0; jBlock<numBlocks; jBlock++) {
        jStart = jBlock*blockSize;
        for (iBlock=0; iBlock<numBlocks; iBlock++) {
            iStart = iBlock*blockSize;
            for (j=jStart; j<(jStart+blockSize); j++) {
                Dj = D[j];
                for (i=iStart; i<(iStart+blockSize); i++) {
                    C[i+size*j] = A[i+size*j]*Dj;
                }
            }
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
