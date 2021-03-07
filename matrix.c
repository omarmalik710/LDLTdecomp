#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pmmintrin.h>
#include "matrix.h"

LD_pair cholDecomp_LD(double* restrict A, const int size) {

    double *L = (double *)calloc(size*size,sizeof(double));
    double *D = (double *)malloc(size*sizeof(double));

    double time1, timeDj, timeLij;
    double Dj;
    double factor;

    __m128d factor_v;
    __m128d Aij_v[2], Lij_v[2], Lik_v[2];

    int i,j,k;
    int iVectRemain, kUnRollRemain;
    for (j=0; j<size; j++) {

        //time1 = get_wall_seconds();
        kUnRollRemain = j%UNROLL_FACT;
        Dj = A[j+size*j];
        for (k=0; k<kUnRollRemain; k++) {
            Dj -= L[j+size*k]*L[j+size*k]*D[k];
        }
        for (k; k<j; k+=UNROLL_FACT) {
            Dj -= L[j+size*k]*L[j+size*k]*D[k];
            Dj -= L[j+size*(k+1)]*L[j+size*(k+1)]*D[k+1];
            Dj -= L[j+size*(k+2)]*L[j+size*(k+2)]*D[k+2];
            Dj -= L[j+size*(k+3)]*L[j+size*(k+3)]*D[k+3];
        }
        D[j] = Dj;
        //timeDj = get_wall_seconds() - time1;

        //time1 = get_wall_seconds();
        iVectRemain = (size-(j+1))%(2*ELEMS_PER_REG);
        L[j+size*j] = 1.0;
        for (k=0; k<j; k++) {
            factor = L[j+size*k]*D[k];
            factor_v = _mm_set1_pd(factor);
            for (i=j+1; i<(iVectRemain+(j+1)); i++) {
                L[i+size*j] -= L[i+size*k]*factor;
            }
            for (i; i<size; i+=(2*ELEMS_PER_REG)) {
                Lij_v[0] = _mm_load_pd(L+(i+size*j));
                Lik_v[0] = _mm_load_pd(L+(i+size*k));
                Lij_v[0] = _mm_sub_pd(Lij_v[0], _mm_mul_pd(Lik_v[0],factor_v));
                _mm_store_pd(L+(i+size*j), Lij_v[0]);

                Lij_v[1] = _mm_load_pd(L+(i+size*j)+ELEMS_PER_REG);
                Lik_v[1] = _mm_load_pd(L+(i+size*k)+ELEMS_PER_REG);
                Lij_v[1] = _mm_sub_pd(Lij_v[1], _mm_mul_pd(Lik_v[1],factor_v));
                _mm_store_pd(L+(i+size*j)+ELEMS_PER_REG, Lij_v[1]);
            }
        }

        iVectRemain = (size-(j+1))%(2*ELEMS_PER_REG);
        factor_v = _mm_set1_pd(Dj);
        for (i=j+1; i<(iVectRemain+(j+1)); i++) {
            L[i+size*j] = (L[i+size*j] + A[i+size*j])/Dj;
        }
        for (i; i<size; i+=(2*ELEMS_PER_REG)) {
            Lij_v[0] = _mm_load_pd(L+(i+size*j));
            Aij_v[0] = _mm_load_pd(A+(i+size*j));
            Lij_v[0] = _mm_div_pd(_mm_add_pd(Lij_v[0],Aij_v[0]), factor_v);
            _mm_store_pd(L+(i+size*j), Lij_v[0]);

            Lij_v[1] = _mm_load_pd(L+(i+size*j)+ELEMS_PER_REG);
            Aij_v[1] = _mm_load_pd(A+(i+size*j)+ELEMS_PER_REG);
            Lij_v[1] = _mm_div_pd(_mm_add_pd(Lij_v[1],Aij_v[1]), factor_v);
            _mm_store_pd(L+(i+size*j)+ELEMS_PER_REG, Lij_v[1]);
        }
        //timeLij = get_wall_seconds() - time1;
    }

    LD_pair LD;
    LD.L = L;
    LD.D = D;
    return LD;
}

LD_pair cholDecomp_LD_blocks(double* restrict A, const int size, const int blockSize) {

    if (blockSize%UNROLL_FACT != 0) {
        printf("[ERROR] Size of cache block (%d) not divisible"
               "by the loop unroll factor (%d)!\n", blockSize, UNROLL_FACT);
        exit(1);
    }
    if (blockSize%ELEMS_PER_REG != 0) {
        printf("[ERROR] Size of cache block (%d) not divisible"
               "by the number of elements per vector register! (%d)!\n",
               blockSize, ELEMS_PER_REG);
        exit(1);
    }
    double *L = (double *)calloc(size*size,sizeof(double));
    double *D = (double *)malloc(size*sizeof(double));

    double time1, timeDj, timeLij;
    double Dj, Lij;
    double factor;

    int numBlocks, blockRemain;
    int iBlock, iStart;

    __m128d factor_v;
    __m128d Aij_v[2], Lij_v[2], Lik_v[2];

    int i,j,k;
    int iVectRemain, kUnrollRemain;
    for (j=0; j<size; j++) {

        time1 = get_wall_seconds();
        kUnrollRemain = j%UNROLL_FACT;
        Dj = A[j+size*j];
        for (k=0; k<kUnrollRemain; k++) {
            Dj -= L[j+size*k]*L[j+size*k]*D[k];
        }
        for (k; k<j; k+=UNROLL_FACT) {
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

            // Loop over the block remainder first. If block
            // remainder is 0, then this nested loop is skipped.
            factor = L[j+size*k]*D[k];
            for (i=j+1; i<((j+1)+blockRemain); i++) {
                L[i+size*j] -= L[i+size*k]*factor;
            }

            // Loop over the blocks from "(j+1) + block remainder" to
            // the last row. If block remainder is 0, then this nested
            // loop starts from j+1 instead.
            factor_v = _mm_set1_pd(factor);
            for (iBlock=0; iBlock<numBlocks; iBlock++) {
                iStart=iBlock*blockSize + ((j+1)+blockRemain);
                for (i=iStart; i<(iStart+blockSize); i+=(2*ELEMS_PER_REG)) {
                    Lij_v[0] = _mm_load_pd(L+(i+size*j));
                    Lik_v[0] = _mm_load_pd(L+(i+size*k));
                    Lij_v[0] = _mm_sub_pd(Lij_v[0], _mm_mul_pd(Lik_v[0],factor_v));
                    _mm_store_pd(L+(i+size*j), Lij_v[0]);

                    Lij_v[1] = _mm_load_pd(L+(i+size*j)+ELEMS_PER_REG);
                    Lik_v[1] = _mm_load_pd(L+(i+size*k)+ELEMS_PER_REG);
                    Lij_v[1] = _mm_sub_pd(Lij_v[1], _mm_mul_pd(Lik_v[1],factor_v));
                    _mm_store_pd(L+(i+size*j)+ELEMS_PER_REG, Lij_v[1]);
                }
            }
        }

        iVectRemain = (size-(j+1))%(2*ELEMS_PER_REG);
        factor_v = _mm_set1_pd(Dj);
        for (i=j+1; i<(iVectRemain+(j+1)); i++) {
            L[i+size*j] = (L[i+size*j] + A[i+size*j])/Dj;
        }
        for (i; i<size; i+=(2*ELEMS_PER_REG)) {
            Lij_v[0] = _mm_load_pd(L+(i+size*j));
            Aij_v[0] = _mm_load_pd(A+(i+size*j));
            Lij_v[0] = _mm_div_pd(_mm_add_pd(Lij_v[0],Aij_v[0]), factor_v);
            _mm_store_pd(L+(i+size*j), Lij_v[0]);

            Lij_v[1] = _mm_load_pd(L+(i+size*j)+ELEMS_PER_REG);
            Aij_v[1] = _mm_load_pd(A+(i+size*j)+ELEMS_PER_REG);
            Lij_v[1] = _mm_div_pd(_mm_add_pd(Lij_v[1],Aij_v[1]), factor_v);
            _mm_store_pd(L+(i+size*j)+ELEMS_PER_REG, Lij_v[1]);
        }
        timeLij = get_wall_seconds() - time1;
    }

    LD_pair LD;
    LD.L = L;
    LD.D = D;
    return LD;
}

double *transpose(double* restrict A, const int size) {
    double* restrict AT = (double *)calloc(size*size,sizeof(double));

    for (int i=0; i<size; i++) {
        for (int j=0; j<size; j++) {
            AT[i+size*j] = A[j+size*i];
        }
    }

    return AT;
}

double *transpose_blocks(double* restrict A, const int size, const int blockSize) {

    double* AT = (double *)calloc(size*size,sizeof(double));
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

double *randHerm(const int size) {
    double* Matrix = (double *) malloc(size*size*sizeof(double));

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

int isHerm(double* restrict Matrix, const int size) {
    for (int i=0; i<size; i++) {
        for (int j=0; j<i; j++) {
            if (Matrix[i+size*j] != Matrix[j+size*i]) { return 0; }
        }
    }

    return 1;
}

double *matMul(double* restrict A, double* restrict B, const int size) {
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

double *matMul_blocks(double* restrict A, double* restrict B, const int size, const int blockSize) {
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

double *matMulDiag(double* restrict A, double *D, const int size) {
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

double *matMulDiag_blocks(double* restrict A, double *D, const int size, const int blockSize) {
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

int matEqual(double* restrict A, double* restrict B, const int size, const double tol) {
    for (int i=0; i<size; i++) {
        for (int j=0; j<size; j++) {
            if ( abs(A[i+size*j]-B[i+size*j]) > tol ) {
                return 0;
            }
        }
    }

    return 1;
}

void printMatrix(double* restrict Matrix, const int size) {
    for (int i=0; i<size; i++) {
        for (int j=0; j<size; j++) {
            printf("%10.6lf ", Matrix[i+size*j]);
        }
        putchar('\n');
    }
}

void printArray(double* restrict array, const int size) {
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
