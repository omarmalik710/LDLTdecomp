#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <pmmintrin.h>
#include <pthread.h>
#include "matrix.h"

extern int numThreads;
extern int waitingThreadsCount;
extern pthread_mutex_t lock;
extern pthread_cond_t signal;

void barrier() {
    pthread_mutex_lock(&lock);
    waitingThreadsCount++;
    if (waitingThreadsCount == numThreads) {
        waitingThreadsCount = 0;
        pthread_cond_broadcast(&signal);
    }
    else {
        pthread_cond_wait(&signal, &lock);
    }
    pthread_mutex_unlock(&lock);
}

void *calcLij_thread(void *myArgs) {

    thrArgs *args = (thrArgs *) myArgs;
    const int N = args->N;
    const int j = args->j;
    double *A = args->A;
    double *D = args->D;
    double *L = args->L;

    // 128-bit vector registers that each storeu 2 doubles (64 bits per double).
    __m128d factor_v;
    __m128d Aij_v[2], Lij_v[2], Lik_v[2]; // Vectors of 2 128-bit registers.

    const int ID = args->thrID;
    const int i1 = args->i1;
    const int i2 = args->i2;
    const int iVectRemain = (i2-i1)%ELEMS_PER_iITER; // Remainder from vectorization.
    int i,k;
    double factor;
    for (k=0; k<j; k++) {
        // Calculate the (negative) Lik*Ljk*Dk sums.
        factor = L[j+N*k]*D[k];
        factor_v = _mm_set1_pd(factor);
        for (i=i1; i<(iVectRemain+i1); i++) {
            L[i+N*j] -= L[i+N*k]*factor;
        }
        for (i; i<i2; i+=ELEMS_PER_iITER) {
            Lij_v[0] = _mm_loadu_pd(L+(i+N*j));
            Lik_v[0] = _mm_loadu_pd(L+(i+N*k));
            Lij_v[0] = _mm_sub_pd(Lij_v[0], _mm_mul_pd(Lik_v[0],factor_v));
            _mm_storeu_pd(L+(i+N*j), Lij_v[0]);

            Lij_v[1] = _mm_loadu_pd(L+(i+N*j)+ELEMS_PER_REG);
            Lik_v[1] = _mm_loadu_pd(L+(i+N*k)+ELEMS_PER_REG);
            Lij_v[1] = _mm_sub_pd(Lij_v[1], _mm_mul_pd(Lik_v[1],factor_v));
            _mm_storeu_pd(L+(i+N*j)+ELEMS_PER_REG, Lij_v[1]);
        }
    }

    // After calculating the (negative) Lik*Ljk*Dk sums,
    // add Aij to them and divide the result by Dj.
    //const double Dj = D[j];
    //factor_v = _mm_set1_pd(Dj);
    //for (i=i1; i<(iVectRemain+(i1)); i++) {
    //    L[i+N*j] = (L[i+N*j] + A[i+N*j])/Dj;
    //}
    //for (i; i<i2; i+=ELEMS_PER_iITER) {
    //    Lij_v[0] = _mm_loadu_pd(L+(i+N*j));
    //    Aij_v[0] = _mm_loadu_pd(A+(i+N*j));
    //    Lij_v[0] = _mm_div_pd(_mm_add_pd(Lij_v[0],Aij_v[0]), factor_v);
    //    _mm_storeu_pd(L+(i+N*j), Lij_v[0]);

    //    Lij_v[1] = _mm_loadu_pd(L+(i+N*j)+ELEMS_PER_REG);
    //    Aij_v[1] = _mm_loadu_pd(A+(i+N*j)+ELEMS_PER_REG);
    //    Lij_v[1] = _mm_div_pd(_mm_add_pd(Lij_v[1],Aij_v[1]), factor_v);
    //    _mm_storeu_pd(L+(i+N*j)+ELEMS_PER_REG, Lij_v[1]);
    //}

    barrier();
}

LD_pair LDLTdecomp(double* restrict A, const int N) {

    double *L = (double *)calloc(N*N,sizeof(double));
    double *D = (double *)malloc(N*sizeof(double));

    double time1, timeDj, timeLij;
    double Dj;
    double factor;

    // 128-bit vector registers that each storeu 2 doubles (64 bits per double).
    __m128d factor_v;
    __m128d Aij_v[2], Lij_v[2], Lik_v[2]; // Vectors of 2 128-bit registers.

    int i,j,k;
    int iVectRemain; // Remainder for vectorization.
    for (j=0; j<N; j++) {

        //time1 = get_wall_seconds();
        Dj = A[j+N*j];
        for (k=0; k<j; k++) {
            Dj -= L[j+N*k]*L[j+N*k]*D[k];
        }
        D[j] = Dj;
        //timeDj = get_wall_seconds() - time1;

        //time1 = get_wall_seconds();
        // The i-loops go from 'j+1' to 'N', so we need to iterate
        // over the remainder of their difference with the total number
        // of elements in the vector registers first. (Two registers are
        // used in each iteration here.) Then we can vectorize the rest.
        iVectRemain = (N-(j+1))%ELEMS_PER_iITER;
        L[j+N*j] = 1.0;
        for (k=0; k<j; k++) {
            // Calculate the (negative) Lik*Ljk*Dk sums.
            factor = L[j+N*k]*D[k];
            factor_v = _mm_set1_pd(factor);
            for (i=j+1; i<(iVectRemain+(j+1)); i++) {
                L[i+N*j] -= L[i+N*k]*factor;
            }
            for (i; i<N; i+=ELEMS_PER_iITER) {
                Lij_v[0] = _mm_loadu_pd(L+(i+N*j));
                Lik_v[0] = _mm_loadu_pd(L+(i+N*k));
                Lij_v[0] = _mm_sub_pd(Lij_v[0], _mm_mul_pd(Lik_v[0],factor_v));
                _mm_storeu_pd(L+(i+N*j), Lij_v[0]);

                Lij_v[1] = _mm_loadu_pd(L+(i+N*j)+ELEMS_PER_REG);
                Lik_v[1] = _mm_loadu_pd(L+(i+N*k)+ELEMS_PER_REG);
                Lij_v[1] = _mm_sub_pd(Lij_v[1], _mm_mul_pd(Lik_v[1],factor_v));
                _mm_storeu_pd(L+(i+N*j)+ELEMS_PER_REG, Lij_v[1]);
            }
        }

        // After calculating the (negative) Lik*Ljk*Dk sums,
        // add Aij to them and divide the result by Dj.
        factor_v = _mm_set1_pd(Dj);
        for (i=j+1; i<(iVectRemain+(j+1)); i++) {
            L[i+N*j] = (L[i+N*j] + A[i+N*j])/Dj;
        }
        for (i; i<N; i+=ELEMS_PER_iITER) {
            Lij_v[0] = _mm_loadu_pd(L+(i+N*j));
            Aij_v[0] = _mm_loadu_pd(A+(i+N*j));
            Lij_v[0] = _mm_div_pd(_mm_add_pd(Lij_v[0],Aij_v[0]), factor_v);
            _mm_storeu_pd(L+(i+N*j), Lij_v[0]);

            Lij_v[1] = _mm_loadu_pd(L+(i+N*j)+ELEMS_PER_REG);
            Aij_v[1] = _mm_loadu_pd(A+(i+N*j)+ELEMS_PER_REG);
            Lij_v[1] = _mm_div_pd(_mm_add_pd(Lij_v[1],Aij_v[1]), factor_v);
            _mm_storeu_pd(L+(i+N*j)+ELEMS_PER_REG, Lij_v[1]);
        }
        //timeLij = get_wall_seconds() - time1;
    }

    LD_pair LD;
    LD.L = L;
    LD.D = D;
    return LD;
}

LD_pair LDLTdecomp_blocks(double* restrict A, const int N, const int blockSize) {

    if (blockSize%UNROLL_FACT != 0) {
        printf("[ERROR] Size of cache block (%d) not divisible "
               "by the loop unroll factor (%d)!\n", blockSize, UNROLL_FACT);
        exit(1);
    }
    if (blockSize%ELEMS_PER_REG != 0) {
        printf("[ERROR] Size of cache block (%d) not divisible"
               "by the number of elements per vector register! (%d)!\n",
               blockSize, ELEMS_PER_REG);
        exit(1);
    }
    double *L = (double *)calloc(N*N,sizeof(double));
    double *D = (double *)malloc(N*sizeof(double));

    double time1, timeDj, timeLij;
    double Dj, Lij;
    double factor;

    int numBlocks, blockRemain;
    int iBlock, iStart;

    // 128-bit vector registers that each storeu 2 doubles (64 bits per double).
    __m128d factor_v;
    __m128d Aij_v[2], Lij_v[2], Lik_v[2]; // Vectors of 2 128-bit registers.

    int i,j,k;
    int iVectRemain, kUnrollRemain; // Remainders for vectorization and loop unrolling.
    for (j=0; j<N; j++) {

        time1 = get_wall_seconds();
        kUnrollRemain = j%UNROLL_FACT;
        Dj = A[j+N*j];
        // Deal with the remainder terms first, then loop unroll afterwards.
        for (k=0; k<kUnrollRemain; k++) {
            Dj -= L[j+N*k]*L[j+N*k]*D[k];
        }
        for (k; k<j; k+=UNROLL_FACT) {
            Dj -= L[j+N*k]*L[j+N*k]*D[k];
            Dj -= L[j+N*(k+1)]*L[j+N*(k+1)]*D[k+1];
            Dj -= L[j+N*(k+2)]*L[j+N*(k+2)]*D[k+2];
            Dj -= L[j+N*(k+3)]*L[j+N*(k+3)]*D[k+3];
            Dj -= L[j+N*(k+4)]*L[j+N*(k+4)]*D[k+4];
        }
        D[j] = Dj;
        timeDj = get_wall_seconds() - time1;

        time1 = get_wall_seconds();

        // The block loops go from 'j+1' to 'N', so we need to iterate
        // over the remainder of their difference with the total number
        // of elements in the vector registers first. (Two registers are
        // used in each iteration here.) Then we can vectorize the rest.
        L[j+N*j] = 1.0;
        numBlocks = (N-(j+1))/blockSize;
        blockRemain = (N-(j+1))%blockSize;
        for (k=0; k<j; k++) {

            // Calculate the (negative) Lik*Ljk*Dk sums for any
            // remainder rows before looping over the blocks.
            factor = L[j+N*k]*D[k];
            for (i=j+1; i<((j+1)+blockRemain); i++) {
                L[i+N*j] -= L[i+N*k]*factor;
            }

            // Loop over the rest of the rows in blocks and
            // calculate their (negative) Lik*Ljk*Dk sums.
            factor_v = _mm_set1_pd(factor);
            for (iBlock=0; iBlock<numBlocks; iBlock++) {
                iStart=iBlock*blockSize + ((j+1)+blockRemain);
                for (i=iStart; i<(iStart+blockSize); i+=(2*ELEMS_PER_REG)) {
                    Lij_v[0] = _mm_loadu_pd(L+(i+N*j));
                    Lik_v[0] = _mm_loadu_pd(L+(i+N*k));
                    Lij_v[0] = _mm_sub_pd(Lij_v[0], _mm_mul_pd(Lik_v[0],factor_v));
                    _mm_storeu_pd(L+(i+N*j), Lij_v[0]);

                    Lij_v[1] = _mm_loadu_pd(L+(i+N*j)+ELEMS_PER_REG);
                    Lik_v[1] = _mm_loadu_pd(L+(i+N*k)+ELEMS_PER_REG);
                    Lij_v[1] = _mm_sub_pd(Lij_v[1], _mm_mul_pd(Lik_v[1],factor_v));
                    _mm_storeu_pd(L+(i+N*j)+ELEMS_PER_REG, Lij_v[1]);
                }
            }
        }

        // After calculating the (negative) Lik*Ljk*Dk sums,
        // add Aij to them and divide the result by Dj.
        iVectRemain = (N-(j+1))%(2*ELEMS_PER_REG);
        factor_v = _mm_set1_pd(Dj);
        for (i=j+1; i<(iVectRemain+(j+1)); i++) {
            L[i+N*j] = (L[i+N*j] + A[i+N*j])/Dj;
        }
        for (i; i<N; i+=(2*ELEMS_PER_REG)) {
            Lij_v[0] = _mm_loadu_pd(L+(i+N*j));
            Aij_v[0] = _mm_loadu_pd(A+(i+N*j));
            Lij_v[0] = _mm_div_pd(_mm_add_pd(Lij_v[0],Aij_v[0]), factor_v);
            _mm_storeu_pd(L+(i+N*j), Lij_v[0]);

            Lij_v[1] = _mm_loadu_pd(L+(i+N*j)+ELEMS_PER_REG);
            Aij_v[1] = _mm_loadu_pd(A+(i+N*j)+ELEMS_PER_REG);
            Lij_v[1] = _mm_div_pd(_mm_add_pd(Lij_v[1],Aij_v[1]), factor_v);
            _mm_storeu_pd(L+(i+N*j)+ELEMS_PER_REG, Lij_v[1]);
        }
        timeLij = get_wall_seconds() - time1;
    }

    LD_pair LD;
    LD.L = L;
    LD.D = D;
    return LD;
}

double *transpose(double* restrict A, const int N) {
    double* restrict AT = (double *)calloc(N*N,sizeof(double));

    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            AT[i+N*j] = A[j+N*i];
        }
    }

    return AT;
}

double *transpose_blocks(double* restrict A, const int N, const int blockSize) {

    double* AT = (double *)calloc(N*N,sizeof(double));
    int numBlocks = N/blockSize;
    int iStart;
    for (int iBlock=0; iBlock<numBlocks; iBlock++) {
        iStart = iBlock*blockSize;
        for (int j=0; j<N; j++) {
            for (int i=iStart; i<(iStart+blockSize); i++) {
                AT[i+N*j] = A[j+N*i];
            }
        }
    }

    return AT;
}

double *randHerm(const int N) {
    //double* Matrix = (double *) malloc(N*N*sizeof(double));
    double *Matrix = (double *) _mm_malloc(N*N*sizeof(double), 16);

    time_t t;
    srand((unsigned) time(&t));
    int i,j;
    for (i=0; i<N; i++) {
        // Ensure that the diagonal elements are larger than the rest
        // to get a more likely (semi-)positive definite Hermitian matrix.
        Matrix[i+N*i] = rand()%20;
        for (j=i+1; j<N; j++) {
            Matrix[i+N*j] = rand()%5;
            Matrix[j+N*i] = Matrix[i+N*j];
        }
    }

    return Matrix;
}

int isHerm(double* restrict Matrix, const int N) {
    for (int i=0; i<N; i++) {
        for (int j=0; j<i; j++) {
            if (Matrix[i+N*j] != Matrix[j+N*i]) { return 0; }
        }
    }

    return 1;
}

double *matMul(double* restrict A, double* restrict B, const int N) {
    double *C = (double *)calloc(N*N,sizeof(double));
    double factor;

    for (int j=0; j<N; j++) {
        for (int k=0; k<N; k++) {
            factor = B[k+N*j];
            for (int i=0; i<N; i++) {
                C[i+N*j] += A[i+N*k]*factor;
            }
        }
    }

    return C;
}

double *matMul_blocks(double* restrict A, double* restrict B, const int N, const int blockSize) {
    double *C = (double *)calloc(N*N,sizeof(double));

    int numBlocks = N/blockSize;
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
                        factor = B[k+N*j];
                        for (i=iStart; i<(iStart+blockSize); i++) {
                            C[i+N*j] += A[i+N*k]*factor;
                        }
                    }
                }
            }
        }
    }

    return C;
}

double *matMulDiag(double* restrict A, double *D, const int N) {
    double *C = (double *)calloc(N*N,sizeof(double));
    double Dj;

    for (int j=0; j<N; j++) {
        Dj = D[j];
        for (int i=0; i<N; i++) {
            C[i+N*j] = A[i+N*j]*Dj;
        }
    }

    return C;
}

double *matMulDiag_blocks(double* restrict A, double *D, const int N, const int blockSize) {
    double *C = (double *)calloc(N*N,sizeof(double));

    int numBlocks = N/blockSize;
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
                    C[i+N*j] = A[i+N*j]*Dj;
                }
            }
        }
    }

    return C;
}

int matEqual(double* restrict A, double* restrict B, const int N, const double tol) {
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            if ( abs(A[i+N*j]-B[i+N*j]) > tol ) {
                return 0;
            }
        }
    }

    return 1;
}

void printMatrix(double* restrict Matrix, const int N) {
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            printf("%10.6lf ", Matrix[i+N*j]);
        }
        putchar('\n');
    }
}

void printArray(double* restrict array, const int N) {
    for (int i=0; i<N; i++) {
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
