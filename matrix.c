#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <pmmintrin.h>
#include <pthread.h>
#include <unistd.h>
#include "matrix.h"

extern int numThreads;
extern volatile int waitingThreadsCount;
extern pthread_mutex_t threadLock;
extern pthread_cond_t threadSignal;

void barrier() {
    pthread_mutex_lock(&threadLock);
    waitingThreadsCount++;
    if (waitingThreadsCount == (numThreads+1)) {
        waitingThreadsCount = 0;
        pthread_cond_broadcast(&threadSignal);
    }
    else {
        pthread_cond_wait(&threadSignal, &threadLock);
    }
    pthread_mutex_unlock(&threadLock);
}

void *calcLij_thread(void *myArgs) {

    thrArgs *args = (thrArgs *) myArgs;
    const int N = args->N;
    const int myID = args->thrID;
    const double *A = args->A;
    double *D = args->D;
    double *L = args->L;

    // 128-bit vector registers that each storeu 2 doubles (64 bits per double).
    __m128d factor_v;
    __m128d Aij_v[2], Lij_v[2], Lik_v[2]; // Vectors of 2 128-bit registers.

    //int i1, i2;
    int iVectRemain; // Remainder from vectorization.
    int i,j,k;
    double factor;

    for (j=0; j<N; j++) {

        int i1 = myID*(N-(j+1))/numThreads + (j+1);
        int i2 = (myID+1)*(N-(j+1))/numThreads + (j+1);
        //printf("[THREAD %d] i1 = %d, i2 = %d\n", args->thrID, i1, i2);
        iVectRemain = (i2-i1)%ELEMS_PER_iITER;
        barrier();

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

        barrier();
        // After calculating the (negative) Lik*Ljk*Dk sums,
        // add Aij to them and divide the result by Dj.
        double Dj = D[j];
        factor_v = _mm_set1_pd(Dj);
        for (i=i1; i<(iVectRemain+(i1)); i++) {
            //printf("[THREAD %d] i = %d, j = %d\n", args->thrID, i, j);
            L[i+N*j] = (L[i+N*j] + A[i+N*j])/Dj;
        }
        for (i; i<i2; i+=ELEMS_PER_iITER) {
            //printf("[THREAD %d] VECT i = %d, j = %d\n", args->thrID, i, j);
            Lij_v[0] = _mm_loadu_pd(L+(i+N*j));
            Aij_v[0] = _mm_loadu_pd(A+(i+N*j));
            Lij_v[0] = _mm_div_pd(_mm_add_pd(Lij_v[0],Aij_v[0]), factor_v);
            _mm_storeu_pd(L+(i+N*j), Lij_v[0]);

            Lij_v[1] = _mm_loadu_pd(L+(i+N*j)+ELEMS_PER_REG);
            Aij_v[1] = _mm_loadu_pd(A+(i+N*j)+ELEMS_PER_REG);
            Lij_v[1] = _mm_div_pd(_mm_add_pd(Lij_v[1],Aij_v[1]), factor_v);
            _mm_storeu_pd(L+(i+N*j)+ELEMS_PER_REG, Lij_v[1]);
        }
    }

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
    double* Matrix = (double *) malloc(N*N*sizeof(double));
    //double *Matrix = (double *) _mm_malloc(N*N*sizeof(double), 16);

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
