#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <pmmintrin.h>
#include "matrix.h"

int numThreads;
int waitingThreadsCount = 0;
pthread_mutex_t lock;
pthread_cond_t signal;

int main(int argc, char **argv) {
    double time1, time2;

    int N;
    int blockSize;
    switch (argc) {
        case 2:
            N = atoi(argv[1]);
            blockSize = 10;
            numThreads = 1;
            break;
        case 3:
            N = atoi(argv[1]);
            blockSize = atoi(argv[2]);
            numThreads = 1;
            break;
        case 4:
            N = atoi(argv[1]);
            blockSize = atoi(argv[2]);
            numThreads = atoi(argv[3]);
            break;
        default:
            N = 2e3;
            blockSize = 10;
            numThreads = 1;
    }

    double *A = randHerm(N);
    double *L = (double *) _mm_malloc(N*N*sizeof(double), 16);
    for (int j=0; j<N; j++) {
        for (int i=0; i<N; i++) { L[i+N*j] = 0.0; }
    }
    double *D = (double *) _mm_malloc(N*sizeof(double), 16);
    for (int j=0; j<N; j++) { D[j] = 0.0; }

    pthread_t *threads = (pthread_t *)malloc(numThreads*sizeof(pthread_t));
    thrArgs *args = (thrArgs *)malloc(numThreads*sizeof(thrArgs));

    for (int n=0; n<numThreads; n++) {
        args[n].A = A; args[n].D = D; args[n].L = L;
        args[n].N = N; args[n].thrID = n;
    }

    pthread_mutex_init(&lock, NULL);
    pthread_cond_init(&signal, NULL);

    // Handle part of A that can be distributed evenly amongst
    // the threads.
    int iItersCutoff = 1;
    int minThreadElems = iItersCutoff*ELEMS_PER_iITER;
    int threadRemain = numThreads*minThreadElems;
    double Dj;
    int j, k;
    for (j=0; j<(N-threadRemain); j++) {
        //printf("[MASTER] j = %d\n", j);
        //time1 = get_wall_seconds();
        Dj = A[j+N*j];
        for (k=0; k<j; k++) {
            Dj -= L[j+N*k]*L[j+N*k]*D[k];
        }
        D[j] = Dj;
        L[j+N*j] = 1.0;
        for (int n=0; n<numThreads; n++) {
            args[n].j = j;
            args[n].i1 = n*(N-j)/numThreads + (j+1);
            args[n].i2 = args[n].i1 + (N-j)/numThreads;
            printf("[THREAD %d] i1 = %d, i2 = %d\n", n, args[n].i1, args[n].i2);
        }
        for (int n=0; n<numThreads; n++) {
            //args[n].j = j;
            //args[n].i1 = n*(N-j)/numThreads + (j+1);
            //args[n].i2 = args[n].i1 + (N-j)/numThreads;
            pthread_create(threads+n, NULL, calcLij_thread, (void *) (args+n));
        }

        for (int n=0; n<numThreads; n++) {
            pthread_join(threads[n], NULL);
        }
    }

    // From this point on, I deduce that the number of
    // elements remaining isn't worth the overhead for
    // repeated thread creation/destruction, so the master
    // thread (this one) finishes the remaining decomposition.

    // 128-bit vector registers that each storeu 2 doubles (64 bits per double).
    __m128d factor_v;
    __m128d Aij_v[2], Lij_v[2], Lik_v[2]; // Vectors of 2 128-bit registers.

    double factor;
    int i;
    int iVectRemain; // Remainder for vectorization.
    for (j; j<N; j++) {

        //printf("[MASTER] j = %d\n", j);
        Dj = A[j+N*j];
        for (k=0; k<j; k++) {
            Dj -= L[j+N*k]*L[j+N*k]*D[k];
        }
        D[j] = Dj;

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
    }

    double *LxD = matMulDiag(L, D, N);
    double *LT = transpose(L, N);
    double *LxDxLT = matMul(LxD,LT, N);

    printf("A =\n");
    printMatrix(A, N);
    putchar('\n');

    printf("D = \n");
    printArray(D, N);

    printf("L*D*LT =\n");
    printMatrix(LxDxLT, N);

    if (matEqual(A, LxDxLT, N, 1e-12)) {
        printf("A = L*D*LT :D\n");
    } else { printf("A != L*D*LT :(\n"); }

    pthread_mutex_destroy(&lock);
    pthread_cond_destroy(&signal);

    _mm_free(A);
    _mm_free(D);
    _mm_free(L);
    free(threads);
    free(args);

    return 0;
}