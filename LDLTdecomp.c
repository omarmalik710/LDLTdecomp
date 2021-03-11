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
    int iItersCutoff;
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
    time1 = get_wall_seconds();
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

    double Dj;
    int j, k;
    for (j=0; j<N; j++) {
        //printf("[INFO] (N-(j+1))/numThreads = %d\n", (N-(j+1))/numThreads);
        //printf("[MASTER] j = %d\n", j);
        Dj = A[j+N*j];
        for (k=0; k<j; k++) {
            Dj -= L[j+N*k]*L[j+N*k]*D[k];
        }
        D[j] = Dj;
        L[j+N*j] = 1.0;
        for (int n=0; n<numThreads; n++) {
            args[n].j = j;
            args[n].i1 = n*(N-(j+1))/numThreads + (j+1);
            //args[n].i2 = args[n].i1 + (N-(j+1))/numThreads;
            args[n].i2 = (n+1)*(N-(j+1))/numThreads + (j+1);
            //printf("[THREAD %d] i1 = %d, i2 = %d\n", n, args[n].i1, args[n].i2);
            pthread_create(threads+n, NULL, calcLij_thread, (void *) (args+n));
        }

        for (int n=0; n<numThreads; n++) {
            pthread_join(threads[n], NULL);
        }
    }

    time2 = get_wall_seconds();
    printf("[INFO] Time taken = %lf seconds.\n", time2-time1);
    //double *LxD = matMulDiag(L, D, N);
    //double *LT = transpose(L, N);
    //double *LxDxLT = matMul(LxD,LT, N);

    //printf("A =\n");
    //printMatrix(A, N);
    //putchar('\n');

    //printf("L =\n");
    //printMatrix(L, N);
    //putchar('\n');

    //printf("D = \n");
    //printArray(D, N);

    //printf("L*D*LT =\n");
    //printMatrix(LxDxLT, N);

    //if (matEqual(A, LxDxLT, N, 1e-12)) {
    //    printf("A = L*D*LT :D\n");
    //} else { printf("A != L*D*LT :(\n"); }

    pthread_mutex_destroy(&lock);
    pthread_cond_destroy(&signal);

    _mm_free(A);
    _mm_free(D);
    _mm_free(L);
    free(threads);
    free(args);

    return 0;
}