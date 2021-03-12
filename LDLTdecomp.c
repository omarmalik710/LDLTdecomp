#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <pmmintrin.h>
#include "matrix.h"

#define TESTPRINT 0

int numThreads;
volatile int waitingThreadsCount = 0;
pthread_mutex_t threadLock;
pthread_cond_t threadSignal;

int main(int argc, char **argv) {
    double time1, time2;

    int N;
    switch (argc) {
        case 2:
            N = atoi(argv[1]);
            numThreads = 1;
            break;
        case 3:
            N = atoi(argv[1]);
            numThreads = atoi(argv[2]);
            break;
        default:
            N = 2e3;
            numThreads = 2;
    }

    // Generate a random symmetric matrix.
    double *A = randSymm(N);

    // Start timing the LDLT decomposition.
    time1 = get_wall_seconds();

    // Initialize L and D.
    double *L = (double *) calloc(N*N,sizeof(double));
    double *D = (double *) calloc(N,sizeof(double));

    // Initialize each thread and its corresponding arguments.
    pthread_t *threads = (pthread_t *)malloc(numThreads*sizeof(pthread_t));
    thrArgs *args = (thrArgs *)malloc(numThreads*sizeof(thrArgs));

    // Get the locks and signals ready.
    pthread_mutex_init(&threadLock, NULL);
    pthread_cond_init(&threadSignal, NULL);

    // Set the threads loose.
    int j, k;
    for (int n=0; n<numThreads; n++) {
        args[n].A = A;
        args[n].D = D;
        args[n].L = L;
        args[n].N = N;
        args[n].thrID = n;
        pthread_create(threads+n, NULL, calcLij_thread, (void *) (args+n));
    }

    double Dj;
    for (j=0; j<N; j++) {
        //printf("[INFO] (N-(j+1))/numThreads = %d\n", (N-(j+1))/numThreads);
        //printf("[MASTER] j = %d\n", j);

        // 1st barrier: wait for all threads at the start of
        // each j iteration. Otherwise, a thread might use
        // Dj before the master thread finishes calculating it!
        barrier();
        Dj = A[j+N*j];
        for (k=0; k<j; k++) {
            Dj -= L[j+N*k]*L[j+N*k]*D[k];
        }
        D[j] = Dj;
        L[j+N*j] = 1.0;

        // 2nd barrier: wait for the threads to calculate Lij.
        // These are needed to calculate the next Dj and Lij!
        //printf("[MASTER] Now waiting for minions...\n");
        barrier();
        //printf("[MASTER] Waiting done. Back to work!\n");
    }

    time2 = get_wall_seconds();
    printf("%lf\n", time2-time1);

    #if TESTPRINT
    double *LxD = matMulDiag(L, D, N);
    double *LT = transpose(L, N);
    double *LxDxLT = matMul(LxD,LT, N);

    printf("A =\n");
    printMatrix(A, N);
    putchar('\n');

    printf("L =\n");
    printMatrix(L, N);
    putchar('\n');

    printf("D = \n");
    printArray(D, N);

    printf("L*D*LT =\n");
    printMatrix(LxDxLT, N);

    if (matEqual(A, LxDxLT, N, 1e-12)) {
        printf("A = L*D*LT :D\n");
    } else { printf("A != L*D*LT :(\n"); }
    #endif

    pthread_mutex_destroy(&threadLock);
    pthread_cond_destroy(&threadSignal);

    _mm_free(A);
    _mm_free(D);
    _mm_free(L);
    free(threads);
    free(args);

    return 0;
}