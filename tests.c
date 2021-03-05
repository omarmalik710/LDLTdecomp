#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"

int main() {
    int size = 3000;
    double *A = randHerm(size);
    double *L = (double *)calloc(size*size,sizeof(double));

    int i,j,k;
    double t1, t2;
    double temp;
    double factor;
    t1 = get_wall_seconds();
    //for (j=0; j<size; j++) {
    //    for (i=j+1; i<size; i++) {
    //        temp = 0.0;
    //        for (k=0; k<j; k++) {
    //            temp += A[i+size*k]*A[j+size*k];
    //        }
    //        L[i+size*j] = temp/10;
    //    }
    //}
    //printMatrix(L, size);
    //putchar('\n');

    //free(L);
    //L = (double *)calloc(size*size,sizeof(double));
    for (j=0; j<size; j++) {
        for (k=0; k<j; k++) {
            factor = A[j+size*k]/10;
            for (i=j+1; i<size; i++) {
                L[i+size*j] += A[i+size*k]*factor;
            }
        }
        //L[i+size*j] /= 10;
    }
    t2 = get_wall_seconds();
    //printMatrix(L, size);
    printf("[TIME] Runtime = %lf\n", t2-t1);

    return 0;
}