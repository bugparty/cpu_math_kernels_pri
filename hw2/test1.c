#include "include.h"
#include "func_call.c"
#include <stdlib.h>
void initRand(double*A,double*A_backup, double*B,double*B_backup,int n){
    srand(time(NULL));
        for (int i=0;i<n*n;i++)
    {
        A_backup[i]=((double)rand()/RAND_MAX)*2-1;
        A[i]=A_backup[i];
    }
    for (int i=0;i<n;i++)
    {
        B_backup[i]=((double)rand()/RAND_MAX)*2-1;
        B[i]=B_backup[i];
    }
}
int main(int argc,char **argv)
{
    if (argc!=3)
    {
        printf("Usage: ./main {func_name} {n}\n");
        exit(0);
    }
    char *func_name=argv[1];
    int n=atoi(argv[2]);
    FILE *pad_file=fopen("pad.txt","r");
    int pad;
    fscanf(pad_file,"%d",&pad);
    fclose(pad_file);
    FILE *file = fopen("matrix1.txt", "r");
    if (file == NULL) {
        printf("Can't open file\n");
        return 1;
    }
    fscanf(file, "%d", &n);
    n=((n+pad-1)/pad)*pad;
    printf("n=%d, pad=%d\n",n,pad);
    size_t alignment = 32;
    double *A_backup=(double*)_mm_malloc(n*n*sizeof(double), alignment);
    double *B_backup=(double*)_mm_malloc(n*sizeof(double), alignment);
    double *A=(double*)_mm_malloc(n*n*sizeof(double), alignment);
    double *B=(double*)_mm_malloc(n*sizeof(double), alignment);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fscanf(file, "%lf", &A[i*n+j]);
        }
    }
    printM(A,n,n);
    for (int i = 0; i < n; i++) {
        fscanf(file, "%lf", &B[i]);
    }


    struct timeval start,end;
    gettimeofday(&start,NULL);
    func_call(func_name,A,B,n);
    gettimeofday(&end,NULL);
    printf("time=%lfs\n",end.tv_sec-start.tv_sec+1e-6*(end.tv_usec-start.tv_usec));
    
    for (int i=0;i<n;i++)
    {
        double sum=0;
        for (int j=0;j<n;j++)
            sum+=A_backup[i*n+j]*B[j];
        if (fabs(sum-B_backup[i])>1e-5)
            printf("Error at row %d: standard=%lf, output=%lf\n",i,B_backup[i],sum);
    }
}