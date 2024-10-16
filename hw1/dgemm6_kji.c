void dgemm6_kji(double *C,double *A,double *B,int n)
{
int i,j,k;
for(k=0;k<n;k++){
for(j=0;j<n;j++){
    register double r = B[k*n+j];
    for(i=0;i<n;i++){
        C[i*n+j] += A[i*n+k] * r;
    }
        
    }} 
}

void dgemm6_kji2(double *C,double *A,double *B,int n)
{
    // complete the missing code here
     
}