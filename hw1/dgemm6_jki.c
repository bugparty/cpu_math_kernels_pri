void dgemm6_jki(double *C,double *A,double *B,int n)
{
   int i,j,k;
for(j=0;j<n;j++){
    for(k=0;k<n;k++){
        register double r = B[k*n+j];
        for(i=0;i<n;i++){
            C[i*n+j] += A[i*n+k] * r;
        }
    }} 
     
}

void dgemm6_jki2(double *C,double *A,double *B,int n)
{
    // complete the missing code here
     
}