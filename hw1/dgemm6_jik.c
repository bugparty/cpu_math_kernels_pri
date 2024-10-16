void dgemm6_jik(double *C,double *A,double *B,int n)
{
int i,j,k;
for(j=0;j<n;j++){
    for(i=0;i<n;i++){
        register double sum = C[i*n+j];
        for(k=0;k<n;k++){
            sum += A[i*n+k] * B[k*n+j];
        }
        C[i*n+j]= sum;
    }} 
}

void dgemm6_jik2(double *C,double *A,double *B,int n)
{
    // complete the missing code here
     
}