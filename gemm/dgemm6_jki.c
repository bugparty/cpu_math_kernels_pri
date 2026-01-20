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
    int i,j,k;
    int ii,jj,kk;
    for(j=0;j<n;j+=BLOCK_SIZE)
    for(k=0;k<n;k+=BLOCK_SIZE)   
    for(i=0;i<n;i+=BLOCK_SIZE)
        for(jj=j;jj<j+BLOCK_SIZE;jj++)
        for(kk=k;kk<k+BLOCK_SIZE;kk++)
        {
            register double r = B[kk*n+jj];
            for(ii=i;ii<i+BLOCK_SIZE;ii++)
            {
                 C[ii*n+jj] += A[ii*n+kk] * r;
            }
        }
}