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
    int i,j,k;
    int ii,jj,kk;
    for(j=0;j<n;j+=BLOCK_SIZE)   
    for(i=0;i<n;i+=BLOCK_SIZE)
    for(k=0;k<n;k+=BLOCK_SIZE)
        
        for(jj=j;jj<j+BLOCK_SIZE;jj++)
        for(ii=i;ii<i+BLOCK_SIZE;ii++){
            register const int iin = ii*n;
            register double sum = C[ii*n+jj];
            for(kk=k;kk<k+BLOCK_SIZE;kk++)
            {
                sum += A[iin+kk] * B[kk*n+jj];
            }
            C[iin+jj]= sum;
        }
     
}