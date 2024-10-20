void dgemm7_ikj(double *C,double *A,double *B,int n)
{
     int i,j,k;
     for(i=0;i<n;i++){
         for(k=0;k<n;k++){
             register double r = A[i*n+k];
             for(j=0;j<n;j++){
                 C[i*n+j] +=r *B[k*n+j];
             }
         }
     }

}

void dgemm7_ikj2(double *C,double *A,double *B,int n)
{
    int i,j,k;
    int ii,jj,kk;
    for(i=0;i<n;i+=BLOCK_SIZE)
    for(k=0;k<n;k+=BLOCK_SIZE)
    for(j=0;j<n;j+=BLOCK_SIZE)   
        for(ii=i;ii<i+BLOCK_SIZE;ii++){
        register int iin = ii*n;
        for(kk=k;kk<k+BLOCK_SIZE;kk++)
        {
            register double r = A[ii*n+kk];
            for(jj=j;jj<j+BLOCK_SIZE;jj++){
                C[iin+jj] += B[kk*n+jj] * r;
            }
        }
        }     
}
        