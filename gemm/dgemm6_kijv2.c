void dgemm6_kij(double *C,double *A,double *B,int n)
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

void dgemm6_kij2(double *C,double *A,double *B,int n)
{
    int i,j,k;
    int ii,jj,kk;
    for(i=0;i<n;i+=BLOCK_SIZE)
        for(j=0;j<n;j+=BLOCK_SIZE)
            for(k=0;k<n;k+=BLOCK_SIZE)
                for(ii=i;ii<i+BLOCK_SIZE;ii++){
                    register int iin = ii*n;
                    for(jj=j;jj<j+BLOCK_SIZE;jj++){
                    register double r=C[iin+jj];
                    for(kk=k;kk<k+BLOCK_SIZE;kk++){
                        r+=A[iin+kk]*B[kk*n+jj]; 
                    }
                    C[iin+jj] = r;
                    }
                }
                
     
}