void dgemm6_ijk(double *C,double *A,double *B,int n)
{
     int i,j,k;
     for(i=0;i<n;i++){
         for(j=0;j<n;j++){
             register double r = C[i*n+j];
             for(k=0;k<n;k++){
                 r+=A[i*n+k]*B[k*n+j];
             }
             C[i*n+j]=r;
         }
     }

}

void dgemm6_ijk2(double *C,double *A,double *B,int n)
{
     int i,j,k;
     int ii,jj,kk;
     for(i=0;i<n;i+=BLOCK_SIZE)
         for(j=0;j<n;j+=BLOCK_SIZE)
             for(k=0;k<n;k+=BLOCK_SIZE)
                 for(ii=i;ii<i+BLOCK_SIZE;ii++){
                    register const int iin = ii*n;
                    for(jj=j;jj<j+BLOCK_SIZE;jj++){
                        register double r=C[ii*n+jj];
                        for(kk=k;kk<k+BLOCK_SIZE;kk++){
                            r+=A[iin+kk]*B[kk*n+jj]; 
                        }
                        C[iin+jj] = r;
                    }
                 }
                    
}