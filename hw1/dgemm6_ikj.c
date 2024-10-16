void dgemm6_ikj(double *C,double *A,double *B,int n)
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

void dgemm6_ikj2(double *C,double *A,double *B,int n)
{
    // complete the missing code here
     
}