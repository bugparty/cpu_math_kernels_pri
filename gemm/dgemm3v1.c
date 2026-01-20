void dgemm3(double *C,double *A,double *B,int n)
{
int i,j,k;
for(int i=0;i<n;i+=2){
     for(int j=0;j<n;j+=2){
          register double c00,c01,c10,c11;
          register int t1,t2;
          t1 = i*n+j;t2 = t1+n;
          c00=C[t1];c01=C[t1+1];c10=C[t2];c11=C[t2+1];
          for(int k=0;k<n;k+=2){
               register int ta = i*n+k; register int tta = ta+n; register int tb = k*n+j; register int ttb = tb+n;
               register double a00 = A[ta]; register double a10 = A[tta]; register double b00 = B[tb]; register double b01 = B[tb+1]; 

               c00 += a00*b00 ; c01 += a00*b01 ; c10 += a10*b00 ; c11 += a10*b01 ;

               a00 = A[ta+1]; a10 = A[tta+1]; b00 = B[ttb]; b01 = B[ttb+1];

               c00 += a00*b00 ; c01 += a00*b01 ; c10 += a10*b00 ; c11 += a10*b01 ;
          }
          C[t1]=c00;
          C[t1+1]=c01;
          C[t2]=c10;
          C[t2+1]=c11;
     }
}
      
}