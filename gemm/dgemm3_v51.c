void dgemm3(double *C,double *A,double *B,int n)
{
int i,j,k;
const int STRIDE = 3;
for(int i=0;i<n;i+=STRIDE){
     for(int j=0;j<n;j+=STRIDE){
          register double c00,c01,c02, c10,c11,c12,c20,c21,c22;
          register int t,tt,ttt;
          double a00, a01, a02, a10, a11, a12, a20, a21, a22;
          double b00, b01, b02, b10, b11, b12, b20, b21, b22;
          t = i*n+j;tt= t+n;ttt=tt+n;
          c00=C[t];c01=C[t+1];c02=C[t+2];
          c10=C[tt];c11=C[tt+1];c12=C[tt+2];
          c20=C[tt];c21=C[ttt+1];c22=C[ttt+2];
          for(int k=0;k<n;k+=STRIDE){
               register int ta = i*n+k; register int tta = ta+n;register int ttta = tta+n;
               register int tb = k*n+j; register int ttb = tb+n;register int tttb = ttb+n;
               // Load elements of A into temporary variables
               a00 = A[ta]; a01 = A[ta+1]; a02 = A[ta+2];
               a10 = A[tta]; a11 = A[tta+1]; a12 = A[tta+2];
               a20 = A[ttta]; a21 = A[ttta+1]; a22 = A[ttta+2];
               
               // Load elements of B into temporary variables
               b00 = B[tb]; b01 = B[tb+1]; b02 = B[tb+2];
               b10 = B[ttb]; b11 = B[ttb+1]; b12 = B[ttb+2];
               b20 = B[tttb]; b21 = B[tttb+1]; b22 = B[tttb+2];
               // Perform matrix multiplication
               c00 += a00 * b00 + a01 * b10 + a02 * b20;
               c01 += a00 * b01 + a01 * b11 + a02 * b21;
               c02 += a00 * b02 + a01 * b12 + a02 * b22;
               
               c10 += a10 * b00 + a11 * b10 + a12 * b20;
               c11 += a10 * b01 + a11 * b11 + a12 * b21;
               c12 += a10 * b02 + a11 * b12 + a12 * b22;
               
               c20 += a20 * b00 + a21 * b10 + a22 * b20;
               c21 += a20 * b01 + a21 * b11 + a22 * b21;
               c22 += a20 * b02 + a21 * b12 + a22 * b22;
          }
          C[t]=c00;C[t+1]=c01;C[t+2]=c02;
          C[tt]=c10;C[tt+1]=c11;C[tt+2]=c12;
          C[ttt]=c20;C[ttt+1]=c21;C[ttt+2]=c22;
     }
}
      
}