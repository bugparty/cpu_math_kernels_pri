void dgemm3(double *C,double *A,double *B,int n)
{
int i,j,k;
const int STRIDE = 3;
for(int i=0;i<n;i+=STRIDE){
     for(int j=0;j<n;j+=STRIDE){
          register double c00,c01,c02, c10,c11,c12,c20,c21,c22;
          register int t,tt,ttt;
          double a00, a10,a20;
          double b00, b01, b02;
          t = i*n+j;tt= t+n;ttt=tt+n;
          c00=C[t];c01=C[t+1];c02=C[t+2];
          c10=C[tt];c11=C[tt+1];c12=C[tt+2];
          c20=C[ttt];c21=C[ttt+1];c22=C[ttt+2];
          for(int k=0;k<n;k+=STRIDE){
               register int ta = i*n+k; register int tta = ta+n;register int ttta = tta+n;
               register int tb = k*n+j; register int ttb = tb+n;register int tttb = ttb+n;
               // Load elements of A into temporary variables
               a00 = A[ta]; a10 = A[tta]; a20 = A[ttta];
               // Load elements of B into temporary variables
               b00 = B[tb]; b01 = B[tb+1]; b02 = B[tb+2];
                
               
               // Perform matrix multiplication
               //below uses register a00 a10 a20  b00 b01 b02  
               c00 += a00 * b00;c01 += a00 * b01;c02 += a00 * b02;c10 += a10 * b00;c20 += a20 * b00 ;c21 += a20 * b01;c12 += a10 * b02 ;
               c22 += a20 * b02;c11 += a10 * b01;
               //after this line a00,a10,a20,b00,b01,b02 will not be used anymore
               // we need to load a01 a02 a11 a12 a21 a22 b10 b11 b12 b20 b21 b22 12 register, but 7 available
               //if we select a01,a11,a21 we need b10 b11 b12
               // so we load a00=A01, a10=A11, a20=A21, b00=B10 b01=B11 b02=B12
               a00 = A[ta+1];a10 = A[tta+1];a20 = A[ttta+1]; b00 = B[ttb];b01 = B[ttb+1]; b02 = B[ttb+2];
               c00 += a00 * b00;c01 += a00 * b01 ;c02 += a00 * b02 ;
               c10 += a10 * b00 ;c11 += a10 * b01;c12 += a10 * b02;
               c20 += a20 * b00;c21 += a20 * b01; c22 += a20 * b02;
               //after this line, a00,a10,a20, b00,b01,b02, b10,b11,b12 is not being used anymore
               // we still need a02 a12 a22 b20 b21 b22
               // so we load a00=A02, a10=A12, a20=A22, b00=B20 b01=B21 b02=B22
               a00 = A[ta+2];a10 = A[tta+2];a20 = A[ttta+2];
               b00 = B[tttb]; b01 = B[tttb+1]; b02 = B[tttb+2];
               c00 += a00 * b00;
               c01 += a00 * b01;
               c02 += a00 * b02;
               
               c10 += a10 * b00;
               c11 += a10 * b01;
               c12 += a10 * b02;
               
               c20 += a20 * b00;
               c21 += a20 * b01;
               c22 += a20 * b02;
          }
          C[t]=c00;C[t+1]=c01;C[t+2]=c02;
          C[tt]=c10;C[tt+1]=c11;C[tt+2]=c12;
          C[ttt]=c20;C[ttt+1]=c21;C[ttt+2]=c22;
     }
}
      
}