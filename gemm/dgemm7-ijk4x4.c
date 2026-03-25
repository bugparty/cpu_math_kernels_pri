void dgemm7(double *C,double *A,double *B,int n)
{
    int i, j, k, ii, jj, kk;
    const int STRIDE = 4; 

    for (i = 0; i < n; i += BLOCK_SIZE) 
        for (j = 0; j < n; j += BLOCK_SIZE) 
            for (k = 0; k < n; k += BLOCK_SIZE)
                for (ii = i; ii < (i + BLOCK_SIZE); ii += STRIDE)
                    for (jj = j; jj < (j + BLOCK_SIZE); jj += STRIDE)
                    {
                        register int t = ii*n + jj; 
                        register int tt = t + n; 
                        register int ttt = tt + n;
                        register int tttt = ttt + n;
                        register double c00 = C[t], c01 = C[t + 1], c02 = C[t + 2], c03 = C[t + 3];
                        register double c10 = C[tt], c11 = C[tt + 1], c12 = C[tt + 2], c13 = C[tt + 3];
                        register double c20 = C[ttt], c21 = C[ttt + 1], c22 = C[ttt + 2], c23 = C[ttt + 3];
                        register double c30 = C[tttt], c31 = C[tttt + 1], c32 = C[tttt + 2], c33 = C[tttt + 3];

                        for(kk = k; kk < (k + BLOCK_SIZE); kk += STRIDE) 
                        {
                            register int ta = ii*n + kk;
                            register int tta = ta + n;
                            register int ttta = tta + n;
                            register int tttta = ttta + n;
                            register double a00 = A[ta]; //, a01 = A[ta + 1], a02 = A[ta + 2], a03 = A[ta + 3];
                            register double a10 = A[tta]; //, a11 = A[tta + 1], a12 = A[tta + 2], a13 = A[tta + 3];
                            register double a20 = A[ttta]; //, a21 = A[ttta + 1], a22 = A[ttta + 2], a23 = A[ttta + 3];
                            register double a30 = A[tttta]; //, a31 = A[tttta + 1], a32 = A[tttta + 2], a33 = A[tttta + 3];

                            register int tb = kk*n + jj;
                            register double b00 = B[tb], b01 = B[tb + 1], b02 = B[tb + 2], b03 = B[tb + 3];


                            c00 += a00 * b00; c10 += a10 * b00; c20 += a20 * b00; c30 += a30 * b00;
                            c01 += a00 * b01; c11 += a10 * b01; c21 += a20 * b01; c31 += a30 * b01;
                            c02 += a00 * b02; c12 += a10 * b02; c22 += a20 * b02; c32 += a30 * b02;
                            c03 += a00 * b03; c13 += a10 * b03; c23 += a20 * b03; c33 += a30 * b03;

                            tb += n;
                            a00 = A[ta + 1], a10 = A[tta + 1], a20 = A[ttta + 1], a30 = A[tttta + 1];
                            b00 = B[tb], b01 = B[tb + 1], b02 = B[tb + 2], b03 = B[tb + 3];

                            c00 += a00 * b00; c10 += a10 * b00; c20 += a20 * b00; c30 += a30 * b00;
                            c01 += a00 * b01; c11 += a10 * b01; c21 += a20 * b01; c31 += a30 * b01;
                            c02 += a00 * b02; c12 += a10 * b02; c22 += a20 * b02; c32 += a30 * b02;
                            c03 += a00 * b03; c13 += a10 * b03; c23 += a20 * b03; c33 += a30 * b03;

                            tb += n;
                            a00 = A[ta + 2], a10 = A[tta + 2], a20 = A[ttta + 2], a30 = A[tttta + 2];
                            b00 = B[tb], b01 = B[tb + 1], b02 = B[tb + 2], b03 = B[tb + 3];

                            c00 += a00 * b00; c10 += a10 * b00; c20 += a20 * b00; c30 += a30 * b00;
                            c01 += a00 * b01; c11 += a10 * b01; c21 += a20 * b01; c31 += a30 * b01;
                            c02 += a00 * b02; c12 += a10 * b02; c22 += a20 * b02; c32 += a30 * b02;
                            c03 += a00 * b03; c13 += a10 * b03; c23 += a20 * b03; c33 += a30 * b03;

                            tb += n;
                            a00 = A[ta + 3], a10 = A[tta + 3], a20 = A[ttta + 3], a30 = A[tttta + 3];
                            b00 = B[tb], b01 = B[tb + 1], b02 = B[tb + 2], b03 = B[tb + 3];

                            c00 += a00 * b00; c10 += a10 * b00; c20 += a20 * b00; c30 += a30 * b00;
                            c01 += a00 * b01; c11 += a10 * b01; c21 += a20 * b01; c31 += a30 * b01;
                            c02 += a00 * b02; c12 += a10 * b02; c22 += a20 * b02; c32 += a30 * b02;
                            c03 += a00 * b03; c13 += a10 * b03; c23 += a20 * b03; c33 += a30 * b03;
                        }

                        C[t] = c00; C[t + 1] = c01; C[t + 2] = c02; C[t + 3] = c03;
                        C[tt] = c10; C[tt + 1] = c11; C[tt + 2] = c12; C[tt + 3] = c13;
                        C[ttt] = c20; C[ttt + 1] = c21; C[ttt + 2] = c22; C[ttt + 3] = c23;
                        C[tttt] = c30; C[tttt + 1] = c31; C[tttt + 2] = c32; C[tttt + 3] = c33;
                    }
}