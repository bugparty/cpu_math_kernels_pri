//
// Created by bhan001 on 10/25/24.
//
void mydtrsv(char UPLO,double *A,double *B,int n,int *ipiv)
{
    double * y = (double*)malloc(n*sizeof(double));
    double * orig = y;
    double * x = 0;
    int i;
    switch(UPLO){
        case 'L':
            /*
            y(1) = b(pvt(1));
            for i = 2 : n,
                y(i) = b(pvt(i)) - sum( y(1:i-1) .* A(i,1:i-1) );
            end
             */
            y[0] = B[ipiv[0]];
            for(i=1;i<n;++i){
                y[i] = B[ipiv[i]]- myddot(i,y, &A[i*n]);
            }
            for(i=0;i<n;i++){
                B[i] = y[i];
            }
            break;
        case 'U':
            x = y;
            y = B;
            x[n-1] = y[n-1] /A[(n-1)*n+n-1];
            for(i=n-2;i>=0;--i){
                // x(n) = y(n) / A(n,n);
                //for i = n-1 : -1 : 1,
                //    x(i) = ( y(i) - sum( x(i+1:n) .* A(i, i+1:n) ) ) / A(i,i);
                //end
                //matlab: if n=3, i= 2 1, i+1:n= 3:3 2:3
                // in c if n=3 i= 1 0 len = 1 2
                //
                int len = n-1-i;
                x[i] = (y[i] - myddot(len,&x[i+1], &A[i*n+i+1]))/A[i*n+i];
            }
            for(i=0;i<n;i++){
                B[i] = x[i];
            }
            break;
        default:
            break;
    }
    free(orig);
}