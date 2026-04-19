#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// Include the code to be tested directly
#include "my.c"

// Helper function to initialize a matrix
void init_matrix(double *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = i * 10 + j; // Row i, Col j is initialized to i*10 + j
        }
    }
}

// Helper function to check if a row matches expected values
void check_row(double *A, int n, int row_idx, double *expected_row) {
    for (int j = 0; j < n; j++) {
        assert(fabs(A[row_idx * n + j] - expected_row[j]) < 1e-9);
    }
}

void test_swapRow_basic() {
    int n = 4;
    double A[16];
    init_matrix(A, n);

    // Expected rows before swap
    // row 0: 0, 1, 2, 3
    // row 1: 10, 11, 12, 13
    // row 2: 20, 21, 22, 23
    // row 3: 30, 31, 32, 33

    double expected_row_1[] = {10, 11, 12, 13};
    double expected_row_2[] = {20, 21, 22, 23};

    // Swap row 1 and row 2
    swapRow(A, n, 1, 2);

    // Verify row 1 is now row 2
    check_row(A, n, 1, expected_row_2);

    // Verify row 2 is now row 1
    check_row(A, n, 2, expected_row_1);

    // Verify other rows are untouched
    double expected_row_0[] = {0, 1, 2, 3};
    double expected_row_3[] = {30, 31, 32, 33};
    check_row(A, n, 0, expected_row_0);
    check_row(A, n, 3, expected_row_3);

    printf("test_swapRow_basic passed.\n");
}

void test_swapRow_same_row() {
    int n = 3;
    double A[9];
    init_matrix(A, n);

    double expected_row_1[] = {10, 11, 12};

    // Swap row 1 with itself
    swapRow(A, n, 1, 1);

    // Verify row 1 is unchanged
    check_row(A, n, 1, expected_row_1);

    printf("test_swapRow_same_row passed.\n");
}

int main() {
    printf("Running swapRow tests...\n");
    test_swapRow_basic();
    test_swapRow_same_row();
    printf("All swapRow tests passed!\n");
    return 0;
}
