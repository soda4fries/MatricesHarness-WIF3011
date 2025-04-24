package org.example.Algorithms;

/**
 * Transpose-based matrix multiplication algorithm
 *
 * This algorithm transposes the second matrix to improve memory access patterns
 * by making both inner loops access consecutive memory locations.
 */
public class SeqTransposeMultiplication implements AbstractMatrixMultiplicationAlgorithm {

    @Override
    public double[][] multiply(double[][] a, double[][] b) {
        assert checkIfValidForMultiplication(a, b) : "Invalid matrices for multiplication";

        int m = a.length;
        int n = a[0].length;
        int p = b[0].length;

        // Transpose matrix B
        double[][] bTransposed = new double[p][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                bTransposed[j][i] = b[i][j];
            }
        }

        double[][] result = new double[m][p];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                double sum = 0.0;
                double[] aRow = a[i];
                double[] bTransposedRow = bTransposed[j];
                for (int k = 0; k < n; k++) {
                    sum += aRow[k] * bTransposedRow[k];
                }

                result[i][j] = sum;
            }
        }

        return result;
    }

    @Override
    public String getName() {
        return "Seq Transpose Multiplication";
    }
}