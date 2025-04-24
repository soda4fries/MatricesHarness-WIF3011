package org.example.Algorithms;

/**
 * Implementation of naive matrix multiplication
 */
public class SeqNaiveMultiplication implements AbstractMatrixMultiplicationAlgorithm {
    @Override
    public double[][] multiply(double[][] a, double[][] b) {
        assert checkIfValidForMultiplication(a, b) : "Invalid matrices for multiplication";

        int m = a.length;
        int n = a[0].length;
        int p = b[0].length;
        double[][] result = new double[m][p];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }

        return result;
    }

    @Override
    public String getName() {
        return "Seq Naive Multiplication";
    }
}