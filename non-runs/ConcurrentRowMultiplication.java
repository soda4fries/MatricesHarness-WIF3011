package org.example.Algorithms;

public class ConcurrentRowMultiplication implements AbstractMatrixMultiplicationAlgorithm {
    @Override
    public double[][] multiply(double[][] a, double[][] b) {
        assert a[0].length == b.length;

        int m = a.length, n = a[0].length, p = b[0].length;
        double[][] result = new double[m][p];

        Thread[] threads = new Thread[m];

        for (int i = 0; i < m; i++) {
            final int row = i;
            threads[i] = new Thread(() -> {
                for (int j = 0; j < p; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < n; k++) {
                        sum += a[row][k] * b[k][j];
                    }
                    result[row][j] = sum;
                }
            });
            threads[i].start();
        }

        for (Thread t : threads) {
            try {
                t.join();
            } catch (InterruptedException ignored) {}
        }

        return result;
    }

    @Override
    public String getName() {
        return "Concurrent Row Multiplication";
    }
}
