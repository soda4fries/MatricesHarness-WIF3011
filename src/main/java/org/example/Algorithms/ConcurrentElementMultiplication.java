package org.example.Algorithms;

import java.util.ArrayList;
import java.util.List;

public class ConcurrentElementMultiplication implements AbstractMatrixMultiplicationAlgorithm {
    @Override
    public double[][] multiply(double[][] a, double[][] b) {
        assert a[0].length == b.length;

        int m = a.length, n = a[0].length, p = b[0].length;
        double[][] result = new double[m][p];

        List<Thread> threads = new ArrayList<>();

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                final int row = i;
                final int col = j;

                Thread t = new Thread(() -> {
                    double sum = 0.0;
                    for (int k = 0; k < n; k++) {
                        sum += a[row][k] * b[k][col];
                    }
                    result[row][col] = sum;
                });

                threads.add(t);
                t.start();
            }
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
        return "Concurrent Element Multiplication";
    }
}
