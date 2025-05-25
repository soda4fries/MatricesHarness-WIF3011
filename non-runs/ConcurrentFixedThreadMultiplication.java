package org.example.Algorithms;

import java.util.concurrent.atomic.AtomicInteger;

public class ConcurrentFixedThreadMultiplication implements AbstractMatrixMultiplicationAlgorithm {
    private final int threadCount;

    public ConcurrentFixedThreadMultiplication(int threadCount) {
        this.threadCount = threadCount;
    }

    public ConcurrentFixedThreadMultiplication() {
        this(Runtime.getRuntime().availableProcessors()); // Use available cores
    }

    @Override
    public double[][] multiply(double[][] a, double[][] b) {
        assert a[0].length == b.length;

        int m = a.length, n = a[0].length, p = b[0].length;
        double[][] result = new double[m][p];

        Thread[] threads = new Thread[threadCount];
        AtomicInteger rowIndex = new AtomicInteger(0);

        for (int t = 0; t < threadCount; t++) {
            threads[t] = new Thread(() -> {
                int i;
                while ((i = rowIndex.getAndIncrement()) < m) {
                    for (int j = 0; j < p; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < n; k++) {
                            sum += a[i][k] * b[k][j];
                        }
                        result[i][j] = sum;
                    }
                }
            });
            threads[t].start();
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
        return "Concurrent Fixed Thread Multiplication (" + threadCount + " threads)";
    }
}
