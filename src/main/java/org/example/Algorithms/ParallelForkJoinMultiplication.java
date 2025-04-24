package org.example.Algorithms;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

/**
 * Parallel matrix multiplication algorithm using Fork/Join framework
 *
 * This algorithm divides the workload among multiple threads to utilize
 * multicore processors for faster computation.
 */
public class ParallelForkJoinMultiplication implements AbstractMatrixMultiplicationAlgorithm {
    private final int threshold;
    private static final ForkJoinPool POOL = new ForkJoinPool();

    public ParallelForkJoinMultiplication() {
        this.threshold = 64;
    }

    public ParallelForkJoinMultiplication(int threshold) {
        this.threshold = threshold;
    }

    @Override
    public double[][] multiply(double[][] a, double[][] b) {
        assert checkIfValidForMultiplication(a, b) : "Invalid matrices for multiplication";

        int m = a.length;
        int n = a[0].length;
        int p = b[0].length;

        double[][] result = new double[m][p];

        POOL.invoke(new MatrixMultiplyTask(a, b, result, 0, m, 0, p));

        return result;
    }

    private class MatrixMultiplyTask extends RecursiveAction {
        private final double[][] a;
        private final double[][] b;
        private final double[][] result;
        private final int rowStart;
        private final int rowEnd;
        private final int colStart;
        private final int colEnd;

        public MatrixMultiplyTask(double[][] a, double[][] b, double[][] result,
                                  int rowStart, int rowEnd, int colStart, int colEnd) {
            this.a = a;
            this.b = b;
            this.result = result;
            this.rowStart = rowStart;
            this.rowEnd = rowEnd;
            this.colStart = colStart;
            this.colEnd = colEnd;
        }

        @Override
        protected void compute() {
            int rowSize = rowEnd - rowStart;
            int colSize = colEnd - colStart;

            // If the problem size is small enough, compute directly
            if (rowSize <= threshold && colSize <= threshold) {
                computeDirectly();
                return;
            }

            // Otherwise, fork subtasks
            invokeAll(createSubtasks());
        }

        private RecursiveAction[] createSubtasks() {
            int rowMiddle = rowStart + (rowEnd - rowStart) / 2;
            int colMiddle = colStart + (colEnd - colStart) / 2;

            return new RecursiveAction[] {
                    new MatrixMultiplyTask(a, b, result, rowStart, rowMiddle, colStart, colMiddle),
                    new MatrixMultiplyTask(a, b, result, rowStart, rowMiddle, colMiddle, colEnd),
                    new MatrixMultiplyTask(a, b, result, rowMiddle, rowEnd, colStart, colMiddle),
                    new MatrixMultiplyTask(a, b, result, rowMiddle, rowEnd, colMiddle, colEnd)
            };
        }

        private void computeDirectly() {
            int n = a[0].length;

            for (int i = rowStart; i < rowEnd; i++) {
                for (int j = colStart; j < colEnd; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < n; k++) {
                        sum += a[i][k] * b[k][j];
                    }
                    result[i][j] = sum;
                }
            }
        }
    }

    @Override
    public String getName() {
        return "Parallel Fork Join Multiplication";
    }
}