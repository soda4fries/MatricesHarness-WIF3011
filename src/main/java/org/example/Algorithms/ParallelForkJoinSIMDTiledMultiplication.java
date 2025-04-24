package org.example.Algorithms;

import jdk.incubator.vector.*;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

/**
 * Optimized matrix multiplication using:
 * 1. Multithreading with Fork/Join framework
 * 2. SIMD vectorization with the Vector API
 * 3. Cache optimization through matrix transposition
 * 4. Tiling/blocking for better cache utilization
 * 5. Loop unrolling for inner loops
 */
public class ParallelForkJoinSIMDTiledMultiplication implements AbstractMatrixMultiplicationAlgorithm {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int VECTOR_SIZE = SPECIES.length();
    private static final int TILE_SIZE = 64;
    private static final ForkJoinPool FORK_JOIN_POOL = new ForkJoinPool();
    private static final int PARALLEL_THRESHOLD = 128;

    @Override
    public String getName() {
        return "Parallel Fork Join SIMD Tiled Multiplication";
    }

    @Override
    public double[][] multiply(double[][] a, double[][] b) {
        if (!checkIfValidForMultiplication(a, b)) {
            throw new IllegalArgumentException("Invalid matrices for multiplication");
        }

        int aRows = a.length;
        int aCols = a[0].length;
        int bCols = b[0].length;


        double[][] bTransposed = transpose(b, bCols, b.length);

        double[][] result = new double[aRows][bCols];

        if (aRows <= PARALLEL_THRESHOLD || bCols <= PARALLEL_THRESHOLD) {
            multiplySequentialBlocked(a, bTransposed, result, aRows, aCols, bCols);
        } else {
            FORK_JOIN_POOL.invoke(new MatrixMultiplyTask(a, bTransposed, result, 0, aRows, 0, bCols, aCols));
        }

        return result;
    }

    /**
     * Optimized sequential blocked multiplication
     */
    private void multiplySequentialBlocked(double[][] a, double[][] bTransposed, double[][] result,
                                           int aRows, int aCols, int bCols) {
        // Tile the computation for better cache locality
        for (int i = 0; i < aRows; i += TILE_SIZE) {
            int iLimit = Math.min(i + TILE_SIZE, aRows);

            for (int j = 0; j < bCols; j += TILE_SIZE) {
                int jLimit = Math.min(j + TILE_SIZE, bCols);

                for (int k = 0; k < aCols; k += TILE_SIZE) {
                    int kLimit = Math.min(k + TILE_SIZE, aCols);

                    // Process the tile
                    for (int ii = i; ii < iLimit; ii++) {
                        for (int jj = j; jj < jLimit; jj++) {
                            computeTileElement(a, bTransposed, result, ii, jj, k, kLimit);
                        }
                    }
                }
            }
        }
    }

    /**
     * Compute a single element in the result matrix using SIMD
     */
    private void computeTileElement(double[][] a, double[][] bTransposed, double[][] result,
                                    int i, int j, int kStart, int kLimit) {
        double sum = 0.0;
        int k = kStart;

        for (; k <= kLimit - VECTOR_SIZE; k += VECTOR_SIZE) {
            DoubleVector av = DoubleVector.fromArray(SPECIES, a[i], k);
            DoubleVector bv = DoubleVector.fromArray(SPECIES, bTransposed[j], k);
            sum += av.mul(bv).reduceLanes(VectorOperators.ADD);
        }

        // Process remaining elements
        for (; k < kLimit; k++) {
            sum += a[i][k] * bTransposed[j][k];
        }

        // Add to any existing result (important for the tiled algorithm)
        result[i][j] += sum;
    }

    /**
     * Optimized matrix transposition
     */
    private double[][] transpose(double[][] matrix, int rows, int cols) {
        double[][] transposed = new double[rows][cols];

        // Use tiling for transpose to improve cache behavior
        for (int i = 0; i < cols; i += TILE_SIZE) {
            int iLimit = Math.min(i + TILE_SIZE, cols);

            for (int j = 0; j < rows; j += TILE_SIZE) {
                int jLimit = Math.min(j + TILE_SIZE, rows);

                // Process the tile
                for (int ii = i; ii < iLimit; ii++) {
                    for (int jj = j; jj < jLimit; jj++) {
                        transposed[jj][ii] = matrix[ii][jj];
                    }
                }
            }
        }

        return transposed;
    }

    /**
     * Fork-Join recursive task for parallel matrix multiplication
     */
    private class MatrixMultiplyTask extends RecursiveAction {
        private final double[][] a;
        private final double[][] bTransposed;
        private final double[][] result;
        private final int rowStart, rowEnd, colStart, colEnd, aCols;

        // Size threshold for splitting the task
        private static final int TASK_THRESHOLD = TILE_SIZE * 2;

        MatrixMultiplyTask(double[][] a, double[][] bTransposed, double[][] result,
                           int rowStart, int rowEnd, int colStart, int colEnd, int aCols) {
            this.a = a;
            this.bTransposed = bTransposed;
            this.result = result;
            this.rowStart = rowStart;
            this.rowEnd = rowEnd;
            this.colStart = colStart;
            this.colEnd = colEnd;
            this.aCols = aCols;
        }

        @Override
        protected void compute() {
            int rowSize = rowEnd - rowStart;
            int colSize = colEnd - colStart;

            // If the task is small enough, compute it directly
            if (rowSize <= TASK_THRESHOLD && colSize <= TASK_THRESHOLD) {
                computeDirectly();
                return;
            }

            // Otherwise, split the task
            invokeAll(createSubtasks());
        }

        /**
         * Compute the partial matrix multiplication directly using blocked algorithm
         */
        private void computeDirectly() {
            for (int i = rowStart; i < rowEnd; i += TILE_SIZE) {
                int iLimit = Math.min(i + TILE_SIZE, rowEnd);

                for (int j = colStart; j < colEnd; j += TILE_SIZE) {
                    int jLimit = Math.min(j + TILE_SIZE, colEnd);

                    for (int k = 0; k < aCols; k += TILE_SIZE) {
                        int kLimit = Math.min(k + TILE_SIZE, aCols);

                        // Process the tile
                        for (int ii = i; ii < iLimit; ii++) {
                            for (int jj = j; jj < jLimit; jj++) {
                                computeTileElement(a, bTransposed, result, ii, jj, k, kLimit);
                            }
                        }
                    }
                }
            }
        }

        /**
         * Create subtasks by splitting the current task
         */
        private RecursiveAction[] createSubtasks() {
            int rowSize = rowEnd - rowStart;
            int colSize = colEnd - colStart;

            RecursiveAction[] tasks;

            if (rowSize >= colSize) {
                int midRow = rowStart + rowSize / 2;
                tasks = new RecursiveAction[2];
                tasks[0] = new MatrixMultiplyTask(a, bTransposed, result, rowStart, midRow, colStart, colEnd, aCols);
                tasks[1] = new MatrixMultiplyTask(a, bTransposed, result, midRow, rowEnd, colStart, colEnd, aCols);
            } else {
                int midCol = colStart + colSize / 2;
                tasks = new RecursiveAction[2];
                tasks[0] = new MatrixMultiplyTask(a, bTransposed, result, rowStart, rowEnd, colStart, midCol, aCols);
                tasks[1] = new MatrixMultiplyTask(a, bTransposed, result, rowStart, rowEnd, midCol, colEnd, aCols);
            }

            return tasks;
        }
    }
}