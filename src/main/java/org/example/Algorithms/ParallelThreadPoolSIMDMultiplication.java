package org.example.Algorithms;

import jdk.incubator.vector.*;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Matrix multiplication using SIMD instructions with lightweight parallelization
 * and transposed columns for better cache locality.
 * if u can tile this prolly will win, kinda getting array bounds exception when dividing by row/thread no kek
 */
public class ParallelThreadPoolSIMDMultiplication implements AbstractMatrixMultiplicationAlgorithm {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    @Override
    public String getName() {
        return "Parallel Thread Pool SIMD Multiplication";
    }

    @Override
    public double[][] multiply(double[][] a, double[][] b) {
        assert checkIfValidForMultiplication(a, b) : "Invalid matrices for multiplication";


        int aRows = a.length;
        int aCols = a[0].length;
        int bCols = b[0].length;

        // Transpose matrix B for better cache locality
        double[][] bTransposed = new double[bCols][b.length];
        for (int i = 0; i < b.length; i++) {
            for (int j = 0; j < bCols; j++) {
                bTransposed[j][i] = b[i][j];
            }
        }

        // Create result matrix
        double[][] result = new double[aRows][bCols];

        // Use a simple thread pool with one thread per available processor
        int threadCount = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);

        // Split the work by rows for simple and efficient parallelization
        int rowsPerThread = Math.max(1, aRows / threadCount);

        for (int t = 0; t < threadCount; t++) {
            final int startRow = t * rowsPerThread;
            final int endRow = (t == threadCount - 1) ? aRows : (t + 1) * rowsPerThread;

            // Skip empty ranges
            if (startRow >= aRows) continue;

            executor.execute(() -> processRowRange(a, bTransposed, result, startRow, endRow, aCols, bCols));
        }

        // Wait for all tasks to complete
        executor.shutdown();
        try {
            executor.awaitTermination(1, TimeUnit.HOURS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        return result;
    }

    /**
     * Process a range of rows using SIMD operations
     */
    private void processRowRange(double[][] a, double[][] bTransposed, double[][] result,
                                 int startRow, int endRow, int aCols, int bCols) {
        // Get vector size for the current platform
        int vectorSize = SPECIES.length();

        for (int i = startRow; i < endRow; i++) {
            for (int j = 0; j < bCols; j++) {
                double sum = 0.0;
                int k = 0;

                // Process elements in SIMD-sized chunks
                for (; k <= aCols - vectorSize; k += vectorSize) {
                    DoubleVector av = DoubleVector.fromArray(SPECIES, a[i], k);
                    DoubleVector bv = DoubleVector.fromArray(SPECIES, bTransposed[j], k);

                    // Multiply corresponding elements and accumulate
                    sum += av.mul(bv).reduceLanes(VectorOperators.ADD);
                }

                // Process remaining elements (less than a full vector)
                for (; k < aCols; k++) {
                    sum += a[i][k] * bTransposed[j][k];
                }

                result[i][j] = sum;
            }
        }
    }
}