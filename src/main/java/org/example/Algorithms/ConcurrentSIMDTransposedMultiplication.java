package org.example.Algorithms;

import jdk.incubator.vector.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Matrix multiplication using SIMD instructions with transposed columns and multiple threads
 */
public class ConcurrentSIMDTransposedMultiplication implements AbstractMatrixMultiplicationAlgorithm {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private final int threadCount;
    
    public ConcurrentSIMDTransposedMultiplication(int threadCount) {
        this.threadCount = threadCount;
    }
    
    public ConcurrentSIMDTransposedMultiplication() {
        this(Runtime.getRuntime().availableProcessors()); // Use available cores
    }
    
    @Override
    public String getName() {
        return "Concurrent SIMD Transposed Multiplication (" + threadCount + " threads)";
    }
    
    @Override
    public double[][] multiply(double[][] a, double[][] b) {
        assert a[0].length == b.length;
        
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
        
        // Get vector length for the current platform
        int vectorSize = SPECIES.length();
        
        // Create and start worker threads
        Thread[] threads = new Thread[threadCount];
        AtomicInteger rowIndex = new AtomicInteger(0);
        
        for (int t = 0; t < threadCount; t++) {
            threads[t] = new Thread(() -> {
                int i;
                while ((i = rowIndex.getAndIncrement()) < aRows) {
                    processRow(a, bTransposed, result, i, aRows, aCols, bCols, vectorSize);
                }
            });
            threads[t].start();
        }
        
        // Wait for all threads to complete
        for (Thread t : threads) {
            try {
                t.join();
            } catch (InterruptedException ignored) {}
        }
        
        return result;
    }
    
    private void processRow(double[][] a, double[][] bTransposed, double[][] result, 
                           int rowIndex, int aRows, int aCols, int bCols, int vectorSize) {
        for (int j = 0; j < bCols; j++) {
            double sum = 0.0;
            int k = 0;
            
            // Process elements in SIMD-sized chunks
            for (; k <= aCols - vectorSize; k += vectorSize) {
                DoubleVector av = DoubleVector.fromArray(SPECIES, a[rowIndex], k);
                DoubleVector bv = DoubleVector.fromArray(SPECIES, bTransposed[j], k);
                
                // Multiply corresponding elements and accumulate
                sum += av.mul(bv).reduceLanes(VectorOperators.ADD);
            }
            
            // Process remaining elements
            for (; k < aCols; k++) {
                sum += a[rowIndex][k] * bTransposed[j][k];
            }
            
            result[rowIndex][j] = sum;
        }
    }
}