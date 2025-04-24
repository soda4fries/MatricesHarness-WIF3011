package org.example.Algorithms;

import jdk.incubator.vector.*;

/**
 * Matrix multiplication using SIMD instructions with transposed columns
 */
public class SeqSIMDTransposedMultiplication implements AbstractMatrixMultiplicationAlgorithm {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    @Override
    public String getName() {
        return "Seq SIMD Transposed Multiplication";
    }

    @Override
    public double[][] multiply(double[][] a, double[][] b) {
        if (!checkIfValidForMultiplication(a, b)) {
            throw new IllegalArgumentException("Invalid matrices for multiplication");
        }

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

        for (int i = 0; i < aRows; i++) {
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

                // Process remaining elements
                for (; k < aCols; k++) {
                    sum += a[i][k] * bTransposed[j][k];
                }

                result[i][j] = sum;
            }
        }

        return result;
    }
}