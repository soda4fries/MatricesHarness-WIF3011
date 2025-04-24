package org.example.Algorithms;

import jdk.incubator.vector.*;

import java.util.Arrays;

public class ParallelArraySetSIMDTransposeMultiplication implements AbstractMatrixMultiplicationAlgorithm {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

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

        // Use Arrays.parallelSetAll to parallelize the filling of the result matrix
        Arrays.parallelSetAll(result, i -> {
            double[] aRow = a[i];
            double[] resultRow = new double[p];

            for (int j = 0; j < p; j++) {
                double sum = 0.0;
                double[] bTransposedRow = bTransposed[j];

                // SIMD multiplication using vectorized operations
                int k = 0;
                int vectorLength = SPECIES.length();

                // Use SIMD for chunks of k until we run out of space for full vectors
                for (; k <= n - vectorLength; k += vectorLength) {
                    DoubleVector av = DoubleVector.fromArray(SPECIES, aRow, k);
                    DoubleVector bv = DoubleVector.fromArray(SPECIES, bTransposedRow, k);

                    // Perform vectorized multiplication and accumulate the sum
                    sum += av.mul(bv).reduceLanes(VectorOperators.ADD);
                }

                // Process the remaining elements that don't fit into a full vector
                for (; k < n; k++) {
                    sum += aRow[k] * bTransposedRow[k];
                }

                resultRow[j] = sum;
            }

            return resultRow;
        });

        return result;
    }

    @Override
    public String getName() {
        return "Parallel SIMD Transpose Multiplication";
    }
}
