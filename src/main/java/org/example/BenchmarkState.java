package org.example;

import org.example.Algorithms.AbstractMatrixMultiplicationAlgorithm;
import org.openjdk.jmh.annotations.*;

/**
 * Base state for matrix multiplication benchmarks
 */
@State(Scope.Benchmark)
public class BenchmarkState {
    @Param({"10", "16", "33", "64", "128", "256", "512"})
    public int size;

    public double[][] matrixA;
    public double[][] matrixB;
    private AbstractMatrixMultiplicationAlgorithm algorithm;

    @Setup(Level.Trial)
    public void setupMatrices() {
        // Create random matrices for benchmarking
        System.out.println("Setting up matrices of size " + size + "x" + size);

        matrixA = MatrixUtils.random(size, size);
        matrixB = MatrixUtils.random(size, size);
    }

    @TearDown(Level.Trial)
    public void cleanupMatrices() {
        System.out.println("Cleaning up matrices of size " + size + "x" + size);
        matrixA = null;
        matrixB = null;
        System.gc();
    }

    public void setAlgorithm(AbstractMatrixMultiplicationAlgorithm algorithm) {
        this.algorithm = algorithm;
    }

    public AbstractMatrixMultiplicationAlgorithm getAlgorithm() {
        return algorithm;
    }
}