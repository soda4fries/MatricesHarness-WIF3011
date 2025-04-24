package org.example;

import org.example.Algorithms.AbstractMatrixMultiplicationAlgorithm;
import org.openjdk.jmh.annotations.*;

import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.results.format.ResultFormatType;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;

import java.util.concurrent.TimeUnit;

public class MatrixMultiplicationBenchmark {
    @State(Scope.Benchmark)
    public static class AlgorithmState extends BenchmarkState {
        @Param({"NaiveMultiplication"})
        public String algorithmName;

        @Setup(Level.Trial)
        public void setupAlgorithm() {
            AbstractMatrixMultiplicationAlgorithm algorithm = findAlgorithmByName(algorithmName);
            setAlgorithm(algorithm);

            int coreCount = Runtime.getRuntime().availableProcessors();
            System.out.println("Available CPU cores: " + coreCount);
            System.out.println("Testing algorithm: " + algorithmName);
            // Test algorithm for correctness if matrix is small enough
            if (size <= 100) {
                try {
                    System.out.println("Testing " + algorithm.getName() + " with matrix size " + size + "x" + size);
                    algorithm.test(matrixA, matrixB);
                } catch (Exception e) {
                    System.err.println("Error testing algorithm " + algorithm.getName() + ": " + e.getMessage());
                }
            }
        }

        private AbstractMatrixMultiplicationAlgorithm findAlgorithmByName(String name) {
            for (AbstractMatrixMultiplicationAlgorithm algorithm : AbstractMatrixMultiplicationAlgorithm.discoverAlgorithms()) {
                if (algorithm.getName().equals(name)) {
                    return algorithm;
                }
            }
            throw new IllegalStateException("Algorithm not found: " + name);
        }
    }
    /**
     * Main benchmark for execution time and throughput
     */
    @Benchmark
    @BenchmarkMode({Mode.AverageTime, Mode.Throughput, Mode.SingleShotTime})
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @Fork(value = 1)
    @Warmup(iterations = 2, time = 3)
    @Measurement(iterations = 3, time = 3)
    public void multiply(AlgorithmState state, Blackhole blackhole) {
        AbstractMatrixMultiplicationAlgorithm algorithm = state.getAlgorithm();
        double[][] result = algorithm.multiply(state.matrixA, state.matrixB);
        blackhole.consume(result);
    }

    /**
     * Main method to run the benchmark
     */
    public static void main(String[] args) throws RunnerException {
        System.out.println("===== MATRIX MULTIPLICATION BENCHMARK =====");

        var algorithms = AbstractMatrixMultiplicationAlgorithm.discoverAlgorithms();
        if (algorithms.isEmpty()) {
            System.err.println("No algorithms found! Cannot run benchmarks.");
            return;
        }
        String[] algorithmNames = algorithms.stream()
                .map(AbstractMatrixMultiplicationAlgorithm::getName)
                .toArray(String[]::new);

        System.out.println("\n===== BENCHMARKING THE FOLLOWING ALGORITHMS =====");
        for (String name : algorithmNames) {
            System.out.println("Algorithm: " + name);
        }
        System.out.println("=================================================\n");

        // Create options with all algorithms and matrix sizes
        var options = new OptionsBuilder()
                .include(MatrixMultiplicationBenchmark.class.getSimpleName())
                // "10", "16", "33", "64", "128", "256", "512", "1024", "2056", "5000",
                .param("size", "3","1000")
                .param("algorithmName", algorithmNames)
                .warmupIterations(1)
                .warmupTime(TimeValue.seconds(1))
                .measurementIterations(2)
                .measurementTime(TimeValue.seconds(1))
                .timeout(TimeValue.minutes(30))
                .forks(1)
                .shouldDoGC(true)
                .resultFormat(ResultFormatType.CSV)
                .result("matrix-multiplication-benchmark-results.csv")
                .jvmArgs(
                        "-Xms8g", "-Xmx12g",                   // Large heap size (adjust based on available RAM)
                        //"-XX:+AlwaysPreTouch",             //Pre-touch memory pages during JVM startup
                        "--add-modules=jdk.incubator.vector",   //vector api
                        "-XX:+UseNUMA",                        // Enable NUMA support
                        "-XX:+UseSuperWord"                   // Enable additional vectorization
                        )
                .build();

        System.out.println("\nRunning matrix multiplication benchmarks with diverse matrix sizes for " +
                algorithmNames.length + " algorithms...");
        new Runner(options).run();

        System.out.println("\nBenchmark complete. Results have been saved to 'matrix-multiplication-benchmark-results.csv'");

    }
}