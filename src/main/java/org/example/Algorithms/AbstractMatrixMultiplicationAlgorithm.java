package org.example.Algorithms;

import org.reflections.Reflections;
import org.reflections.scanners.Scanners;
import org.reflections.util.ClasspathHelper;
import org.reflections.util.ConfigurationBuilder;

import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Interface for matrix multiplication algorithms
 */
public interface AbstractMatrixMultiplicationAlgorithm {


    /**
     * Multiply matrices A and B
     */
    double[][] multiply(double[][] a, double[][] b);

    /**
     * Get the name of the algorithm
     */
    String getName();

    /**
     * Check if matrices are valid for multiplication (columns of A = rows of B)
     */
    default boolean checkIfValidForMultiplication(double[][] a, double[][] b) {
        return a[0].length == b.length;
    }

    /**
     * Default test method to verify the algorithm's correctness
     */
    default void test(double[][] a, double[][] b) {
        System.out.println("\n===== Testing " + getName() + " =====");

        // Create a naive implementation for comparison
        SeqNaiveMultiplication naive = new SeqNaiveMultiplication();
        double[][] expected = naive.multiply(a, b);
        double[][] actual = multiply(a, b);

        // Calculate the error percentage
        double maxError = 0.0;
        double sumError = 0.0;
        int rows = expected.length;
        int cols = expected[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double e = expected[i][j];
                double act = actual[i][j];
                double diff = Math.abs(e - act);
                double relError = e != 0.0 ? diff / Math.abs(e) * 100.0 : diff * 100.0;
                maxError = Math.max(maxError, relError);
                sumError += relError;
            }
        }

        double avgError = sumError / (rows * cols);

        System.out.printf("Matrix size: %dx%d%n", a.length, a[0].length);
        System.out.printf("Maximum error: %.4f%%%n", maxError);
        System.out.printf("Average error: %.4f%%%n", avgError);

        boolean isCorrect = maxError <= 2.0; // 2% error threshold
        System.out.println("Result: " + (isCorrect ? "CORRECT" : "INCORRECT"));
        if (!isCorrect) {
            throw new RuntimeException("The Result seems to be wrong, Please verify algorithm");
        }

    }

    /**
     * Discover all matrix multiplication algorithms
     */
    static List<AbstractMatrixMultiplicationAlgorithm> discoverAlgorithms() {
        var algorithms = new ArrayList<AbstractMatrixMultiplicationAlgorithm>();

        System.out.println("Discovering matrix multiplication algorithms...");

        try {
            var configBuilder = new ConfigurationBuilder()
                    .setScanners(Scanners.SubTypes)
                    .setUrls(ClasspathHelper.forPackage("org.example.algorithms"));

            var reflections = new Reflections(configBuilder);

            var subTypes =
                    reflections.getSubTypesOf(AbstractMatrixMultiplicationAlgorithm.class);

            System.out.println("Found " + subTypes.size() + " algorithm implementations");

            for (var clazz : subTypes) {
                try {
                    // Skip abstract classes and interfaces
                    if (Modifier.isAbstract(clazz.getModifiers()) || clazz.isInterface()) {
                        System.out.println("Skipping abstract class or interface: " + clazz.getName());
                        continue;
                    }

                    var algorithm = clazz.getDeclaredConstructor().newInstance();
                    algorithms.add(algorithm);
                    System.out.println("Algorithm loaded: " + algorithm.getClass().getSimpleName());
                } catch (Exception e) {
                    System.err.println("Failed to instantiate algorithm: " + clazz.getName() + " - " + e.getMessage());
                }
            }

            // Add fallback algorithm if none were discovered
            if (algorithms.isEmpty()) {
                System.out.println("No algorithms discovered via reflection, adding fallback NaiveMultiplication");
                algorithms.add(new SeqNaiveMultiplication());
            }
            algorithms.sort(Comparator.comparing(AbstractMatrixMultiplicationAlgorithm::getName));

            System.out.println("Total algorithms loaded: " + algorithms.size());
        } catch (Exception e) {
            System.err.println("Failed to discover algorithms: " + e.getMessage());
            if (algorithms.isEmpty()) {
                System.out.println("Adding fallback NaiveMultiplication after reflection failure");
                algorithms.add(new SeqNaiveMultiplication());
            }
        }

        return algorithms;
    }
}


