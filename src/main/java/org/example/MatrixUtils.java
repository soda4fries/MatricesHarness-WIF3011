package org.example;

import java.util.Random; /**
 * Utility class for matrix operations
 */
public class MatrixUtils {
    /**
     * Generate a random matrix with the given dimensions
     */
    public static double[][] random(int rows, int cols) {
        double[][] matrix = new double[rows][cols];
        Random rand = new Random();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = rand.nextDouble();
            }
        }
        return matrix;
    }

    /**
     * Pretty print a matrix
     */
    public static String toString(double[][] matrix) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < matrix.length; i++) {
            sb.append("[");
            for (int j = 0; j < matrix[0].length; j++) {
                sb.append(String.format("%.4f", matrix[i][j]));
                if (j < matrix[0].length - 1) {
                    sb.append(", ");
                }
            }
            sb.append("]\n");
        }
        return sb.toString();
    }

    /**
     * Return the number of rows in a matrix
     */
    public static int getRows(double[][] matrix) {
        return matrix.length;
    }

    /**
     * Return the number of columns in a matrix
     */
    public static int getCols(double[][] matrix) {
        return matrix[0].length;
    }
}
