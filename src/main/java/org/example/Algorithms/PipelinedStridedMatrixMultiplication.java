package org.example.Algorithms;
import org.example.Algorithms.AbstractMatrixMultiplicationAlgorithm;
import org.example.MatrixUtils;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

public class PipelinedStridedMatrixMultiplication implements AbstractMatrixMultiplicationAlgorithm {
    private static final int STRIP_WIDTH = 100; // Columns per strip
    private static final int LOADER_THREADS = 1;
    private static final int COMPUTE_THREADS = Runtime.getRuntime().availableProcessors();
    private static final int WRITER_THREADS = 1;

    @Override
    public double[][] multiply(double[][] matrixA, double[][] matrixB) {
        final long startTime = System.nanoTime();
        final int m = matrixA.length;
        final int n = matrixA[0].length;
        final int p = matrixB[0].length;
        
        double[][] matrixC = new double[m][p];
        
        // Queues for pipeline stages
        BlockingQueue<MatrixStrip> bStripQueue = new LinkedBlockingQueue<>();
        BlockingQueue<MatrixPartialResult> cPartialQueue = new LinkedBlockingQueue<>();
        
        // Thread pools for each stage
        ExecutorService loaderPool = Executors.newFixedThreadPool(LOADER_THREADS);
        ExecutorService computePool = Executors.newFixedThreadPool(COMPUTE_THREADS);
        ExecutorService writerPool = Executors.newFixedThreadPool(WRITER_THREADS);
        
        // Start writer stage first since it's the consumer
        writerPool.submit(new WriterStage(matrixC, cPartialQueue));
        
        // Start compute stage
        for (int i = 0; i < COMPUTE_THREADS; i++) {
            computePool.submit(new ComputeStage(matrixA, bStripQueue, cPartialQueue));
        }
        
        // Start loader stage
        loaderPool.submit(new LoaderStage(matrixB, bStripQueue));
        
        // Shutdown pools and wait for completion
        loaderPool.shutdown();
        try {
            loaderPool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        // 🔥🔥 Insert poison pills for compute threads
        for (int i = 0; i < COMPUTE_THREADS; i++) {
            try {
                bStripQueue.put(MatrixStrip.POISON_PILL);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        
        computePool.shutdown();
        try {
            computePool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        // 🔥🔥 Insert poison pills for writer threads
        for (int i = 0; i < WRITER_THREADS; i++) {
            try {
                cPartialQueue.put(MatrixPartialResult.POISON_PILL);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        writerPool.shutdown();
        try {
            writerPool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        long totalTime = System.nanoTime() - startTime;
        
        System.out.println("Pipeline Stage Timings (ms):");
        System.out.printf("  Loader: %.3f%n", LoaderStage.totalTime / 1_000_000.0);
        System.out.printf("  Compute: %.3f%n", ComputeStage.totalTime / 1_000_000.0);
        System.out.printf("  Writer: %.3f%n", WriterStage.totalTime / 1_000_000.0);
        System.out.printf("Total time: %.3f ms%n", totalTime / 1_000_000.0);
        
        return matrixC;
    }

    @Override
    public String getName() {
        return "Pipelined Strided Matrix Multiplication";
    }

    
    private static class MatrixStrip {
        final double[][] strip;
        final int startCol;
        final int endCol;

        static final MatrixStrip POISON_PILL = new MatrixStrip(null, -1, -1); // 🔥🔥

        MatrixStrip(double[][] strip, int startCol, int endCol) {
            this.strip = strip;
            this.startCol = startCol;
            this.endCol = endCol;
        }
    }
    
    private static class MatrixPartialResult {
        final double[][] partialC;
        final int startCol;
        final int endCol;

        static final MatrixPartialResult POISON_PILL = new MatrixPartialResult(null, -1, -1); // 🔥🔥

        MatrixPartialResult(double[][] partialC, int startCol, int endCol) {
            this.partialC = partialC;
            this.startCol = startCol;
            this.endCol = endCol;
        }
    }
    
    private static class LoaderStage implements Runnable {
        private static volatile long totalTime = 0;
        private final double[][] matrixB;
        private final BlockingQueue<MatrixStrip> outputQueue;
        
        LoaderStage(double[][] matrixB, BlockingQueue<MatrixStrip> outputQueue) {
            this.matrixB = matrixB;
            this.outputQueue = outputQueue;
        }
        
        @Override
        public void run() {
            final long stageStart = System.nanoTime();
            final int n = matrixB.length;
            final int p = matrixB[0].length;
            
            for (int j = 0; j < p; j += STRIP_WIDTH) {
                int endCol = Math.min(j + STRIP_WIDTH, p);
                double[][] strip = new double[n][endCol - j];
                
                for (int col = j; col < endCol; col++) {
                    for (int row = 0; row < n; row++) {
                        strip[row][col - j] = matrixB[row][col];
                    }
                }
                
                try {
                    outputQueue.put(new MatrixStrip(strip, j, endCol));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
            totalTime += System.nanoTime() - stageStart;
        }
    }
    
    private static class ComputeStage implements Runnable {
        private static volatile long totalTime = 0;
        private final double[][] matrixA;
        private final BlockingQueue<MatrixStrip> inputQueue;
        private final BlockingQueue<MatrixPartialResult> outputQueue;
        
        ComputeStage(double[][] matrixA, BlockingQueue<MatrixStrip> inputQueue,
                    BlockingQueue<MatrixPartialResult> outputQueue) {
            this.matrixA = matrixA;
            this.inputQueue = inputQueue;
            this.outputQueue = outputQueue;
        }
        
        @Override
        public void run() {
            final long stageStart = System.nanoTime();
            final int m = matrixA.length;
            
            try {
                while (true) {
                    MatrixStrip bStrip = inputQueue.take();
                    if (bStrip == MatrixStrip.POISON_PILL) break; // 🔥🔥 poison detected

                    double[][] partialC = new double[m][bStrip.endCol - bStrip.startCol];
                    
                    for (int i = 0; i < m; i++) {
                        for (int k = 0; k < matrixA[0].length; k++) {
                            for (int j = 0; j < partialC[0].length; j++) {
                                partialC[i][j] += matrixA[i][k] * bStrip.strip[k][j];
                            }
                        }
                    }
                    
                    outputQueue.put(new MatrixPartialResult(partialC, bStrip.startCol, bStrip.endCol));
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            totalTime += System.nanoTime() - stageStart;
        }
    }
    
    private static class WriterStage implements Runnable {
        private static volatile long totalTime = 0;
        private final double[][] matrixC;
        private final BlockingQueue<MatrixPartialResult> inputQueue;
        
        WriterStage(double[][] matrixC, BlockingQueue<MatrixPartialResult> inputQueue) {
            this.matrixC = matrixC;
            this.inputQueue = inputQueue;
        }
        
        @Override
        public void run() {
            final long stageStart = System.nanoTime();
            try {
                while (true) {
                    MatrixPartialResult partial = inputQueue.take();
                    if (partial == MatrixPartialResult.POISON_PILL) break; // 🔥🔥 poison detected

                    for (int i = 0; i < matrixC.length; i++) {
                        System.arraycopy(partial.partialC[i], 0, 
                                       matrixC[i], partial.startCol, 
                                       partial.endCol - partial.startCol);
                    }
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            totalTime += System.nanoTime() - stageStart;
        }
    }
}
