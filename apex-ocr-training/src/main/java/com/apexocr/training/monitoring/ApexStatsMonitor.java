package com.apexocr.training.monitoring;

import com.apexocr.core.neural.Layer;
import com.apexocr.core.tensor.Tensor;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Comprehensive statistics monitor for OCR training.
 * Collects and reports detailed metrics about training progress,
 * layer statistics, gradients, and system resources.
 */
public class ApexStatsMonitor implements TrainingMonitor {
    
    private final List<TrainingListener> listeners = new ArrayList<>();
    private final Map<String, LayerStats> layerStats = new ConcurrentHashMap<>();
    private final AtomicBoolean continueTraining = new AtomicBoolean(true);
    private final AtomicLong batchCounter = new AtomicLong(0);
    
    private long trainingStartTime;
    private long epochStartTime;
    private float bestLoss = Float.MAX_VALUE;
    private int patienceCounter = 0;
    private final int patience = 5; // Early stopping patience
    
    // Gradient monitoring thresholds
    private static final float VANISHING_GRADIENT_THRESHOLD = 1e-7f;
    private static final float EXPLODING_GRADIENT_THRESHOLD = 1e3f;
    
    // Metrics history for visualization
    private final Deque<BatchMetrics> batchHistory = new ArrayDeque<>(1000);
    private final Deque<EpochMetrics> epochHistory = new ArrayDeque<>(100);
    
    public ApexStatsMonitor() {
        // Listeners are added externally via addListener()
        // Default logging is configured when added to OCRTrainer
    }
    
    @Override
    public void onEpochStart(int epoch, int totalEpochs) {
        this.epochStartTime = System.currentTimeMillis();
        epochHistory.clear(); // Clear batch-level history for new epoch
        
        EpochMetrics epochMetrics = new EpochMetrics(epoch, totalEpochs);
        epochMetrics.startTime = System.currentTimeMillis();
        
        fireEvent(metric -> metric.onEpochStart(epoch, totalEpochs));
    }
    
    @Override
    public void onEpochEnd(int epoch, float epochLoss, float epochAccuracy) {
        long epochTime = System.currentTimeMillis() - epochStartTime;
        
        EpochMetrics epochMetrics = new EpochMetrics(epoch, epochHistory.size());
        epochMetrics.endTime = System.currentTimeMillis();
        epochMetrics.durationMs = epochTime;
        epochMetrics.avgLoss = epochLoss;
        epochMetrics.accuracy = epochAccuracy;
        epochMetrics.totalBatches = batchHistory.size();
        
        // Calculate epoch statistics
        if (!batchHistory.isEmpty()) {
            float totalLoss = batchHistory.stream()
                .map(m -> m.loss)
                .reduce(0f, Float::sum);
            epochMetrics.avgLoss = totalLoss / batchHistory.size();
        }
        
        epochHistory.addFirst(epochMetrics);
        if (epochHistory.size() > 100) epochHistory.removeLast();
        
        // Early stopping check
        if (epochLoss < bestLoss) {
            bestLoss = epochLoss;
            patienceCounter = 0;
        } else {
            patienceCounter++;
        }
        
        fireEvent(metric -> metric.onEpochEnd(epoch, epochLoss, epochAccuracy));
        
        // Check for early stopping
        if (patienceCounter >= patience) {
            System.out.println("[APEX-STATS] Early stopping triggered after " + epoch + " epochs");
            continueTraining.set(false);
        }
    }
    
    @Override
    public void onBatchStart(int epoch, int batch, int totalBatches) {
        // Reset batch-level statistics
        batchCounter.incrementAndGet();
    }
    
    @Override
    public void onBatchEnd(int epoch, int batch, int totalBatches, float batchLoss, float batchAccuracy, long processingTimeMs) {
        BatchMetrics metrics = new BatchMetrics();
        metrics.epoch = epoch;
        metrics.batch = batch;
        metrics.totalBatches = totalBatches;
        metrics.loss = batchLoss;
        metrics.accuracy = batchAccuracy;
        metrics.processingTimeMs = processingTimeMs;
        metrics.timestamp = System.currentTimeMillis();
        
        batchHistory.addFirst(metrics);
        if (batchHistory.size() > 1000) batchHistory.removeLast();
        
        fireEvent(metric -> metric.onBatchEnd(epoch, batch, totalBatches, batchLoss, batchAccuracy, processingTimeMs));
    }
    
    @Override
    public void onTrainingComplete(int totalEpochs, float finalLoss, float finalAccuracy, long totalTrainingTimeMs) {
        fireEvent(metric -> metric.onTrainingComplete(totalEpochs, finalLoss, finalAccuracy, totalTrainingTimeMs));
        
        // Print final summary
        printTrainingSummary(totalEpochs, finalLoss, finalAccuracy, totalTrainingTimeMs);
    }
    
    @Override
    public void onError(int epoch, int batch, Exception error) {
        System.err.println("[APEX-STATS] ERROR at epoch " + epoch + ", batch " + batch + ": " + error.getMessage());
        error.printStackTrace();
        
        fireEvent(metric -> metric.onError(epoch, batch, error));
    }
    
    @Override
    public boolean shouldContinue() {
        return continueTraining.get();
    }
    
    /**
     * Record gradient statistics for a layer.
     * @param layerName Name of the layer
     * @param gradients Tensor containing gradients
     */
    public void recordGradients(String layerName, Tensor gradients) {
        float[] data = gradients.getFloatArray((int) gradients.getSize());
        if (data == null || data.length == 0) return;
        
        LayerStats stats = layerStats.computeIfAbsent(layerName, k -> new LayerStats(k));
        
        float sum = 0, sumSq = 0, min = Float.MAX_VALUE, max = Float.MIN_VALUE;
        for (float v : data) {
            sum += v;
            sumSq += v * v;
            min = Math.min(min, v);
            max = Math.max(max, v);
        }
        
        int n = data.length;
        stats.gradientMean = sum / n;
        stats.gradientStd = (float) Math.sqrt((sumSq / n) - (sum / n) * (sum / n));
        stats.gradientMin = min;
        stats.gradientMax = max;
        stats.gradientL2Norm = (float) Math.sqrt(sumSq);
        
        // Check for vanishing/exploding gradients
        if (Math.abs(stats.gradientMean) < VANISHING_GRADIENT_THRESHOLD) {
            System.out.println("[APEX-STATS] WARNING: Vanishing gradients detected in layer '" + layerName + 
                             "', mean: " + stats.gradientMean);
        }
        if (Math.abs(stats.gradientMean) > EXPLODING_GRADIENT_THRESHOLD) {
            System.out.println("[APEX-STATS] WARNING: Exploding gradients detected in layer '" + layerName + 
                             "', mean: " + stats.gradientMean);
        }
    }
    
    /**
     * Record weight statistics for a layer.
     * @param layerName Name of the layer
     * @param weights Tensor containing weights
     */
    public void recordWeights(String layerName, Tensor weights) {
        float[] data = weights.getFloatArray((int) weights.getSize());
        if (data == null || data.length == 0) return;
        
        LayerStats stats = layerStats.computeIfAbsent(layerName, k -> new LayerStats(k));
        
        float sum = 0, sumSq = 0, min = Float.MAX_VALUE, max = Float.MIN_VALUE;
        for (float v : data) {
            sum += v;
            sumSq += v * v;
            min = Math.min(min, v);
            max = Math.max(max, v);
        }
        
        int n = data.length;
        stats.weightMean = sum / n;
        stats.weightStd = (float) Math.sqrt((sumSq / n) - (sum / n) * (sum / n));
        stats.weightMin = min;
        stats.weightMax = max;
        stats.weightCount = n;
    }
    
    /**
     * Record activation statistics for a layer.
     * @param layerName Name of the layer
     * @param activations Tensor containing activations
     */
    public void recordActivations(String layerName, Tensor activations) {
        float[] data = activations.getFloatArray((int) activations.getSize());
        if (data == null || data.length == 0) return;
        
        LayerStats stats = layerStats.computeIfAbsent(layerName, k -> new LayerStats(k));
        
        float sum = 0, sumSq = 0, min = Float.MAX_VALUE, max = Float.MIN_VALUE;
        for (float v : data) {
            sum += v;
            sumSq += v * v;
            min = Math.min(min, v);
            max = Math.max(max, v);
        }
        
        int n = data.length;
        stats.activationMean = sum / n;
        stats.activationStd = (float) Math.sqrt((sumSq / n) - (sum / n) * (sum / n));
        stats.activationMin = min;
        stats.activationMax = max;
    }
    
    /**
     * Get the latest batch metrics for visualization.
     * @return Latest batch metrics or empty if no batches recorded
     */
    public Optional<BatchMetrics> getLatestBatchMetrics() {
        return batchHistory.isEmpty() ? Optional.empty() : Optional.of(batchHistory.peekFirst());
    }
    
    /**
     * Get batch history for plotting.
     * @return Copy of batch history
     */
    public List<BatchMetrics> getBatchHistory() {
        return new ArrayList<>(batchHistory);
    }
    
    /**
     * Get epoch history for plotting.
     * @return Copy of epoch history
     */
    public List<EpochMetrics> getEpochHistory() {
        return new ArrayList<>(epochHistory);
    }
    
    /**
     * Get statistics for all layers.
     * @return Map of layer name to statistics
     */
    public Map<String, LayerStats> getLayerStats() {
        return new ConcurrentHashMap<>(layerStats);
    }
    
    /**
     * Get current training progress.
     * @return Progress as a value between 0 and 1
     */
    public float getProgress() {
        if (epochHistory.isEmpty()) return 0f;
        EpochMetrics latest = epochHistory.peekFirst();
        return (float) latest.epoch / 100; // Simplified progress calculation
    }
    
    /**
     * Get current learning rate (from optimizer).
     * @return Current learning rate
     */
    public float getCurrentLearningRate() {
        return 0.001f; // Default, should be updated by optimizer
    }
    
    /**
     * Get memory usage statistics.
     * @return Memory usage metrics
     */
    public MemoryStats getMemoryStats() {
        Runtime runtime = Runtime.getRuntime();
        long totalMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();
        long maxMemory = runtime.maxMemory();
        long usedMemory = totalMemory - freeMemory;
        
        MemoryStats stats = new MemoryStats();
        stats.usedMemoryMB = usedMemory / (1024 * 1024);
        stats.totalMemoryMB = totalMemory / (1024 * 1024);
        stats.maxMemoryMB = maxMemory / (1024 * 1024);
        stats.freeMemoryMB = freeMemory / (1024 * 1024);
        stats.usagePercent = (usedMemory * 100f) / maxMemory;
        
        return stats;
    }
    
    /**
     * Get total number of batches processed.
     * @return Batch counter value
     */
    public long getTotalBatchesProcessed() {
        return batchCounter.get();
    }
    
    /**
     * Add a custom listener for training events.
     * @param listener Listener to add
     */
    public void addListener(TrainingListener listener) {
        listeners.add(listener);
    }
    
    private void fireEvent(java.util.function.Consumer<TrainingListener> action) {
        for (TrainingListener listener : listeners) {
            try {
                action.accept(listener);
            } catch (Exception e) {
                System.err.println("[APEX-STATS] Listener error: " + e.getMessage());
            }
        }
    }
    
    private void printTrainingSummary(int totalEpochs, float finalLoss, float finalAccuracy, long totalTrainingTimeMs) {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("[APEX-STATS] TRAINING COMPLETE");
        System.out.println("=".repeat(60));
        System.out.printf("Total Epochs: %d%n", totalEpochs);
        System.out.printf("Final Loss: %.6f%n", finalLoss);
        System.out.printf("Final Accuracy: %.2f%%%n", finalAccuracy * 100);
        System.out.printf("Total Training Time: %.2f seconds%n", totalTrainingTimeMs / 1000.0);
        System.out.printf("Total Batches Processed: %d%n", batchCounter.get());
        System.out.printf("Best Loss Achieved: %.6f%n", bestLoss);
        System.out.println("=".repeat(60) + "\n");
    }
    
    // Inner classes for metrics and statistics
    
    public static class BatchMetrics {
        public int epoch;
        public int batch;
        public int totalBatches;
        public float loss;
        public float accuracy;
        public long processingTimeMs;
        public long timestamp;
        
        @Override
        public String toString() {
            return String.format("Batch[epoch=%d, batch=%d, loss=%.6f, acc=%.2f%%, time=%dms]",
                               epoch, batch, loss, accuracy * 100, processingTimeMs);
        }
    }
    
    public static class EpochMetrics {
        public int epoch;
        public int totalEpochs;
        public float avgLoss;
        public float accuracy;
        public int totalBatches;
        public long startTime;
        public long endTime;
        public long durationMs;
        
        public EpochMetrics(int epoch, int totalEpochs) {
            this.epoch = epoch;
            this.totalEpochs = totalEpochs;
        }
        
        @Override
        public String toString() {
            return String.format("Epoch[epoch=%d/%d, loss=%.6f, acc=%.2f%%, batches=%d, time=%ds]",
                               epoch, totalEpochs, avgLoss, accuracy * 100, totalBatches, durationMs / 1000);
        }
    }
    
    public static class LayerStats {
        public final String layerName;
        public float gradientMean;
        public float gradientStd;
        public float gradientMin;
        public float gradientMax;
        public float gradientL2Norm;
        public float weightMean;
        public float weightStd;
        public float weightMin;
        public float weightMax;
        public int weightCount;
        public float activationMean;
        public float activationStd;
        public float activationMin;
        public float activationMax;
        
        public LayerStats(String layerName) {
            this.layerName = layerName;
        }
    }
    
    public static class MemoryStats {
        public long usedMemoryMB;
        public long totalMemoryMB;
        public long maxMemoryMB;
        public long freeMemoryMB;
        public float usagePercent;
    }
}
