package com.apexocr.training.monitoring;

import java.util.Map;
import java.util.Optional;

/**
 * Immutable snapshot of training state for visualization.
 * Contains all data needed to render the current training state.
 */
public class TrainingSnapshot {
    
    public final int epoch;
    public final int totalEpochs;
    public final int batch;
    public final int totalBatches;
    public final float currentLoss;
    public final float currentAccuracy;
    public final float learningRate;
    public final long timestamp;
    public final Map<String, LayerSnapshot> layerSnapshots;
    public final SystemStats systemStats;
    public final TrainingPhase phase;
    public final float epochProgress;
    
    public TrainingSnapshot(Builder builder) {
        this.epoch = builder.epoch;
        this.totalEpochs = builder.totalEpochs;
        this.batch = builder.batch;
        this.totalBatches = builder.totalBatches;
        this.currentLoss = builder.currentLoss;
        this.currentAccuracy = builder.currentAccuracy;
        this.learningRate = builder.learningRate;
        this.timestamp = builder.timestamp;
        this.layerSnapshots = Map.copyOf(builder.layerSnapshots);
        this.systemStats = builder.systemStats;
        this.phase = builder.phase;
        this.epochProgress = totalBatches > 0 ? (float) batch / totalBatches : 0f;
    }
    
    /**
     * Create a builder for constructing snapshots.
     * @return New builder instance
     */
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * Get snapshot for a specific layer.
     * @param layerName Name of the layer
     * @return Layer snapshot if found
     */
    public Optional<LayerSnapshot> getLayerSnapshot(String layerName) {
        return Optional.ofNullable(layerSnapshots.get(layerName));
    }
    
    /**
     * Get total training progress.
     * @return Value between 0.0 and 1.0
     */
    public float getTotalProgress() {
        if (totalEpochs == 0) return 0f;
        float epochFraction = (float) epoch / totalEpochs;
        float batchFraction = totalBatches > 0 ? ((float) batch / totalBatches) / totalEpochs : 0f;
        return epochFraction + batchFraction;
    }
    
    /**
     * Check if training is in forward pass phase.
     * @return true if in forward pass
     */
    public boolean isForwardPass() {
        return phase == TrainingPhase.FORWARD_PASS;
    }
    
    /**
     * Check if training is in backward pass phase.
     * @return true if in backward pass
     */
    public boolean isBackwardPass() {
        return phase == TrainingPhase.BACKWARD_PASS;
    }
    
    /**
     * Check if training is in optimization phase.
     * @return true if in optimization
     */
    public boolean isOptimizing() {
        return phase == TrainingPhase.OPTIMIZATION;
    }
    
    /**
     * Training phase enumeration.
     */
    public enum TrainingPhase {
        IDLE,
        FORWARD_PASS,
        BACKWARD_PASS,
        OPTIMIZATION,
        EVALUATION
    }
    
    /**
     * System statistics snapshot.
     */
    public static class SystemStats {
        public final long usedMemoryMB;
        public final long totalMemoryMB;
        public final float memoryUsagePercent;
        public final double cpuUsagePercent;
        public final long processingTimeMs;
        
        public SystemStats(long usedMemoryMB, long totalMemoryMB, float memoryUsagePercent, 
                          double cpuUsagePercent, long processingTimeMs) {
            this.usedMemoryMB = usedMemoryMB;
            this.totalMemoryMB = totalMemoryMB;
            this.memoryUsagePercent = memoryUsagePercent;
            this.cpuUsagePercent = cpuUsagePercent;
            this.processingTimeMs = processingTimeMs;
        }
    }
    
    /**
     * Builder for constructing TrainingSnapshot instances.
     */
    public static class Builder {
        private int epoch = 0;
        private int totalEpochs = 1;
        private int batch = 0;
        private int totalBatches = 1;
        private float currentLoss = 0f;
        private float currentAccuracy = 0f;
        private float learningRate = 0.001f;
        private long timestamp = System.currentTimeMillis();
        private Map<String, LayerSnapshot> layerSnapshots = Map.of();
        private SystemStats systemStats;
        private TrainingPhase phase = TrainingPhase.IDLE;
        
        public Builder setEpoch(int epoch) {
            this.epoch = epoch;
            return this;
        }
        
        public Builder setTotalEpochs(int totalEpochs) {
            this.totalEpochs = totalEpochs;
            return this;
        }
        
        public Builder setBatch(int batch) {
            this.batch = batch;
            return this;
        }
        
        public Builder setTotalBatches(int totalBatches) {
            this.totalBatches = totalBatches;
            return this;
        }
        
        public Builder setCurrentLoss(float currentLoss) {
            this.currentLoss = currentLoss;
            return this;
        }
        
        public Builder setCurrentAccuracy(float currentAccuracy) {
            this.currentAccuracy = currentAccuracy;
            return this;
        }
        
        public Builder setLearningRate(float learningRate) {
            this.learningRate = learningRate;
            return this;
        }
        
        public Builder setTimestamp(long timestamp) {
            this.timestamp = timestamp;
            return this;
        }
        
        public Builder setLayerSnapshots(Map<String, LayerSnapshot> layerSnapshots) {
            this.layerSnapshots = layerSnapshots;
            return this;
        }
        
        public Builder setSystemStats(SystemStats systemStats) {
            this.systemStats = systemStats;
            return this;
        }
        
        public Builder setPhase(TrainingPhase phase) {
            this.phase = phase;
            return this;
        }
        
        public TrainingSnapshot build() {
            if (systemStats == null) {
                systemStats = new SystemStats(0, 0, 0, 0, 0);
            }
            return new TrainingSnapshot(this);
        }
    }
}
