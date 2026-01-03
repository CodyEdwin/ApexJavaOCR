package com.apexocr.core.monitoring;

import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Thread-safe singleton service that bridges the training process and the visualizer.
 * Provides real-time access to training snapshots for visualization.
 */
public class VisualizationService {
    
    private static volatile VisualizationService instance;
    private final AtomicReference<TrainingSnapshot> latestSnapshot = new AtomicReference<>();
    private final AtomicReference<NetworkArchitecture> networkArchitecture = new AtomicReference<>();
    private volatile boolean isVisualizerConnected = false;
    private volatile boolean isPaused = false;
    
    private VisualizationService() {
        // Private constructor for singleton
    }
    
    /**
     * Get the singleton instance of VisualizationService.
     * @return The singleton instance
     */
    public static VisualizationService getInstance() {
        if (instance == null) {
            synchronized (VisualizationService.class) {
                if (instance == null) {
                    instance = new VisualizationService();
                }
            }
        }
        return instance;
    }
    
    /**
     * Push a training snapshot to the service.
     * Called by the training thread to update visualizations.
     * @param snapshot The snapshot to push
     */
    public void pushSnapshot(TrainingSnapshot snapshot) {
        latestSnapshot.set(snapshot);
    }
    
    /**
     * Get the latest training snapshot.
     * Called by the visualization thread to render current state.
     * @return Latest snapshot, or empty if no training data available
     */
    public Optional<TrainingSnapshot> getLatestSnapshot() {
        return Optional.ofNullable(latestSnapshot.get());
    }
    
    /**
     * Set the network architecture for visualization.
     * @param architecture Network architecture description
     */
    public void setNetworkArchitecture(NetworkArchitecture architecture) {
        networkArchitecture.set(architecture);
    }
    
    /**
     * Get the network architecture.
     * @return Network architecture, or empty if not set
     */
    public Optional<NetworkArchitecture> getNetworkArchitecture() {
        return Optional.ofNullable(networkArchitecture.get());
    }
    
    /**
     * Check if visualizer is connected and ready.
     * @return true if visualizer is connected
     */
    public boolean isVisualizerConnected() {
        return isVisualizerConnected;
    }
    
    /**
     * Set visualizer connection status.
     * @param connected true if visualizer is connected
     */
    public void setVisualizerConnected(boolean connected) {
        this.isVisualizerConnected = connected;
    }
    
    /**
     * Check if training is paused.
     * @return true if training is paused
     */
    public boolean isPaused() {
        return isPaused;
    }
    
    /**
     * Set training pause state.
     * @param paused true to pause training
     */
    public void setPaused(boolean paused) {
        this.isPaused = paused;
    }
    
    /**
     * Toggle pause state.
     * @return New pause state
     */
    public boolean togglePause() {
        this.isPaused = !this.isPaused;
        return this.isPaused;
    }
    
    /**
     * Reset the service state.
     * Called when training starts/ends.
     */
    public void reset() {
        latestSnapshot.set(null);
        isPaused = false;
    }
    
    /**
     * Get training progress as a percentage.
     * @return Progress value between 0.0 and 1.0
     */
    public float getProgress() {
        TrainingSnapshot snapshot = latestSnapshot.get();
        if (snapshot == null) return 0f;
        if (snapshot.totalEpochs == 0) return 0f;
        
        float epochProgress = (float) snapshot.epoch / snapshot.totalEpochs;
        float batchProgress = snapshot.totalBatches > 0 
            ? (float) snapshot.batch / snapshot.totalBatches 
            : 0f;
        
        return epochProgress + (batchProgress / snapshot.totalEpochs);
    }
    
    /**
     * Get current loss value.
     * @return Current loss, or 0 if not available
     */
    public float getCurrentLoss() {
        TrainingSnapshot snapshot = latestSnapshot.get();
        return snapshot != null ? snapshot.currentLoss : 0f;
    }
    
    /**
     * Get current accuracy.
     * @return Current accuracy, or 0 if not available
     */
    public float getCurrentAccuracy() {
        TrainingSnapshot snapshot = latestSnapshot.get();
        return snapshot != null ? snapshot.currentAccuracy : 0f;
    }
    
    /**
     * Get current learning rate.
     * @return Current learning rate
     */
    public float getCurrentLearningRate() {
        TrainingSnapshot snapshot = latestSnapshot.get();
        return snapshot != null ? snapshot.learningRate : 0.001f;
    }
    
    /**
     * Get current epoch.
     * @return Current epoch number
     */
    public int getCurrentEpoch() {
        TrainingSnapshot snapshot = latestSnapshot.get();
        return snapshot != null ? snapshot.epoch : 0;
    }
    
    /**
     * Get total epochs.
     * @return Total epochs to train
     */
    public int getTotalEpochs() {
        TrainingSnapshot snapshot = latestSnapshot.get();
        return snapshot != null ? snapshot.totalEpochs : 1;
    }
    
    /**
     * Get current batch.
     * @return Current batch number
     */
    public int getCurrentBatch() {
        TrainingSnapshot snapshot = latestSnapshot.get();
        return snapshot != null ? snapshot.batch : 0;
    }
    
    /**
     * Get total batches.
     * @return Total batches per epoch
     */
    public int getTotalBatches() {
        TrainingSnapshot snapshot = latestSnapshot.get();
        return snapshot != null ? snapshot.totalBatches : 1;
    }
}
