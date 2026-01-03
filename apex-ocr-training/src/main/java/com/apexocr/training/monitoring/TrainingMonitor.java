package com.apexocr.training.monitoring;

/**
 * Interface for monitoring training progress and collecting metrics.
 * Implementations can hook into the training loop at various points.
 */
public interface TrainingMonitor {
    
    /**
     * Called when a new epoch starts.
     * @param epoch Current epoch number (0-indexed)
     * @param totalEpochs Total number of epochs
     */
    void onEpochStart(int epoch, int totalEpochs);
    
    /**
     * Called when an epoch ends.
     * @param epoch Current epoch number
     * @param epochLoss Average loss for this epoch
     * @param epochAccuracy Accuracy for this epoch
     */
    void onEpochEnd(int epoch, float epochLoss, float epochAccuracy);
    
    /**
     * Called when a new batch starts.
     * @param epoch Current epoch
     * @param batch Current batch
     * @param totalBatches Total batches per epoch
     */
    void onBatchStart(int epoch, int batch, int totalBatches);
    
    /**
     * Called when a batch completes processing.
     * @param epoch Current epoch
     * @param batch Current batch
     * @param totalBatches Total batches per epoch
     * @param batchLoss Loss for this batch
     * @param batchAccuracy Accuracy for this batch
     * @param processingTimeMs Time taken to process this batch in milliseconds
     */
    void onBatchEnd(int epoch, int batch, int totalBatches, float batchLoss, float batchAccuracy, long processingTimeMs);
    
    /**
     * Called when training completes successfully.
     * @param totalEpochs Number of epochs trained
     * @param finalLoss Final loss value
     * @param finalAccuracy Final accuracy
     * @param totalTrainingTimeMs Total training time in milliseconds
     */
    void onTrainingComplete(int totalEpochs, float finalLoss, float finalAccuracy, long totalTrainingTimeMs);
    
    /**
     * Called when training encounters an error.
     * @param epoch Current epoch
     * @param batch Current batch
     * @param error The exception that occurred
     */
    void onError(int epoch, int batch, Exception error);
    
    /**
     * Called to check if monitoring should continue.
     * @return true if training should continue, false to abort
     */
    boolean shouldContinue();
}
