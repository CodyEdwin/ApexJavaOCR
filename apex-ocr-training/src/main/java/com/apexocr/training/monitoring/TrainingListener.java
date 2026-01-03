package com.apexocr.training.monitoring;

/**
 * Listener interface for training events.
 * Implementations can receive callbacks at various points during training.
 */
public interface TrainingListener {
    
    /**
     * Called when an epoch starts.
     * @param epoch Current epoch number (0-indexed)
     * @param totalEpochs Total number of epochs
     */
    default void onEpochStart(int epoch, int totalEpochs) {}
    
    /**
     * Called when an epoch ends.
     * @param epoch Current epoch number
     * @param epochLoss Average loss for this epoch
     * @param epochAccuracy Accuracy for this epoch
     */
    default void onEpochEnd(int epoch, float epochLoss, float epochAccuracy) {}
    
    /**
     * Called when a batch starts.
     * @param epoch Current epoch
     * @param batch Current batch
     * @param totalBatches Total batches per epoch
     */
    default void onBatchStart(int epoch, int batch, int totalBatches) {}
    
    /**
     * Called when a batch completes.
     * @param epoch Current epoch
     * @param batch Current batch
     * @param totalBatches Total batches per epoch
     * @param batchLoss Loss for this batch
     * @param batchAccuracy Accuracy for this batch
     * @param processingTimeMs Time taken to process this batch
     */
    default void onBatchEnd(int epoch, int batch, int totalBatches, float batchLoss, float batchAccuracy, long processingTimeMs) {}
    
    /**
     * Called when training completes.
     * @param totalEpochs Number of epochs trained
     * @param finalLoss Final loss value
     * @param finalAccuracy Final accuracy
     * @param totalTrainingTimeMs Total training time
     */
    default void onTrainingComplete(int totalEpochs, float finalLoss, float finalAccuracy, long totalTrainingTimeMs) {}
    
    /**
     * Called when an error occurs.
     * @param epoch Current epoch
     * @param batch Current batch
     * @param error The exception that occurred
     */
    default void onError(int epoch, int batch, Exception error) {}
}
