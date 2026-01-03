package com.apexocr.training.monitoring;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Console logger implementation for training events.
 * Provides formatted logging output to the console.
 */
public class ConsoleLogger implements TrainingListener {
    
    private static final DateTimeFormatter TIMESTAMP_FORMAT = 
        DateTimeFormatter.ofPattern("HH:mm:ss.SSS");
    
    private final LogLevel minLevel;
    private final boolean showTimestamps;
    private final int batchLogInterval;
    private int lastLoggedBatch = -1;
    
    public ConsoleLogger() {
        this(LogLevel.INFO, true, 10);
    }
    
    public ConsoleLogger(LogLevel minLevel, boolean showTimestamps, int batchLogInterval) {
        this.minLevel = minLevel;
        this.showTimestamps = showTimestamps;
        this.batchLogInterval = batchLogInterval;
    }
    
    @Override
    public void onEpochStart(int epoch, int totalEpochs) {
        log(LogLevel.INFO, String.format("=== EPOCH %d/%d STARTED ===", epoch + 1, totalEpochs));
    }
    
    @Override
    public void onEpochEnd(int epoch, float epochLoss, float epochAccuracy) {
        log(LogLevel.INFO, String.format("=== EPOCH %d COMPLETED ===", epoch));
        log(LogLevel.INFO, String.format("  Loss: %.6f | Accuracy: %.2f%%", epochLoss, epochAccuracy * 100));
    }
    
    @Override
    public void onBatchStart(int epoch, int batch, int totalBatches) {
        // No logging at batch start to avoid too much output
    }
    
    @Override
    public void onBatchEnd(int epoch, int batch, int totalBatches, float batchLoss, float batchAccuracy, long processingTimeMs) {
        // Log at intervals or if this is a special batch
        if (batch == 0 || batch % batchLogInterval == 0 || batch == totalBatches - 1) {
            log(LogLevel.DEBUG, String.format(
                "Batch %d/%d | Loss: %.6f | Acc: %.2f%% | Time: %dms",
                batch + 1, totalBatches, batchLoss, batchAccuracy * 100, processingTimeMs
            ));
        }
        lastLoggedBatch = batch;
    }
    
    @Override
    public void onTrainingComplete(int totalEpochs, float finalLoss, float finalAccuracy, long totalTrainingTimeMs) {
        log(LogLevel.INFO, "=".repeat(50));
        log(LogLevel.INFO, "TRAINING COMPLETE");
        log(LogLevel.INFO, "=".repeat(50));
        log(LogLevel.INFO, String.format("Total Epochs: %d", totalEpochs));
        log(LogLevel.INFO, String.format("Final Loss: %.6f", finalLoss));
        log(LogLevel.INFO, String.format("Final Accuracy: %.2f%%", finalAccuracy * 100));
        log(LogLevel.INFO, String.format("Training Time: %.2f seconds", totalTrainingTimeMs / 1000.0));
        log(LogLevel.INFO, "=".repeat(50));
    }
    
    @Override
    public void onError(int epoch, int batch, Exception error) {
        log(LogLevel.ERROR, String.format("ERROR at epoch %d, batch %d: %s", epoch, batch, error.getMessage()));
    }
    
    /**
     * Log a message at the specified level.
     * @param level Log level
     * @param message Message to log
     */
    private void log(LogLevel level, String message) {
        if (level.ordinal() < minLevel.ordinal()) {
            return;
        }
        
        StringBuilder sb = new StringBuilder();
        
        if (showTimestamps) {
            sb.append("[").append(LocalDateTime.now().format(TIMESTAMP_FORMAT)).append("] ");
        }
        
        sb.append("[").append(level.name()).append("] ");
        sb.append(message);
        
        switch (level) {
            case ERROR:
                System.err.println(sb);
                break;
            case WARN:
                System.out.println("\u001B[33m" + sb + "\u001B[0m"); // Yellow
                break;
            case DEBUG:
            case TRACE:
                System.out.println("\u001B[36m" + sb + "\u001B[0m"); // Cyan
                break;
            default:
                System.out.println(sb);
        }
    }
    
    /**
     * Log level enumeration.
     */
    public enum LogLevel {
        TRACE,  // Most verbose
        DEBUG,
        INFO,
        WARN,
        ERROR   // Least verbose
    }
}
