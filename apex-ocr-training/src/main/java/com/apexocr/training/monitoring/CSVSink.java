package com.apexocr.training.monitoring;

import java.io.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * CSV sink for exporting training metrics to CSV files.
 * Supports both batch-level and epoch-level logging.
 */
public class CSVSink implements TrainingListener, Runnable {
    
    private final String basePath;
    private final String filePrefix;
    private final AtomicBoolean isRunning = new AtomicBoolean(false);
    private final ConcurrentLinkedQueue<String> writeQueue = new ConcurrentLinkedQueue<>();
    private final DateTimeFormatter dateFormat = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");
    
    private PrintWriter batchWriter;
    private PrintWriter epochWriter;
    private PrintWriter layerWriter;
    private volatile boolean isInitialized = false;
    
    private int totalBatches = 0;
    private int currentEpoch = 0;
    
    public CSVSink() {
        this("training_logs", "apex_ocr");
    }
    
    public CSVSink(String basePath, String filePrefix) {
        this.basePath = basePath;
        this.filePrefix = filePrefix;
    }
    
    /**
     * Initialize the CSV writers.
     */
    public synchronized void initialize() {
        if (isInitialized) return;
        
        // Create base directory
        new File(basePath).mkdirs();
        
        String timestamp = LocalDateTime.now().format(dateFormat);
        
        try {
            // Batch metrics file
            String batchFile = String.format("%s/%s_batch_metrics_%s.csv", basePath, filePrefix, timestamp);
            batchWriter = new PrintWriter(new FileWriter(batchFile));
            batchWriter.println("epoch,batch,total_batches,loss,accuracy,learning_rate,processing_time_ms,timestamp");
            
            // Epoch metrics file
            String epochFile = String.format("%s/%s_epoch_metrics_%s.csv", basePath, filePrefix, timestamp);
            epochWriter = new PrintWriter(new FileWriter(epochFile));
            epochWriter.println("epoch,total_epochs,avg_loss,accuracy,total_batches,duration_ms,timestamp");
            
            // Layer statistics file
            String layerFile = String.format("%s/%s_layer_stats_%s.csv", basePath, filePrefix, timestamp);
            layerWriter = new PrintWriter(new FileWriter(layerFile));
            layerWriter.println("epoch,batch,layer_name,layer_type,activation_mean,activation_std,gradient_mean,gradient_std,weight_mean,weight_std");
            
            isInitialized = true;
            
            // Start background writer thread
            isRunning.set(true);
            Thread writerThread = new Thread(this, "CSV-Writer");
            writerThread.setDaemon(true);
            writerThread.start();
            
        } catch (IOException e) {
            System.err.println("[APEX-CSV] Failed to initialize CSV writers: " + e.getMessage());
        }
    }
    
    @Override
    public void onEpochStart(int epoch, int totalEpochs) {
        this.currentEpoch = epoch;
        if (!isInitialized) initialize();
    }
    
    @Override
    public void onEpochEnd(int epoch, float epochLoss, float epochAccuracy) {
        if (!isInitialized) return;
        
        String line = String.format("%d,%d,%.6f,%.4f,%d,%d,%s",
            epoch + 1,
            totalBatches,
            epochLoss,
            epochAccuracy,
            totalBatches,
            System.currentTimeMillis(),
            LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)
        );
        
        writeQueue.offer("EPOCH:" + line);
    }
    
    @Override
    public void onBatchStart(int epoch, int batch, int totalBatches) {
        this.totalBatches = totalBatches;
    }
    
    @Override
    public void onBatchEnd(int epoch, int batch, int totalBatches, float batchLoss, float batchAccuracy, long processingTimeMs) {
        if (!isInitialized) return;
        
        String line = String.format("%d,%d,%d,%.6f,%.4f,%.6f,%d,%s",
            epoch + 1,
            batch + 1,
            totalBatches,
            batchLoss,
            batchAccuracy,
            0.001f, // learning rate placeholder
            processingTimeMs,
            LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)
        );
        
        writeQueue.offer("BATCH:" + line);
    }
    
    @Override
    public void onTrainingComplete(int totalEpochs, float finalLoss, float finalAccuracy, long totalTrainingTimeMs) {
        isRunning.set(false);
        
        // Flush remaining writes
        flush();
        
        if (batchWriter != null) batchWriter.close();
        if (epochWriter != null) epochWriter.close();
        if (layerWriter != null) layerWriter.close();
        
        System.out.println("[APEX-CSV] Training logs saved to " + basePath);
    }
    
    @Override
    public void onError(int epoch, int batch, Exception error) {
        // Log errors to a separate file
        if (!isInitialized) return;
        
        try {
            FileWriter errorWriter = new FileWriter(basePath + "/training_errors.log", true);
            PrintWriter pw = new PrintWriter(errorWriter);
            pw.printf("[%s] ERROR at epoch %d, batch %d: %s%n",
                LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME),
                epoch, batch, error.getMessage());
            pw.close();
        } catch (IOException e) {
            System.err.println("[APEX-CSV] Failed to write error log: " + e.getMessage());
        }
    }
    
    /**
     * Log layer statistics.
     * @param epoch Current epoch
     * @param batch Current batch
     * @param stats Layer statistics
     */
    public void logLayerStats(int epoch, int batch, ApexStatsMonitor.LayerStats stats) {
        if (!isInitialized || layerWriter == null) return;
        
        String line = String.format("%d,%d,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
            epoch + 1,
            batch + 1,
            stats.layerName,
            "DENSE", // layer type placeholder
            stats.activationMean,
            stats.activationStd,
            stats.gradientMean,
            stats.gradientStd,
            stats.weightMean,
            stats.weightStd
        );
        
        writeQueue.offer("LAYER:" + line);
    }
    
    /**
     * Log layer statistics for multiple layers.
     * @param epoch Current epoch
     * @param batch Current batch
     * @param statsList List of layer statistics
     */
    public void logAllLayerStats(int epoch, int batch, List<ApexStatsMonitor.LayerStats> statsList) {
        for (ApexStatsMonitor.LayerStats stats : statsList) {
            logLayerStats(epoch, batch, stats);
        }
    }
    
    @Override
    public void run() {
        while (isRunning.get() || !writeQueue.isEmpty()) {
            try {
                String item = writeQueue.poll();
                if (item != null) {
                    String[] parts = item.split(":", 2);
                    if (parts.length == 2) {
                        String type = parts[0];
                        String data = parts[1];
                        
                        switch (type) {
                            case "BATCH":
                                if (batchWriter != null) {
                                    batchWriter.println(data);
                                    batchWriter.flush();
                                }
                                break;
                            case "EPOCH":
                                if (epochWriter != null) {
                                    epochWriter.println(data);
                                    epochWriter.flush();
                                }
                                break;
                            case "LAYER":
                                if (layerWriter != null) {
                                    layerWriter.println(data);
                                }
                                break;
                        }
                    }
                } else {
                    // Wait for more items
                    Thread.sleep(10);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }
    
    /**
     * Force flush all pending writes.
     */
    public void flush() {
        // Process remaining queue items
        String item;
        while ((item = writeQueue.poll()) != null) {
            try {
                String[] parts = item.split(":", 2);
                if (parts.length == 2) {
                    String type = parts[0];
                    String data = parts[1];
                    
                    switch (type) {
                        case "BATCH":
                            if (batchWriter != null) batchWriter.println(data);
                            break;
                        case "EPOCH":
                            if (epochWriter != null) epochWriter.println(data);
                            break;
                        case "LAYER":
                            if (layerWriter != null) layerWriter.println(data);
                            break;
                    }
                }
            } catch (Exception e) {
                System.err.println("[APEX-CSV] Flush error: " + e.getMessage());
            }
        }
        
        if (batchWriter != null) batchWriter.flush();
        if (epochWriter != null) epochWriter.flush();
        if (layerWriter != null) layerWriter.flush();
    }
    
    /**
     * Get the path to the batch metrics CSV file.
     * @return File path or null if not initialized
     */
    public String getBatchMetricsPath() {
        // This would need to be stored during initialization
        return null;
    }
    
    /**
     * Get the path to the epoch metrics CSV file.
     * @return File path or null if not initialized
     */
    public String getEpochMetricsPath() {
        // This would need to be stored during initialization
        return null;
    }
}
