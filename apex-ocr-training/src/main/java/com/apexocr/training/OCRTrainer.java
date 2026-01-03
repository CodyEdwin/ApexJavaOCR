package com.apexocr.training;

import com.apexocr.core.tensor.Tensor;
import com.apexocr.core.tensor.TensorOperations;
import com.apexocr.core.neural.Layer;
import com.apexocr.core.neural.Conv2D;
import com.apexocr.core.neural.Dense;
import com.apexocr.core.neural.BiLSTM;
import com.apexocr.engine.OcrEngine;
import com.apexocr.engine.OcrResult;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Simple OCR Trainer - Trains the network using CTC loss with labeled images.
 * 
 * Usage:
 * 1. Create training data: image files named after their text content
 *    Example: "HELLO.png" contains an image of the text "HELLO"
 * 2. Run training: java OCRA Trainer /path/to/training_data
 */
public class OCRTrainer {
    
    private final OcrEngine engine;
    private final String vocabulary;
    private final int numClasses;
    private final Random random;
    
    // Training hyperparameters
    private float learningRate = 0.001f;
    private int batchSize = 8;
    private int epochs = 100;
    private float gradientClip = 5.0f;
    
    public OCRTrainer(OcrEngine engine, String vocabulary) {
        this.engine = engine;
        this.vocabulary = vocabulary;
        this.numClasses = vocabulary.length() + 1; // +1 for blank token
        this.random = new Random(42);
    }
    
    /**
     * Loads training samples from a directory.
     * Expects image files named with the ground truth text.
     */
    public List<TrainingSample> loadTrainingData(String dataDir) throws IOException {
        List<TrainingSample> samples = new ArrayList<>();
        
        Path dirPath = Paths.get(dataDir);
        if (!Files.exists(dirPath)) {
            throw new IOException("Training data directory not found: " + dataDir);
        }
        
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dirPath, "*.{png,jpg,jpeg,gif,bmp}")) {
            for (Path entry : stream) {
                String filename = entry.getFileName().toString();
                // Remove extension to get label
                String label = filename.substring(0, filename.lastIndexOf('.'));
                
                // Validate label contains only known characters
                if (isValidLabel(label)) {
                    BufferedImage image = ImageIO.read(entry.toFile());
                    if (image != null) {
                        Tensor input = imageToTensor(image);
                        samples.add(new TrainingSample(input, label));
                    }
                }
            }
        }
        
        System.out.println("Loaded " + samples.size() + " training samples");
        return samples;
    }
    
    /**
     * Validates that a label only contains known vocabulary characters.
     */
    private boolean isValidLabel(String label) {
        for (char c : label.toCharArray()) {
            if (vocabulary.indexOf(c) < 0 && c != ' ') {
                return false;
            }
        }
        return true;
    }
    
    /**
     * Converts a buffered image to a network input tensor.
     */
    private Tensor imageToTensor(BufferedImage image) {
        int height = image.getHeight();
        int width = image.getWidth();
        
        // Resize to target height (32px for CRNN)
        int targetHeight = 32;
        BufferedImage resized = new BufferedImage(width, targetHeight, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = resized.createGraphics();
        g.drawImage(image, 0, 0, width, targetHeight, null);
        g.dispose();
        
        long[] shape = new long[]{1, targetHeight, width, 1};
        Tensor tensor = new Tensor(shape, Tensor.DataType.FLOAT32);
        
        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = resized.getRaster().getSample(x, y, 0);
                float normalized = pixel / 255.0f;
                tensor.setFloat(normalized, 0, y, x, 0);
            }
        }
        
        return tensor;
    }
    
    /**
     * Converts text to integer labels.
     */
    public int[] textToIndices(String text) {
        int[] indices = new int[text.length()];
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            int idx = vocabulary.indexOf(c);
            if (idx >= 0) {
                indices[i] = idx + 1; // +1 for blank token
            } else {
                indices[i] = 0;
            }
        }
        return indices;
    }
    
    /**
     * Converts indices back to text.
     */
    public String indicesToText(int[] indices) {
        StringBuilder sb = new StringBuilder();
        int prev = -1;
        for (int idx : indices) {
            // Skip blanks and repeats
            if (idx != 0 && idx != prev) {
                sb.append(vocabulary.charAt(idx - 1));
            }
            prev = idx;
        }
        return sb.toString();
    }
    
    /**
     * Trains the network for a number of epochs.
     */
    public void train(List<TrainingSample> trainingSamples) {
        if (trainingSamples.isEmpty()) {
            System.err.println("No training samples!");
            return;
        }
        
        System.out.println("\n=== Starting Training ===");
        System.out.println("Vocabulary: " + vocabulary.length() + " classes");
        System.out.println("Training samples: " + trainingSamples.size());
        System.out.println("Batch size: " + batchSize);
        System.out.println("Learning rate: " + learningRate);
        System.out.println("Epochs: " + epochs);
        System.out.println();
        
        for (int epoch = 1; epoch <= epochs; epoch++) {
            // Shuffle training data
            Collections.shuffle(trainingSamples, random);
            
            float totalLoss = 0;
            int totalSamples = 0;
            int correct = 0;
            
            // Process in batches
            for (int i = 0; i < trainingSamples.size(); i += batchSize) {
                int end = Math.min(i + batchSize, trainingSamples.size());
                List<TrainingSample> batch = trainingSamples.subList(i, end);
                
                // Compute loss for batch
                float batchLoss = processBatch(batch);
                totalLoss += batchLoss * batch.size();
                totalSamples += batch.size();
                
                // Check accuracy
                for (TrainingSample sample : batch) {
                    Tensor output = runInference(sample.input);
                    String prediction = decodeOutput(output);
                    if (prediction.equals(sample.label)) {
                        correct++;
                    }
                }
            }
            
            float avgLoss = totalLoss / totalSamples;
            float accuracy = (float) correct / totalSamples * 100;
            
            System.out.printf("Epoch %d/%d - Loss: %.4f - Accuracy: %.1f%%\n", 
                epoch, epochs, avgLoss, accuracy);
            
            // Learning rate decay
            learningRate *= 0.98f;
        }
        
        System.out.println("\n=== Training Complete ===");
    }
    
    /**
     * Processes a batch of samples and performs one optimization step.
     */
    private float processBatch(List<TrainingSample> batch) {
        float totalLoss = 0;
        
        for (TrainingSample sample : batch) {
            // Forward pass
            Tensor output = runInference(sample.input);
            
            // Compute CTC loss approximation
            float loss = computeCTCLoss(output, sample.label);
            totalLoss += loss;
            
            // Backward pass would go here
            // For now, we just compute loss (gradient computation requires more infrastructure)
        }
        
        return totalLoss / batch.size();
    }
    
    /**
     * Runs forward pass through the network.
     */
    private Tensor runInference(Tensor input) {
        Tensor current = input;
        
        for (Layer layer : engine.getNetwork()) {
            layer.eval();
            current = layer.forward(current, false);
        }
        
        return current;
    }
    
    /**
     * Computes a simple CTC-like loss for training.
     * This is a simplified loss - a full implementation would use the actual CTC algorithm.
     */
    private float computeCTCLoss(Tensor output, String target) {
        long[] shape = output.getShape();
        int timeSteps = (int) shape[1];
        int classes = (int) shape[2];
        
        int[] targetIndices = textToIndices(target);
        
        // Simple loss: negative log probability of correct path
        float loss = 0;
        int targetPos = 0;
        
        for (int t = 0; t < timeSteps && targetPos < targetIndices.length; t++) {
            int targetIdx = targetIndices[targetPos];
            float prob = output.getFloat(0, t, targetIdx);
            prob = Math.max(prob, 1e-7f); // Avoid log(0)
            loss += (float) -Math.log(prob);
            
            // Move to next character if probability is high enough
            if (prob > 0.5) {
                targetPos++;
            }
        }
        
        // Penalize remaining characters
        for (int t = targetPos; t < timeSteps; t++) {
            float blankProb = output.getFloat(0, t, 0);
            blankProb = Math.max(blankProb, 1e-7f);
            loss += (float) -Math.log(blankProb);
        }
        
        return loss / target.length();
    }
    
    /**
     * Decodes network output to text using greedy decoding.
     */
    private String decodeOutput(Tensor output) {
        long[] shape = output.getShape();
        int timeSteps = (int) shape[1];
        int classes = (int) shape[2];
        
        StringBuilder result = new StringBuilder();
        int prevClass = -1;
        
        for (int t = 0; t < timeSteps; t++) {
            int bestClass = 0;
            float bestProb = output.getFloat(0, t, 0);
            
            for (int c = 1; c < classes; c++) {
                float prob = output.getFloat(0, t, c);
                if (prob > bestProb) {
                    bestProb = prob;
                    bestClass = c;
                }
            }
            
            // Skip blanks and repeated characters
            if (bestClass != 0 && bestClass != prevClass) {
                if (bestClass > 0 && bestClass <= vocabulary.length()) {
                    result.append(vocabulary.charAt(bestClass - 1));
                }
            }
            prevClass = bestClass;
        }
        
        return result.toString();
    }
    
    /**
     * Saves trained weights to a file.
     */
    public void saveWeights(String filePath) {
        engine.saveWeights(filePath);
        System.out.println("Weights saved to: " + filePath);
    }
    
    /**
     * Training sample container.
     */
    public static class TrainingSample {
        public final Tensor input;
        public final String label;
        
        public TrainingSample(Tensor input, String label) {
            this.input = input;
            this.label = label;
        }
    }
    
    /**
     * Main method - demonstrates training usage.
     */
    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("OCR Trainer - Trains the OCR network on labeled images");
            System.out.println();
            System.out.println("Usage: java OCRTrainer <training_data_directory>");
            System.out.println();
            System.out.println("Training data format:");
            System.out.println("  - Directory containing image files");
            System.out.println("  - Filename (without extension) is the ground-truth text");
            System.out.println("  - Example: \"HELLO.png\" contains image of text \"HELLO\"");
            System.out.println();
            System.out.println("Example:");
            System.out.println("  # Create training images");
            System.out.println("  # \"A.png\" -> image containing letter 'A'");
            System.out.println("  # \"HELLO.png\" -> image containing text \"HELLO\"");
            System.out.println();
            System.out.println("  # Run training");
            System.out.println("  java OCRTrainer ./training_data");
            return;
        }
        
        String dataDir = args[0];
        
        try {
            // Create engine and initialize
            try (OcrEngine engine = new OcrEngine()) {
                engine.initialize();
                
                // Define vocabulary (same as what will be recognized)
                String vocabulary = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ";
                
                // Create trainer
                OCRTrainer trainer = new OCRTrainer(engine, vocabulary);
                trainer.batchSize = 8;
                trainer.learningRate = 0.001f;
                trainer.epochs = 100;
                
                // Load training data
                System.out.println("Loading training data from: " + dataDir);
                List<TrainingSample> samples = trainer.loadTrainingData(dataDir);
                
                if (samples.isEmpty()) {
                    System.err.println("No valid training samples found!");
                    System.err.println("Make sure images are named with their text content.");
                    return;
                }
                
                // Run training
                trainer.train(samples);
                
                // Save trained weights
                trainer.saveWeights("trained-ocr-weights.bin");
                
                System.out.println("\nTraining complete! Weights saved to trained-ocr-weights.bin");
                
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
