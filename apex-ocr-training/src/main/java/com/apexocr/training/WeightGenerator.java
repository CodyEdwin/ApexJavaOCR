package com.apexocr.training;

import com.apexocr.core.tensor.Tensor;
import com.apexocr.core.tensor.TensorOperations;
import com.apexocr.core.neural.Layer;
import com.apexocr.core.neural.Conv2D;
import com.apexocr.core.neural.Dense;
import com.apexocr.core.neural.BiLSTM;
import com.apexocr.engine.OcrEngine;
import com.apexocr.engine.OcrResult;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;

/**
 * Weight Generator - Creates properly initialized weights for the OCR network.
 * Uses He initialization for ReLU layers, Xavier for recurrent layers.
 */
public class WeightGenerator {
    
    private final OcrEngine engine;
    private final Random random;
    
    public WeightGenerator(OcrEngine engine) {
        this.engine = engine;
        this.random = new Random(42); // Fixed seed for reproducibility
    }
    
    /**
     * Generates properly initialized weights for all layers.
     */
    public void generateAllWeights() {
        System.out.println("Generating initialized weights for OCR network...");
        
        int layerNum = 0;
        for (Layer layer : engine.getNetwork()) {
            layerNum++;
            if (layer instanceof Conv2D) {
                initializeConv2D((Conv2D) layer, layerNum);
            } else if (layer instanceof Dense) {
                initializeDense((Dense) layer, layerNum);
            } else if (layer instanceof BiLSTM) {
                initializeBiLSTM((BiLSTM) layer, layerNum);
            }
        }
        
        System.out.println("Weight initialization complete!");
    }
    
    /**
     * Initialize Conv2D layer with He normal initialization.
     */
    private void initializeConv2D(Conv2D layer, int layerNum) {
        Tensor weights = layer.getWeights();
        Tensor bias = layer.getBiases();
        
        if (weights != null && weights.getSize() > 0) {
            long[] shape = weights.getShape();
            // shape is [filters, channels, height, width]
            long fanIn = shape[1] * shape[2] * shape[3];
            float std = (float) Math.sqrt(2.0 / fanIn);
            
            // Use linear indexing for the weights tensor
            for (long i = 0; i < weights.getSize(); i++) {
                double u1 = random.nextDouble();
                double u2 = random.nextDouble();
                double z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
                weights.setFloat(i, (float) (z * std));
            }
            System.out.println("  Initialized Conv2D: " + layer.getName() + " [" + Arrays.toString(shape) + "]");
            
            // Initialize bias
            if (bias != null && bias.getSize() > 0) {
                for (long i = 0; i < bias.getSize(); i++) {
                    bias.setFloat(i, 0.1f); // Small positive bias
                }
            }
        }
    }
    
    /**
     * Initialize Dense layer with He normal initialization.
     */
    private void initializeDense(Dense layer, int layerNum) {
        Tensor weights = layer.getWeights();
        Tensor bias = layer.getBiases();
        
        if (weights != null && weights.getSize() > 0) {
            long[] shape = weights.getShape();
            // shape is [input, output]
            long fanIn = shape[0];
            float std = (float) Math.sqrt(2.0 / fanIn);
            
            for (long i = 0; i < weights.getSize(); i++) {
                double u1 = random.nextDouble();
                double u2 = random.nextDouble();
                double z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
                weights.setFloat(i, (float) (z * std * 0.5)); // Smaller scale for Dense layers
            }
            System.out.println("  Initialized Dense: " + layer.getName() + " [" + Arrays.toString(shape) + "]");
            
            // Initialize bias
            if (bias != null && bias.getSize() > 0) {
                for (long i = 0; i < bias.getSize(); i++) {
                    bias.setFloat(i, 0.1f);
                }
            }
        }
    }
    
    /**
     * Initialize BiLSTM layer with Xavier initialization.
     */
    private void initializeBiLSTM(BiLSTM layer, int layerNum) {
        Tensor weights = layer.getWeights();
        Tensor bias = layer.getBiases();
        
        if (weights != null && weights.getSize() > 0) {
            long[] shape = weights.getShape();
            // Xavier initialization for RNN
            long fanIn = shape[0];
            long fanOut = shape[1];
            float std = (float) Math.sqrt(2.0 / (fanIn + fanOut));
            
            for (long i = 0; i < weights.getSize(); i++) {
                double u1 = random.nextDouble();
                double u2 = random.nextDouble();
                double z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
                weights.setFloat(i, (float) (z * std * 0.5));
            }
            System.out.println("  Initialized BiLSTM: " + layer.getName() + " [" + Arrays.toString(shape) + "]");
            
            // Initialize bias
            if (bias != null && bias.getSize() > 0) {
                // LSTM bias: small positive for forget gate bias
                for (long i = 0; i < bias.getSize(); i++) {
                    bias.setFloat(i, 1.0f); // Slightly higher initial bias for LSTM
                }
            }
        }
    }
    
    /**
     * Saves weights using the engine's save method.
     */
    public void saveWeights(String filePath) {
        engine.saveWeights(filePath);
        System.out.println("Weights saved to: " + filePath);
    }
    
    /**
     * Loads weights using the engine's load method.
     */
    public boolean loadWeights(String filePath) {
        return engine.loadWeights(filePath);
    }
    
    /**
     * Main method to generate and save weights.
     */
    public static void main(String[] args) {
        try (OcrEngine engine = new OcrEngine()) {
            engine.initialize();
            
            // Run warm-up to initialize shapes and weights
            BufferedImage testImage = new BufferedImage(128, 32, BufferedImage.TYPE_BYTE_GRAY);
            Graphics2D g = testImage.createGraphics();
            g.setColor(Color.WHITE);
            g.fillRect(0, 0, 128, 32);
            g.setColor(Color.BLACK);
            g.drawString("TEST", 10, 25);
            g.dispose();
            
            System.out.println("Running warm-up to initialize network shapes...");
            OcrResult warmup = engine.process(testImage);
            System.out.println("Warm-up complete: " + warmup.getText());
            
            // Generate properly initialized weights
            WeightGenerator generator = new WeightGenerator(engine);
            generator.generateAllWeights();
            
            // Save weights
            String weightFile = "apex-ocr-weights.bin";
            generator.saveWeights(weightFile);
            System.out.println("\nWeights saved to: " + weightFile);
            
            // Verify by running inference
            System.out.println("\nTesting with generated weights...");
            OcrResult result = engine.process(testImage);
            System.out.println("Test result: \"" + result.getText() + "\" (confidence: " + 
                String.format("%.2f%%", result.getConfidence() * 100) + ")");
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
