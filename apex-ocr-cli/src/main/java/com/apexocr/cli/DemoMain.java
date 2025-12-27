package com.apexocr.cli;

import com.apexocr.engine.OcrEngine;
import com.apexocr.engine.OcrResult;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

/**
 * Demo utility for testing the OCR engine with pre-trained weights.
 * Automatically loads EasyOCR pre-trained weights for English text recognition.
 */
public class DemoMain {
    
    /**
     * Pre-trained EasyOCR weights file name.
     */
    private static final String PRETRAINED_WEIGHTS = "easyocr-english-weights.bin";
    
    /**
     * Demo weights file name (fallback).
     */
    private static final String DEMO_WEIGHTS = "demo-weights.bin";
    
    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Usage: java DemoMain <image_path> [weights_path]");
            System.out.println("  image_path        - Path to the input image (required)");
            System.out.println("  weights_path      - Optional path to custom weights file");
            System.out.println("");
            System.out.println("The engine will automatically try to load:");
            System.out.println("  1. easyocr-english-weights.bin (pre-trained English OCR)");
            System.out.println("  2. demo-weights.bin (fallback deterministic weights)");
            return;
        }
        
        String imagePath = args[0];
        String customWeightsPath = args.length > 1 ? args[1] : null;
        
        try {
            System.out.println("ApexOCR Demo - English Text Recognition");
            System.out.println("========================================");
            
            // Load the image first to initialize the network
            BufferedImage image = ImageIO.read(new File(imagePath));
            if (image == null) {
                System.err.println("Failed to load image: " + imagePath);
                return;
            }
            
            System.out.println("Loaded image: " + image.getWidth() + "x" + image.getHeight());
            
            // Create engine and initialize
            try (OcrEngine engine = new OcrEngine()) {
                engine.initialize();
                
                System.out.println("\nNetwork Architecture:");
                System.out.println(engine.getArchitectureSummary());
                
                // Determine which weights to load
                String weightsToLoad = null;
                boolean usingPretrained = false;
                
                if (customWeightsPath != null && new File(customWeightsPath).exists()) {
                    // Use custom weights if provided and exists
                    weightsToLoad = customWeightsPath;
                    System.out.println("\nUsing custom weights: " + weightsToLoad);
                } else if (new File(PRETRAINED_WEIGHTS).exists()) {
                    // Try pre-trained EasyOCR weights
                    weightsToLoad = PRETRAINED_WEIGHTS;
                    usingPretrained = true;
                    System.out.println("\nLoading pre-trained EasyOCR English weights...");
                } else if (new File(DEMO_WEIGHTS).exists()) {
                    // Fall back to demo weights
                    weightsToLoad = DEMO_WEIGHTS;
                    System.out.println("\nLoading demo weights...");
                } else {
                    // Generate demo weights
                    System.out.println("\nNo weights file found, generating demo weights...");
                    engine.generateDemoWeights();
                }
                
                // Load weights if we found a file
                if (weightsToLoad != null) {
                    boolean loaded = engine.loadWeights(weightsToLoad);
                    if (loaded) {
                        if (usingPretrained) {
                            System.out.println("Pre-trained weights loaded successfully!");
                            System.out.println("Ready for English text recognition on scanned documents.");
                        } else {
                            System.out.println("Weights loaded successfully!");
                        }
                    } else {
                        System.err.println("Failed to load weights from " + weightsToLoad);
                        System.out.println("Generating demo weights instead...");
                        engine.generateDemoWeights();
                    }
                }
                
                // Run actual OCR
                System.out.println("\nRunning OCR inference...");
                OcrResult result = engine.process(image);
                
                System.out.println("\nResults:");
                System.out.println("  Recognized text: \"" + result.getText() + "\"");
                System.out.println("  Confidence: " + String.format("%.2f%%", result.getConfidence() * 100));
                System.out.println("  Processing time: " + result.getProcessingTimeMs() + "ms");
            }
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
