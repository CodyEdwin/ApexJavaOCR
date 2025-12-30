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
    
    /**
     * Looks for a weights file in multiple locations.
     * Checks: current directory, parent directory (project root)
     * 
     * @param fileName The weights file name
     * @return Full path to the weights file, or null if not found
     */
    private static String findWeightsFile(String fileName) {
        // Check current directory
        File currentDir = new File(fileName);
        if (currentDir.exists()) {
            return currentDir.getAbsolutePath();
        }
        
        // Check parent directory (project root)
        File parentDir = new File(".." + File.separator + fileName);
        if (parentDir.exists()) {
            return parentDir.getAbsolutePath();
        }
        
        // Check common project locations
        String[] possiblePaths = {
            fileName,
            ".." + File.separator + fileName,
            "." + File.separator + fileName,
            System.getProperty("user.dir") + File.separator + fileName,
            System.getProperty("user.dir") + File.separator + ".." + File.separator + fileName
        };
        
        for (String path : possiblePaths) {
            File f = new File(path);
            if (f.exists()) {
                return f.getAbsolutePath();
            }
        }
        
        return null;
    }
    
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
                
                // Run a warm-up inference to trigger lazy initialization of all weights
                // This is necessary to set proper input/output shapes in all layers
                System.out.println("\nRunning warm-up inference to initialize network shapes...");
                OcrResult warmup = engine.process(image);
                System.out.println("Warm-up result: \"" + warmup.getText() + "\" (" + warmup.getProcessingTimeMs() + "ms)");
                
                // Now show the initialized architecture
                System.out.println("\nInitialized Network:");
                System.out.println(engine.getArchitectureSummary());
                
                // Determine which weights to load
                String weightsToLoad = null;
                boolean usingPretrained = false;
                
                if (customWeightsPath != null && new File(customWeightsPath).exists()) {
                    // Use custom weights if provided and exists
                    weightsToLoad = customWeightsPath;
                    System.out.println("\nUsing custom weights: " + weightsToLoad);
                } else {
                    // Try pre-trained EasyOCR weights
                    String pretrainedPath = findWeightsFile(PRETRAINED_WEIGHTS);
                    if (pretrainedPath != null) {
                        weightsToLoad = pretrainedPath;
                        usingPretrained = true;
                        System.out.println("\nLoading pre-trained EasyOCR English weights from: " + weightsToLoad);
                    } else {
                        // Try demo weights
                        String demoPath = findWeightsFile(DEMO_WEIGHTS);
                        if (demoPath != null) {
                            weightsToLoad = demoPath;
                            System.out.println("\nLoading demo weights from: " + weightsToLoad);
                        } else {
                            // Generate demo weights
                            System.out.println("\nNo weights file found, generating demo weights...");
                            engine.generateDemoWeights();
                        }
                    }
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
