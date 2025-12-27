package com.apexocr.cli;

import com.apexocr.engine.OcrEngine;
import com.apexocr.engine.OcrResult;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

/**
 * Demo utility for testing the OCR engine with deterministic weights.
 */
public class DemoMain {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Usage: java DemoMain <image_path> [weights_output_path]");
            System.out.println("  image_path        - Path to the input image");
            System.out.println("  weights_output_path - Optional path to save demo weights");
            return;
        }
        
        String imagePath = args[0];
        String weightsPath = args.length > 1 ? args[1] : "demo-weights.bin";
        
        try {
            System.out.println("ApexOCR Demo - Testing with deterministic demo weights");
            System.out.println("======================================================");
            
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
                System.out.println("\nRunning warm-up inference to initialize network weights...");
                OcrResult warmup = engine.process(image);
                System.out.println("Warm-up result: \"" + warmup.getText() + "\" (" + warmup.getProcessingTimeMs() + "ms)");
                
                // Now show the initialized architecture
                System.out.println("\nInitialized Network:");
                System.out.println(engine.getArchitectureSummary());
                
                // Try to load weights from file if it exists, otherwise generate demo weights
                if (new File(weightsPath).exists()) {
                    System.out.println("\nLoading weights from " + weightsPath + "...");
                    boolean loaded = engine.loadWeights(weightsPath);
                    if (loaded) {
                        System.out.println("Weights loaded successfully!");
                    } else {
                        System.out.println("Failed to load weights, generating demo weights instead...");
                        engine.generateDemoWeights();
                    }
                } else {
                    // Generate deterministic demo weights
                    System.out.println("\nGenerating demo weights...");
                    engine.generateDemoWeights();
                }
                
                // Save demo weights if requested
                if (weightsPath != null) {
                    engine.saveWeights(weightsPath);
                }
                
                // Run actual OCR
                System.out.println("\nRunning OCR inference with demo weights...");
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
