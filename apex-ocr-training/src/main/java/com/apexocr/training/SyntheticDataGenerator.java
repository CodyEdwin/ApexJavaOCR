package com.apexocr.training;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Synthetic Training Data Generator
 * Creates training images with known text labels.
 * 
 * Usage: java SyntheticDataGenerator <output_dir> <num_samples>
 * 
 * Creates images named after their text content.
 * Example: "HELLO.png" contains an image of the text "HELLO"
 */
public class SyntheticDataGenerator {
    
    private static final String UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    private static final String LOWERCASE = "abcdefghijklmnopqrstuvwxyz";
    private static final String DIGITS = "0123456789";
    private static final String SPECIAL = "!?.,;:";
    
    private final Random random;
    private final Font font;
    private final int imageHeight;
    private final int minWidth;
    private final int maxWidth;
    
    public SyntheticDataGenerator(int imageHeight, int minWidth, int maxWidth) {
        this.random = new Random(42);
        this.imageHeight = imageHeight;
        this.minWidth = minWidth;
        this.maxWidth = maxWidth;
        
        // Create a readable font
        this.font = new Font("Arial", Font.BOLD, imageHeight - 4);
    }
    
    /**
     * Generates a random text string of specified length.
     */
    public String generateRandomText(int minLen, int maxLen) {
        int length = random.nextInt(maxLen - minLen + 1) + minLen;
        StringBuilder sb = new StringBuilder();
        
        String chars = UPPERCASE + DIGITS;
        for (int i = 0; i < length; i++) {
            sb.append(chars.charAt(random.nextInt(chars.length())));
        }
        return sb.toString();
    }
    
    /**
     * Creates a synthetic text image.
     */
    public BufferedImage createTextImage(String text) {
        // Create a temporary image to get proper FontMetrics
        BufferedImage tempImage = new BufferedImage(1, 1, BufferedImage.TYPE_INT_ARGB);
        Graphics2D tempG = tempImage.createGraphics();
        tempG.setFont(font);
        FontMetrics fm = tempG.getFontMetrics();
        tempG.dispose();

        int charWidth = fm.charWidth('X');
        int textWidth = text.length() * charWidth + 20;
        textWidth = Math.max(textWidth, minWidth);
        textWidth = Math.min(textWidth, maxWidth);

        // Create image
        BufferedImage image = new BufferedImage(textWidth, imageHeight, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = image.createGraphics();

        // Fill with varied background
        int bgValue = 200 + random.nextInt(55); // Light gray
        g.setColor(new Color(bgValue, bgValue, bgValue));
        g.fillRect(0, 0, textWidth, imageHeight);

        // Add some noise
        for (int i = 0; i < textWidth * imageHeight / 10; i++) {
            int x = random.nextInt(textWidth);
            int y = random.nextInt(imageHeight);
            int noise = random.nextInt(40) - 20;
            int pixel = bgValue + noise;
            pixel = Math.max(0, Math.min(255, pixel));
            int gray = (pixel << 16) | (pixel << 8) | pixel;
            image.setRGB(x, y, gray);
        }
        
        // Draw text
        g.setFont(font);
        int textColor = random.nextInt(50); // Dark text
        g.setColor(new Color(textColor, textColor, textColor));
        int x = 10;
        int y = imageHeight - 6;
        g.drawString(text, x, y);
        
        g.dispose();
        
        return image;
    }
    
    /**
     * Generates and saves a batch of training images.
     */
    public void generateBatch(String outputDir, int count, int minLen, int maxLen) throws IOException {
        Path dirPath = Paths.get(outputDir);
        Files.createDirectories(dirPath);
        
        System.out.println("Generating " + count + " training samples to: " + outputDir);
        
        for (int i = 0; i < count; i++) {
            // Generate random text
            String text = generateRandomText(minLen, maxLen);
            
            // Create image
            BufferedImage image = createTextImage(text);
            
            // Save with text as filename
            String filename = text + ".png";
            File outputFile = dirPath.resolve(filename).toFile();
            ImageIO.write(image, "png", outputFile);
            
            if ((i + 1) % 100 == 0) {
                System.out.println("  Generated " + (i + 1) + "/" + count + " images");
            }
        }
        
        System.out.println("Done! Created " + count + " training samples.");
    }
    
    /**
     * Main method for standalone execution.
     */
    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Synthetic Training Data Generator");
            System.out.println("==================================");
            System.out.println();
            System.out.println("Usage: java SyntheticDataGenerator <output_dir> <num_samples> [min_len] [max_len]");
            System.out.println();
            System.out.println("Arguments:");
            System.out.println("  output_dir   - Directory to save training images");
            System.out.println("  num_samples  - Number of images to generate");
            System.out.println("  min_len      - Minimum text length (default: 3)");
            System.out.println("  max_len      - Maximum text length (default: 8)");
            System.out.println();
            System.out.println("Output:");
            System.out.println("  Images named after their text content");
            System.out.println("  Example: \"HELLO.png\" contains text \"HELLO\"");
            System.out.println();
            System.out.println("Example:");
            System.out.println("  java SyntheticDataGenerator ./training_data 1000 4 6");
            System.out.println("  # Creates 1000 images with 4-6 character alphanumeric text");
            return;
        }
        
        String outputDir = args[0];
        int count = Integer.parseInt(args[1]);
        int minLen = args.length > 2 ? Integer.parseInt(args[2]) : 3;
        int maxLen = args.length > 3 ? Integer.parseInt(args[3]) : 8;
        
        try {
            SyntheticDataGenerator generator = new SyntheticDataGenerator(32, 100, 400);
            generator.generateBatch(outputDir, count, minLen, maxLen);
        } catch (IOException e) {
            System.err.println("Error generating data: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
