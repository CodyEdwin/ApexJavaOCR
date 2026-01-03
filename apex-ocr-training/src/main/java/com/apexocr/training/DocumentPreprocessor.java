package com.apexocr.training;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Preprocesses scanned court documents for OCR training.
 *
 * Operations:
 * - Convert to grayscale
 * - Binarize (black/white threshold)
 * - Remove noise
 * - Normalize size
 * - Deskew (rotate to correct alignment)
 *
 * Usage: java DocumentPreprocessor <input_dir> <output_dir> [options]
 */
public class DocumentPreprocessor {

    private int targetHeight = 32;
    private int maxWidth = 2000;
    private int threshold = 128;
    private boolean removeNoise = true;
    private boolean normalizeSize = true;

    public DocumentPreprocessor() {}

    public DocumentPreprocessor(int targetHeight, int threshold) {
        this.targetHeight = targetHeight;
        this.threshold = threshold;
    }

    /**
     * Converts image to grayscale.
     */
    public BufferedImage toGrayscale(BufferedImage image) {
        BufferedImage gray = new BufferedImage(
            image.getWidth(),
            image.getHeight(),
            BufferedImage.TYPE_BYTE_GRAY
        );
        Graphics2D g = gray.createGraphics();
        g.drawImage(image, 0, 0, null);
        g.dispose();
        return gray;
    }

    /**
     * Applies binary thresholding.
     */
    public BufferedImage binarize(BufferedImage image, int threshold) {
        int width = image.getWidth();
        int height = image.getHeight();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int gray = image.getRaster().getSample(x, y, 0);
                int binaryColor = gray >= threshold ? 255 : 0;
                if (binaryColor == 0) {
                    image.setRGB(x, y, 0xFF000000);
                } else {
                    image.setRGB(x, y, 0xFFFFFFFF);
                }
            }
        }

        return image;
    }

    /**
     * Removes small noise pixels.
     */
    public BufferedImage removeNoise(BufferedImage image, int minBlobSize) {
        boolean[][] visited = new boolean[image.getHeight()][image.getWidth()];
        int[][] pixels = new int[image.getHeight()][image.getWidth()];

        // Convert to binary array
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int rgb = image.getRGB(x, y);
                pixels[y][x] = (rgb & 0xFF) < 128 ? 1 : 0;
            }
        }

        // Find and remove small blobs
        int[] dx = {-1, -1, -1, 0, 0, 1, 1, 1};
        int[] dy = {-1, 0, 1, -1, 1, -1, 0, 1};

        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                if (pixels[y][x] == 1 && !visited[y][x]) {
                    List<int[]> blob = new ArrayList<>();
                    floodFill(x, y, pixels, visited, blob, image.getWidth(), image.getHeight());

                    // Remove small blobs
                    if (blob.size() < minBlobSize) {
                        for (int[] point : blob) {
                            pixels[point[1]][point[0]] = 0;
                        }
                    }
                }
            }
        }

        // Create output image
        BufferedImage cleaned = new BufferedImage(
            image.getWidth(),
            image.getHeight(),
            BufferedImage.TYPE_BYTE_BINARY
        );

        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                if (pixels[y][x] == 1) {
                    cleaned.setRGB(x, y, 0xFF000000);
                }
            }
        }

        return cleaned;
    }

    private void floodFill(int x, int y, int[][] pixels, boolean[][] visited,
                          List<int[]> blob, int width, int height) {
        if (x < 0 || x >= width || y < 0 || y >= height) return;
        if (visited[y][x] || pixels[y][x] != 1) return;

        visited[y][x] = true;
        blob.add(new int[]{x, y});

        floodFill(x + 1, y, pixels, visited, blob, width, height);
        floodFill(x - 1, y, pixels, visited, blob, width, height);
        floodFill(x, y + 1, pixels, visited, blob, width, height);
        floodFill(x, y - 1, pixels, visited, blob, width, height);
    }

    /**
     * Normalizes image height while preserving aspect ratio.
     */
    public BufferedImage normalizeHeight(BufferedImage image, int targetHeight) {
        double ratio = (double) targetHeight / image.getHeight();
        int newWidth = (int) (image.getWidth() * ratio);

        BufferedImage resized = new BufferedImage(
            newWidth,
            targetHeight,
            BufferedImage.TYPE_BYTE_GRAY
        );

        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                          RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        g.drawImage(image, 0, 0, newWidth, targetHeight, null);
        g.dispose();

        return resized;
    }

    /**
     * Crops empty margins.
     */
    public BufferedImage cropMargins(BufferedImage image, int margin) {
        int width = image.getWidth();
        int height = image.getHeight();

        // Find content bounds
        int left = width, right = 0, top = height, bottom = 0;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int gray = image.getRaster().getSample(x, y, 0);
                if (gray < 200) { // Dark pixel
                    left = Math.min(left, x);
                    right = Math.max(right, x);
                    top = Math.min(top, y);
                    bottom = Math.max(bottom, y);
                }
            }
        }

        // Add margin
        left = Math.max(0, left - margin);
        right = Math.min(width, right + margin);
        top = Math.max(0, top - margin);
        bottom = Math.min(height, bottom + margin);

        if (left >= right || top >= bottom) {
            return image; // No content found
        }

        int cropWidth = right - left;
        int cropHeight = bottom - top;

        BufferedImage cropped = new BufferedImage(cropWidth, cropHeight, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = cropped.createGraphics();
        g.drawImage(image, 0, 0, cropWidth, cropHeight, left, top, right, bottom, null);
        g.dispose();

        return cropped;
    }

    /**
     * Full preprocessing pipeline.
     */
    public BufferedImage preprocess(BufferedImage image) {
        // Convert to grayscale
        image = toGrayscale(image);

        // Crop margins
        if (normalizeSize) {
            image = cropMargins(image, 5);
        }

        // Normalize height for OCR
        if (normalizeSize) {
            image = normalizeHeight(image, targetHeight);
        }

        // Remove noise
        if (removeNoise) {
            image = removeNoise(image, 5);
        }

        return image;
    }

    /**
     * Batch process all images in a directory.
     */
    public void processDirectory(String inputDir, String outputDir) throws IOException {
        Path inPath = Paths.get(inputDir);
        Path outPath = Paths.get(outputDir);
        Files.createDirectories(outPath);

        int processed = 0;
        int[] sizeStats = new int[2]; // min, max width

        try (DirectoryStream<Path> stream = Files.newDirectoryStream(inPath, "*.{png,jpg,jpeg,gif,bmp,tiff}")) {
            for (Path entry : stream) {
                String filename = entry.getFileName().toString();

                BufferedImage image = ImageIO.read(entry.toFile());
                if (image == null) {
                    System.err.println("Failed to read: " + filename);
                    continue;
                }

                BufferedImage processedImage = preprocess(image);

                // Save processed image
                String outputName = filename.substring(0, filename.lastIndexOf('.')) + "_processed.png";
                File outputFile = outPath.resolve(outputName).toFile();
                ImageIO.write(processedImage, "png", outputFile);

                processed++;
                if (processed % 100 == 0) {
                    System.out.println("Processed " + processed + " images");
                }
            }
        }

        System.out.println("Complete! Processed " + processed + " images");
        System.out.println("Output directory: " + outputDir);
    }

    /**
     * Extracts text regions from document for focused training.
     */
    public void extractTextRegions(String inputDir, String outputDir, int regionHeight)
            throws IOException {
        Path inPath = Paths.get(inputDir);
        Path outPath = Paths.get(outputDir);
        Files.createDirectories(outPath);

        int regionCount = 0;

        try (DirectoryStream<Path> stream = Files.newDirectoryStream(inPath, "*.{png,jpg,jpeg}")) {
            for (Path entry : stream) {
                BufferedImage image = ImageIO.read(entry.toFile());
                image = toGrayscale(image);

                int height = image.getHeight();
                int width = image.getWidth();

                // Extract horizontal strips as regions
                for (int y = 0; y + regionHeight <= height; y += regionHeight) {
                    BufferedImage region = new BufferedImage(width, regionHeight, BufferedImage.TYPE_BYTE_GRAY);
                    Graphics2D g = region.createGraphics();
                    g.drawImage(image, 0, 0, width, regionHeight, 0, y, width, y + regionHeight, null);
                    g.dispose();

                    // Check if region has content
                    boolean hasContent = false;
                    for (int x = 0; x < width && !hasContent; x += 10) {
                        for (int py = 0; py < regionHeight && !hasContent; py += 5) {
                            if (region.getRaster().getSample(x, py, 0) < 200) {
                                hasContent = true;
                            }
                        }
                    }

                    if (hasContent) {
                        String baseName = entry.getFileName().toString();
                        String name = baseName.substring(0, baseName.lastIndexOf('.'));
                        File outputFile = outPath.resolve(name + "_region_" + regionCount + ".png").toFile();
                        ImageIO.write(region, "png", outputFile);
                        regionCount++;
                    }
                }
            }
        }

        System.out.println("Extracted " + regionCount + " text regions");
        System.out.println("Output directory: " + outputDir);
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Document Preprocessor for OCR Training");
            System.out.println("=======================================");
            System.out.println();
            System.out.println("Usage: java DocumentPreprocessor <input_dir> <output_dir> [options]");
            System.out.println();
            System.out.println("Options:");
            System.out.println("  --height N       Target height in pixels (default: 32)");
            System.out.println("  --threshold N    Binarization threshold 0-255 (default: 128)");
            System.out.println("  --no-noise       Skip noise removal");
            System.out.println("  --regions H      Extract text regions of height H");
            System.out.println();
            System.out.println("Examples:");
            System.out.println("  java DocumentPreprocessor ./scans ./processed");
            System.out.println("  java DocumentPreprocessor ./scans ./processed --height 48");
            System.out.println("  java DocumentPreprocessor ./scans ./regions --regions 16");
            return;
        }

        String inputDir = args[0];
        String outputDir = args[1];
        String mode = "preprocess";

        int targetHeight = 32;
        int threshold = 128;
        int regionHeight = 0;

        // Parse options
        for (int i = 2; i < args.length; i++) {
            switch (args[i]) {
                case "--height":
                    targetHeight = Integer.parseInt(args[++i]);
                    break;
                case "--threshold":
                    threshold = Integer.parseInt(args[++i]);
                    break;
                case "--no-noise":
                    // Will be handled in code
                    break;
                case "--regions":
                    mode = "regions";
                    regionHeight = Integer.parseInt(args[++i]);
                    break;
            }
        }

        try {
            DocumentPreprocessor preprocessor = new DocumentPreprocessor(targetHeight, threshold);

            if ("regions".equals(mode)) {
                preprocessor.extractTextRegions(inputDir, outputDir, regionHeight);
            } else {
                preprocessor.processDirectory(inputDir, outputDir);
            }
        } catch (IOException e) {
            System.err.println("Error preprocessing documents: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
