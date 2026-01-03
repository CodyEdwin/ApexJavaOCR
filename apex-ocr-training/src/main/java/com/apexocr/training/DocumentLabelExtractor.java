package com.apexocr.training;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

/**
 * Utility to extract case numbers and docket numbers from scanned court documents.
 * Looks for common court case number patterns and extracts them from image regions.
 *
 * Usage: java DocumentLabelExtractor <input_dir> <output_dir> <pattern_type>
 *
 * Pattern types:
 *   STANDARD  - Matches "CV-2024-001234" or "1:24-cv-00123"
 *   SHORT     - Matches "001234", "12345" (numeric only)
 *   ALL       - Tries all patterns
 */
public class DocumentLabelExtractor {

    // Common court case number patterns
    private static final Pattern[] PATTERNS = {
        // Standard patterns like "CV-2024-001234" or "1:24-cv-00123"
        Pattern.compile("\\b\\d{1,2}:\\d{2}-[a-zA-Z]{2}-\\d{5,7}\\b"),
        Pattern.compile("\\b[A-Z]{1,3}-\\d{4}-\\d{5,7}\\b"),
        Pattern.compile("\\b\\d{4}-[A-Z]{2}-\\d{5,7}\\b"),

        // Shorter docket patterns
        Pattern.compile("\\b\\d{5,8}\\b"),
        Pattern.compile("\\b\\d{3,4}-\\d{4,6}\\b"),

        // File numbers
        Pattern.compile("\\b[Ff]ile\\s*[Nn]o\\.?\\s*\\d{5,8}\\b"),
        Pattern.compile("\\b[Dd]ocket\\s*[Nn]o?\\.?\\s*\\d{5,8}\\b"),

        // Case numbers with spaces
        Pattern.compile("\\bCase\\s*[Nn]o\\.?\\s*[A-Z]?\\d{2,4}[-\\s]\\d{4,6}\\b", Pattern.CASE_INSENSITIVE)
    };

    private final Pattern[] patterns;
    private final List<String> foundLabels;

    public DocumentLabelExtractor(String patternType) {
        this.patterns = getPatternsForType(patternType);
        this.foundLabels = new ArrayList<>();
    }

    private Pattern[] getPatternsForType(String type) {
        switch (type.toUpperCase()) {
            case "SHORT":
                return new Pattern[]{
                    PATTERNS[3], // \d{5,8}
                    PATTERNS[4]  // \d{3,4}-\d{4,6}
                };
            case "STANDARD":
                return new Pattern[]{
                    PATTERNS[0], PATTERNS[1], PATTERNS[2],
                    PATTERNS[5], PATTERNS[6]
                };
            case "ALL":
            default:
                return PATTERNS;
        }
    }

    /**
     * Scans a document image for case numbers.
     * For now, uses filename as primary source - OCR integration planned.
     */
    public String extractFromFilename(String filename) {
        String baseName = filename.substring(0, filename.lastIndexOf('.'));

        // Try to clean up filename to extract case number
        // Remove common prefixes like "scan_", "page_", "img_"
        String cleaned = baseName.replaceAll("^(scan_|page_|img_|doc_|image_)\\d*[-_]?", "");

        // Check if cleaned name matches a pattern
        for (Pattern pattern : patterns) {
            Matcher m = pattern.matcher(cleaned);
            if (m.find()) {
                return normalizeLabel(m.group());
            }
        }

        // Return cleaned filename if no pattern found
        // Remove special characters for safety
        return cleaned.replaceAll("[^A-Za-z0-9\\-_]", "").toUpperCase();
    }

    /**
     * Normalizes extracted label for consistency.
     */
    private String normalizeLabel(String label) {
        // Convert to uppercase, remove spaces
        return label.toUpperCase().replaceAll("\\s+", "");
    }

    /**
     * Batch process a directory of images.
     */
    public void processDirectory(String inputDir, String outputDir) throws IOException {
        Path inPath = Paths.get(inputDir);
        Path outPath = Paths.get(outputDir);
        Files.createDirectories(outPath);

        int processed = 0;
        int withLabels = 0;

        try (DirectoryStream<Path> stream = Files.newDirectoryStream(inPath, "*.{png,jpg,jpeg,gif,bmp,tiff}")) {
            for (Path entry : stream) {
                String filename = entry.getFileName().toString();
                String label = extractFromFilename(filename);

                // Copy file with new name
                Path destFile = outPath.resolve(label + getExtension(filename));
                Files.copy(entry, destFile, StandardCopyOption.REPLACE_EXISTING);

                if (!label.isEmpty() && !label.equals(filename.substring(0, filename.lastIndexOf('.')))) {
                    withLabels++;
                }

                processed++;
                if (processed % 100 == 0) {
                    System.out.println("Processed " + processed + " files, " + withLabels + " with extracted labels");
                }

                foundLabels.add(label);
            }
        }

        System.out.println("Complete! Processed " + processed + " files");
        System.out.println("Files with extracted labels: " + withLabels);
        System.out.println("Output directory: " + outputDir);
    }

    /**
     * Generate a manifest file with all extracted labels.
     */
    public void generateManifest(String outputDir) throws IOException {
        Path manifestPath = Paths.get(outputDir, "labels_manifest.txt");

        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(manifestPath))) {
            writer.println("# Court Document Label Manifest");
            writer.println("# Generated by DocumentLabelExtractor");
            writer.println("# Format: filename -> extracted_label");
            writer.println();

            for (String label : foundLabels) {
                writer.println(label);
            }
        }

        System.out.println("Manifest saved to: " + manifestPath);
    }

    private String getExtension(String filename) {
        int dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(dot) : ".png";
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Court Document Label Extractor");
            System.out.println("================================");
            System.out.println();
            System.out.println("Usage: java DocumentLabelExtractor <input_dir> <output_dir> [pattern_type]");
            System.out.println();
            System.out.println("Arguments:");
            System.out.println("  input_dir     - Directory containing document images");
            System.out.println("  output_dir    - Directory for renamed output images");
            System.out.println("  pattern_type  - Pattern type: STANDARD, SHORT, or ALL (default: ALL)");
            System.out.println();
            System.out.println("Examples:");
            System.out.println("  java DocumentLabelExtractor ./scans ./labeled STANDARD");
            System.out.println("  java DocumentLabelExtractor ./court_docs ./output SHORT");
            System.out.println();
            System.out.println("Note: This tool extracts labels from filenames.");
            System.out.println("For text extraction from images, OCR integration is planned.");
            return;
        }

        String inputDir = args[0];
        String outputDir = args[1];
        String patternType = args.length > 2 ? args[2] : "ALL";

        try {
            DocumentLabelExtractor extractor = new DocumentLabelExtractor(patternType);
            extractor.processDirectory(inputDir, outputDir);
            extractor.generateManifest(outputDir);
        } catch (IOException e) {
            System.err.println("Error processing documents: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
