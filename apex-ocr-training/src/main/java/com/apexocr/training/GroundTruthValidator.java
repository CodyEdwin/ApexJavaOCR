package com.apexocr.training;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Tool for validating and correcting OCR ground truth labels.
 *
 * Usage: java GroundTruthValidator <input_dir> <output_dir>
 *
 * Workflow:
 * 1. Generate initial labels using DocumentLabelExtractor
 * 2. Review and correct labels with this tool
 * 3. Export corrected labels for training
 *
 * This is a batch validation tool - it presents images one by one
 * and allows quick label verification/correction.
 */
public class GroundTruthValidator {

    private final List<ValidationItem> items;
    private int currentIndex;
    private final Scanner scanner;

    private static class ValidationItem {
        String filename;
        String currentLabel;
        String suggestedLabel;
        boolean verified;

        ValidationItem(String filename, String label) {
            this.filename = filename;
            this.currentLabel = label;
            this.suggestedLabel = label;
            this.verified = false;
        }
    }

    public GroundTruthValidator() {
        this.items = new ArrayList<>();
        this.currentIndex = 0;
        this.scanner = new Scanner(System.in);
    }

    /**
     * Loads images and labels from a directory.
     */
    public void loadFromDirectory(String dirPath) throws IOException {
        Path path = Paths.get(dirPath);

        // Try to load existing manifest
        Path manifestPath = path.resolve("labels_manifest.txt");
        Map<String, String> labelMap = new HashMap<>();

        if (Files.exists(manifestPath)) {
            try (BufferedReader reader = Files.newBufferedReader(manifestPath)) {
                String line;
                while ((line = reader.readLine()) != null) {
                    line = line.trim();
                    if (!line.isEmpty() && !line.startsWith("#")) {
                        String[] parts = line.split("->");
                        if (parts.length == 2) {
                            labelMap.put(parts[0].trim(), parts[1].trim());
                        }
                    }
                }
            }
        }

        // Load all images
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(path, "*.{png,jpg,jpeg,gif,bmp}")) {
            for (Path entry : stream) {
                String filename = entry.getFileName().toString();
                String baseName = filename.substring(0, filename.lastIndexOf('.'));

                // Get label from manifest or use filename
                String label = labelMap.containsKey(filename)
                    ? labelMap.get(filename)
                    : baseName;

                items.add(new ValidationItem(filename, label));
            }
        }

        // Sort items by filename for consistent ordering
        items.sort((a, b) -> a.filename.compareTo(b.filename));

        System.out.println("Loaded " + items.size() + " images for validation");
    }

    /**
     * Validates a single image and returns corrected label.
     */
    public String validateImage(BufferedImage image, String currentLabel) {
        // Display image info
        System.out.println("\n" + "=".repeat(50));
        System.out.println("Image: " + currentLabel);
        System.out.println("Dimensions: " + image.getWidth() + "x" + image.getHeight());
        System.out.println("=".repeat(50));

        // In a real implementation, this would display the image
        // For batch processing, we provide keyboard commands
        System.out.println("\nCommands:");
        System.out.println("  [Enter] - Accept current label");
        System.out.println("  c <new> - Change label to <new>");
        System.out.println("  s       - Skip this image");
        System.out.println("  q       - Quit validation");
        System.out.println("  r       - Regenerate label from filename");

        System.out.print("\nLabel [" + currentLabel + "]: ");
        String input = scanner.nextLine().trim();

        switch (input.toLowerCase()) {
            case "":
                return currentLabel;
            case "s":
                return null; // Skip
            case "q":
                return "QUIT";
            case "r":
                // Regenerate from filename
                String baseName = currentLabel.replaceAll("_processed$", "");
                return baseName.replaceAll("[^A-Za-z0-9\\-_]", "").toUpperCase();
            default:
                if (input.startsWith("c ")) {
                    return input.substring(2).trim().toUpperCase();
                }
                System.out.println("Unknown command: " + input);
                return currentLabel;
        }
    }

    /**
     * Runs batch validation.
     */
    public void runBatchValidation(String outputDir) throws IOException {
        Files.createDirectories(Paths.get(outputDir));

        int validated = 0;
        int skipped = 0;
        int quit = 0;

        for (ValidationItem item : items) {
            String result = validateImage(null, item.currentLabel);

            if (result == null) {
                skipped++;
            } else if ("QUIT".equals(result)) {
                quit = validated;
                break;
            } else {
                item.suggestedLabel = result;
                item.verified = true;
                validated++;
            }

            currentIndex++;
        }

        // Export results
        exportResults(outputDir);

        System.out.println("\n" + "=".repeat(50));
        System.out.println("Validation Complete!");
        System.out.println("  Validated: " + validated);
        System.out.println("  Skipped: " + skipped);
        System.out.println("  Quit at: " + quit);
    }

    /**
     * Exports validation results to various formats.
     */
    public void exportResults(String outputDir) throws IOException {
        Path outPath = Paths.get(outputDir);

        // Export manifest
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(outPath.resolve("validated_labels.txt")))) {
            writer.println("# Validated Ground Truth Labels");
            writer.println("# Format: filename -> label");
            writer.println("# Generated by GroundTruthValidator");
            writer.println();

            for (ValidationItem item : items) {
                if (item.verified) {
                    writer.println(item.filename + " -> " + item.suggestedLabel);
                }
            }
        }

        // Export as CSV for spreadsheet import
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(outPath.resolve("labels.csv")))) {
            writer.println("filename,label,verified");
            for (ValidationItem item : items) {
                String verified = item.verified ? "yes" : "no";
                writer.println(item.filename + "," + item.suggestedLabel + "," + verified);
            }
        }

        // Export just verified labels for training
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(outPath.resolve("training_labels.txt")))) {
            for (ValidationItem item : items) {
                if (item.verified) {
                    writer.println(item.suggestedLabel);
                }
            }
        }

        System.out.println("Results exported to: " + outputDir);
        System.out.println("  - validated_labels.txt (full manifest)");
        System.out.println("  - labels.csv (spreadsheet format)");
        System.out.println("  - training_labels.txt (training ready)");
    }

    /**
     * Auto-validates based on filename patterns.
     */
    public void autoValidate() {
        int autoValidated = 0;

        for (ValidationItem item : items) {
            // Check if current label matches expected pattern
            String label = item.currentLabel;

            // Validate format: alphanumeric with dashes/underscores
            if (label.matches("[A-Za-z0-9\\-_]+") && label.length() >= 3) {
                item.verified = true;
                item.suggestedLabel = label.toUpperCase();
                autoValidated++;
            }
        }

        System.out.println("Auto-validated " + autoValidated + " labels based on format");
    }

    /**
     * Generates a validation report.
     */
    public void generateReport(String outputDir) throws IOException {
        Path outPath = Paths.get(outputDir);

        int verified = 0;
        int total = items.size();
        int shortLabels = 0;
        int longLabels = 0;
        Set<String> labelSet = new HashSet<>();

        for (ValidationItem item : items) {
            if (item.verified) verified++;
            if (item.suggestedLabel.length() < 3) shortLabels++;
            if (item.suggestedLabel.length() > 20) longLabels++;
            labelSet.add(item.suggestedLabel);
        }

        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(outPath.resolve("validation_report.txt")))) {
            writer.println("Ground Truth Validation Report");
            writer.println("==============================");
            writer.println();
            writer.println("Summary:");
            writer.println("  Total images: " + total);
            writer.println("  Verified labels: " + verified);
            writer.println("  Unverified: " + (total - verified));
            writer.println("  Unique labels: " + labelSet.size());
            writer.println();
            writer.println("Label Statistics:");
            writer.println("  Short labels (<3 chars): " + shortLabels);
            writer.println("  Long labels (>20 chars): " + longLabels);
            writer.println();
            writer.println("Label Length Distribution:");
            Map<Integer, Integer> lengthDist = new TreeMap<>();
            for (ValidationItem item : items) {
                int len = item.suggestedLabel.length();
                lengthDist.merge(len, 1, Integer::sum);
            }
            for (Map.Entry<Integer, Integer> entry : lengthDist.entrySet()) {
                writer.printf("  %2d chars: %d labels%n", entry.getKey(), entry.getValue());
            }
            writer.println();
            writer.println("Sample Labels:");
            labelSet.stream().limit(20).sorted().forEach(label ->
                writer.println("  " + label)
            );
        }

        System.out.println("Report saved to: " + outPath.resolve("validation_report.txt"));
    }

    /**
     * Creates training-ready labels by copying and renaming files.
     */
    public void prepareTrainingData(String outputDir, String vocabulary) throws IOException {
        Path outPath = Paths.get(outputDir);
        Files.createDirectories(outPath);

        int prepared = 0;
        int skipped = 0;

        for (ValidationItem item : items) {
            if (!item.verified) {
                skipped++;
                continue;
            }

            // Validate label only contains vocabulary characters
            boolean valid = true;
            for (char c : item.suggestedLabel.toCharArray()) {
                if (vocabulary.indexOf(c) < 0 && c != ' ') {
                    valid = false;
                    break;
                }
            }

            if (!valid) {
                skipped++;
                continue;
            }

            // Copy file with label as filename
            Path sourcePath = Paths.get(item.filename);
            String newName = item.suggestedLabel + ".png";
            Path destPath = outPath.resolve(newName);

            if (Files.exists(sourcePath) && !Files.exists(destPath)) {
                Files.copy(sourcePath, destPath);
            }

            prepared++;
        }

        System.out.println("Training data prepared: " + prepared + " files");
        System.out.println("Skipped (invalid labels): " + skipped);
        System.out.println("Output: " + outputDir);
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Ground Truth Validator for OCR Training");
            System.out.println("========================================");
            System.out.println();
            System.out.println("Usage: java GroundTruthValidator <input_dir> <output_dir>");
            System.out.println();
            System.out.println("Workflow:");
            System.out.println("  1. Run DocumentLabelExtractor to get initial labels");
            System.out.println("  2. Run GroundTruthValidator to review/correct labels");
            System.out.println("  3. Use validated labels for training");
            System.out.println();
            System.out.println("Commands during validation:");
            System.out.println("  [Enter] - Accept current label");
            System.out.println("  c <new> - Change label to <new>");
            System.out.println("  s       - Skip this image");
            System.out.println("  q       - Quit");
            System.out.println("  r       - Regenerate from filename");
            return;
        }

        String inputDir = args[0];
        String outputDir = args[1];

        try {
            GroundTruthValidator validator = new GroundTruthValidator();
            validator.loadFromDirectory(inputDir);

            // Auto-validate well-formatted labels
            validator.autoValidate();

            // Run interactive validation
            validator.runBatchValidation(outputDir);

            // Generate report
            validator.generateReport(outputDir);

        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
