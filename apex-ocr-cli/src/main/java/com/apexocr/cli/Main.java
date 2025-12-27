package com.apexocr.cli;

import com.apexocr.engine.OcrEngine;
import com.apexocr.engine.OcrResult;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * ApexJavaOCR CLI - Command-line interface for the ApexOCR engine.
 * Provides various commands for image text recognition, batch processing,
 * and engine information.
 *
 * This CLI supports multiple input formats, batch processing, and various
 * output options for flexible integration into workflows.
 *
 * @author ApexOCR Team
 * @version 1.0.0
 */
public class Main implements Callable<Integer> {
    private String inputPath;
    private String outputPath;
    private String format = "text";
    private boolean verbose = false;
    private boolean listFiles = false;
    private int batchSize = 1;
    private String config = "default";

    // Command constants
    private static final String CMD_VERSION = "version";
    private static final String CMD_INFO = "info";
    private static final String CMD_PROCESS = "process";
    private static final String CMD_BATCH = "batch";
    private static final String CMD_HELP = "help";

    @Override
    public Integer call() throws Exception {
        try {
            if (listFiles) {
                return listInputFiles();
            }

            if (inputPath == null) {
                printUsage();
                return 1;
            }

            File inputFile = new File(inputPath);
            if (!inputFile.exists()) {
                System.err.println("Error: Input file or directory not found: " + inputPath);
                return 1;
            }

            OcrEngine.EngineConfig engineConfig = createConfig();

            try (OcrEngine engine = new OcrEngine(engineConfig)) {
                engine.initialize();

                if (verbose) {
                    printEngineInfo(engine);
                }

                if (inputFile.isDirectory()) {
                    return processDirectory(engine, inputFile);
                } else {
                    return processFile(engine, inputFile);
                }
            }
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            if (verbose) {
                e.printStackTrace();
            }
            return 1;
        }
    }

    /**
     * Creates the engine configuration based on command-line options.
     */
    private OcrEngine.EngineConfig createConfig() {
        OcrEngine.EngineConfig config = new OcrEngine.EngineConfig();

        switch (this.config) {
            case "accuracy":
                return OcrEngine.EngineConfig.forHighAccuracy();
            case "speed":
                return OcrEngine.EngineConfig.forSpeed();
            default:
                return config;
        }
    }

    /**
     * Processes a single image file.
     */
    private int processFile(OcrEngine engine, File file) throws IOException {
        long startTime = System.currentTimeMillis();
        OcrResult result = engine.processFile(file.getAbsolutePath());
        long elapsed = System.currentTimeMillis() - startTime;

        String output;
        switch (format.toLowerCase()) {
            case "json":
                output = toJson(result, file.getName());
                break;
            case "xml":
                output = toXml(result, file.getName());
                break;
            case "verbose":
                output = toVerbose(result, file.getName(), elapsed);
                break;
            default:
                output = result.getText();
        }

        if (outputPath != null) {
            Files.write(Paths.get(outputPath), output.getBytes());
            if (verbose) {
                System.out.println("Output written to: " + outputPath);
            }
        } else {
            System.out.println(output);
        }

        return 0;
    }

    /**
     * Processes all images in a directory.
     */
    private int processDirectory(OcrEngine engine, File directory) throws IOException {
        List<File> imageFiles = findImageFiles(directory);

        if (imageFiles.isEmpty()) {
            System.err.println("No image files found in directory: " + directory);
            return 1;
        }

        if (verbose) {
            System.out.println("Found " + imageFiles.size() + " image files");
        }

        AtomicInteger successCount = new AtomicInteger(0);
        AtomicInteger failCount = new AtomicInteger(0);
        StringBuilder allResults = new StringBuilder();

        ExecutorService executor = Executors.newFixedThreadPool(
            Math.min(batchSize, Runtime.getRuntime().availableProcessors())
        );

        try {
            for (File file : imageFiles) {
                executor.submit(() -> {
                    try {
                        OcrResult result = engine.processFile(file.getAbsolutePath());
                        successCount.incrementAndGet();

                        String output;
                        switch (format.toLowerCase()) {
                            case "json":
                                output = toJson(result, file.getName()) + "\n";
                                break;
                            case "xml":
                                output = toXml(result, file.getName()) + "\n";
                                break;
                            default:
                                output = file.getName() + ": " + result.getText() + "\n";
                        }

                        synchronized (allResults) {
                            allResults.append(output);
                        }
                    } catch (Exception e) {
                        failCount.incrementAndGet();
                        if (verbose) {
                            System.err.println("Failed to process: " + file.getName());
                        }
                    }
                });
            }
        } finally {
            executor.shutdown();
            try {
                executor.awaitTermination(5, TimeUnit.MINUTES);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        if (outputPath != null) {
            Files.write(Paths.get(outputPath), allResults.toString().getBytes());
            if (verbose) {
                System.out.println("Results written to: " + outputPath);
            }
        } else {
            System.out.print(allResults.toString());
        }

        if (verbose) {
            System.out.println("\nProcessing complete:");
            System.out.println("  Successful: " + successCount.get());
            System.out.println("  Failed: " + failCount.get());
            System.out.println("  Total: " + imageFiles.size());
        }

        return failCount.get() > 0 ? 1 : 0;
    }

    /**
     * Lists input files without processing.
     */
    private int listInputFiles() throws IOException {
        File inputFile = new File(inputPath);
        if (!inputFile.exists()) {
            System.err.println("Path not found: " + inputPath);
            return 1;
        }

        if (inputFile.isDirectory()) {
            List<File> files = findImageFiles(inputFile);
            for (File f : files) {
                System.out.println(f.getAbsolutePath());
            }
            System.out.println("\nTotal: " + files.size() + " files");
        } else {
            System.out.println(inputFile.getAbsolutePath());
        }

        return 0;
    }

    /**
     * Finds all image files in a directory.
     */
    private List<File> findImageFiles(File directory) throws IOException {
        List<File> files = new ArrayList<>();
        String[] extensions = {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif"};

        for (String ext : extensions) {
            File[] found = directory.listFiles((dir, name) ->
                name.toLowerCase().endsWith("." + ext)
            );
            if (found != null) {
                for (File f : found) {
                    if (f.isFile()) {
                        files.add(f);
                    }
                }
            }
        }

        return files;
    }

    /**
     * Prints engine information.
     */
    private void printEngineInfo(OcrEngine engine) {
        System.out.println("ApexJavaOCR Engine Information");
        System.out.println("================================");
        System.out.println("Initialized: " + engine.isInitialized());
        System.out.println("Layers: " + engine.getLayerCount());
        System.out.println("Parameters: " + String.format("%,d", engine.getParameterCount()));
        System.out.println();

        String summary = engine.getArchitectureSummary();
        System.out.println(summary);
        System.out.println();
    }

    /**
     * Converts result to JSON format.
     */
    private String toJson(OcrResult result, String filename) {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        sb.append("\"file\": \"").append(escapeJson(filename)).append("\",");
        sb.append("\"text\": \"").append(escapeJson(result.getText())).append("\",");
        sb.append("\"confidence\": ").append(result.getConfidence()).append(",");
        sb.append("\"confidence_percent\": ").append(String.format("%.2f", result.getConfidencePercent())).append(",");
        sb.append("\"word_count\": ").append(result.getWordCount()).append(",");
        sb.append("\"char_count\": ").append(result.getCharacterCount()).append(",");
        sb.append("\"processing_time_ms\": ").append(result.getProcessingTimeMs());
        sb.append("}");
        return sb.toString();
    }

    /**
     * Converts result to XML format.
     */
    private String toXml(OcrResult result, String filename) {
        StringBuilder sb = new StringBuilder();
        sb.append("<ocr_result>\n");
        sb.append("  <file>").append(escapeXml(filename)).append("</file>\n");
        sb.append("  <text>").append(escapeXml(result.getText())).append("</text>\n");
        sb.append("  <confidence>").append(result.getConfidence()).append("</confidence>\n");
        sb.append("  <confidence_percent>").append(String.format("%.2f", result.getConfidencePercent())).append("</confidence_percent>\n");
        sb.append("  <word_count>").append(result.getWordCount()).append("</word_count>\n");
        sb.append("  <char_count>").append(result.getCharacterCount()).append("</char_count>\n");
        sb.append("  <processing_time_ms>").append(result.getProcessingTimeMs()).append("</processing_time_ms>\n");
        sb.append("</ocr_result>");
        return sb.toString();
    }

    /**
     * Converts result to verbose text format.
     */
    private String toVerbose(OcrResult result, String filename, long elapsed) {
        StringBuilder sb = new StringBuilder();
        sb.append("File: ").append(filename).append("\n");
        sb.append("Text: \"").append(result.getText()).append("\"\n");
        sb.append("Confidence: ").append(String.format("%.2f%%", result.getConfidencePercent())).append("\n");
        sb.append("Words: ").append(result.getWordCount()).append("\n");
        sb.append("Characters: ").append(result.getCharacterCount()).append("\n");
        sb.append("Processing Time: ").append(elapsed).append("ms\n");
        return sb.toString();
    }

    /**
     * Escapes special characters for JSON.
     */
    private String escapeJson(String str) {
        if (str == null) return "";
        return str.replace("\\", "\\\\")
                  .replace("\"", "\\\"")
                  .replace("\n", "\\n")
                  .replace("\r", "\\r")
                  .replace("\t", "\\t");
    }

    /**
     * Escapes special characters for XML.
     */
    private String escapeXml(String str) {
        if (str == null) return "";
        return str.replace("&", "&amp;")
                  .replace("<", "&lt;")
                  .replace(">", "&gt;")
                  .replace("\"", "&quot;")
                  .replace("'", "&apos;");
    }

    /**
     * Prints usage information.
     */
    private void printUsage() {
        System.out.println("ApexJavaOCR - High-Performance Java OCR Engine");
        System.out.println("===============================================");
        System.out.println();
        System.out.println("Usage: apex-ocr [options] <input>");
        System.out.println();
        System.out.println("Options:");
        System.out.println("  -i, --input <path>     Input file or directory");
        System.out.println("  -o, --output <path>    Output file (default: stdout)");
        System.out.println("  -f, --format <format>  Output format: text, json, xml, verbose");
        System.out.println("  -c, --config <type>    Engine config: default, accuracy, speed");
        System.out.println("  -b, --batch <size>     Batch processing size");
        System.out.println("  -l, --list             List input files only");
        System.out.println("  -v, --verbose          Verbose output");
        System.out.println("  -h, --help             Show this help");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  apex-ocr -i image.png");
        System.out.println("  apex-ocr -i photo.jpg -f json -o result.json");
        System.out.println("  apex-ocr -i ./images/ -o all.txt");
        System.out.println("  apex-ocr -i image.png -c accuracy");
    }

    /**
     * Main entry point.
     */
    public static void main(String[] args) {
        Main main = new Main();

        // Parse command-line arguments
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];

            switch (arg) {
                case "-h":
                case "--help":
                    main.printUsage();
                    return;

                case "-i":
                case "--input":
                    if (i + 1 < args.length) {
                        main.inputPath = args[++i];
                    }
                    break;

                case "-o":
                case "--output":
                    if (i + 1 < args.length) {
                        main.outputPath = args[++i];
                    }
                    break;

                case "-f":
                case "--format":
                    if (i + 1 < args.length) {
                        main.format = args[++i];
                    }
                    break;

                case "-c":
                case "--config":
                    if (i + 1 < args.length) {
                        main.config = args[++i];
                    }
                    break;

                case "-b":
                case "--batch":
                    if (i + 1 < args.length) {
                        try {
                            main.batchSize = Integer.parseInt(args[++i]);
                        } catch (NumberFormatException e) {
                            System.err.println("Invalid batch size: " + args[i]);
                        }
                    }
                    break;

                case "-l":
                case "--list":
                    main.listFiles = true;
                    break;

                case "-v":
                case "--verbose":
                    main.verbose = true;
                    break;

                default:
                    if (!arg.startsWith("-")) {
                        main.inputPath = arg;
                    }
                    break;
            }
        }

        try {
            int exitCode = main.call();
            System.exit(exitCode);
        } catch (Exception e) {
            System.err.println("Fatal error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
