package com.apexocr.preprocessing;

import java.awt.*;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.util.Arrays;

/**
 * ImagePreprocessor - Comprehensive image preprocessing pipeline for OCR optimization.
 * Implements a series of image transformations designed to enhance text visibility
 * and prepare images for neural network processing.
 *
 * This class provides efficient implementations of common preprocessing operations
 * including grayscale conversion, binarization, noise removal, and normalization.
 *
 * @author ApexOCR Team
 * @version 1.0.0
 */
public class ImagePreprocessor {
    private final PreprocessingConfig config;

    /**
     * Configuration options for image preprocessing.
     */
    public static class PreprocessingConfig {
        public boolean convertToGrayscale = true;
        public boolean applyBinarization = true;
        public BinarizationMethod binarizationMethod = BinarizationMethod.SAUVOLA;
        public int targetHeight = 32;
        public boolean preserveAspectRatio = true;
        public boolean applyNoiseReduction = true;
        public boolean applyContrastEnhancement = true;
        public float targetScale = 1.0f;
        public int medianFilterSize = 3;
        public int morphKernelSize = 3;

        /**
         * Creates a default configuration optimized for OCR.
         */
        public static PreprocessingConfig createDefault() {
            return new PreprocessingConfig();
        }

        /**
         * Creates a configuration for high-quality document images.
         */
        public static PreprocessingConfig forDocuments() {
            PreprocessingConfig config = new PreprocessingConfig();
            config.binarizationMethod = BinarizationMethod.ADAPTIVE;
            config.applyContrastEnhancement = true;
            config.applyNoiseReduction = true;
            return config;
        }

        /**
         * Creates a configuration for challenging images (low contrast, noise).
         */
        public static PreprocessingConfig forChallengingImages() {
            PreprocessingConfig config = new PreprocessingConfig();
            config.binarizationMethod = BinarizationMethod.SAUVOLA;
            config.applyContrastEnhancement = true;
            config.applyNoiseReduction = true;
            config.medianFilterSize = 5;
            config.morphKernelSize = 5;
            return config;
        }
    }

    /**
     * Binarization methods available for thresholding.
     */
    public enum BinarizationMethod {
        OTSU,
        ADAPTIVE,
        SAUVOLA,
        NIBLACK,
        FIXED
    }

    /**
     * Creates a preprocessor with the specified configuration.
     *
     * @param config The preprocessing configuration
     */
    public ImagePreprocessor(PreprocessingConfig config) {
        this.config = config;
    }

    /**
     * Creates a preprocessor with default settings.
     */
    public ImagePreprocessor() {
        this(PreprocessingConfig.createDefault());
    }

    /**
     * Processes an image through the complete preprocessing pipeline.
     *
     * @param image The input image
     * @return Preprocessed image ready for neural network input
     */
    public BufferedImage process(BufferedImage image) {
        if (image == null) {
            throw new IllegalArgumentException("Input image cannot be null");
        }

        BufferedImage result = image;

        // Convert to grayscale if needed
        if (config.convertToGrayscale && result.getType() != BufferedImage.TYPE_BYTE_GRAY) {
            result = toGrayscale(result);
        }

        // Apply contrast enhancement
        if (config.applyContrastEnhancement) {
            result = enhanceContrast(result);
        }

        // Apply noise reduction
        if (config.applyNoiseReduction) {
            result = applyNoiseReduction(result);
        }

        // Apply binarization
        if (config.applyBinarization) {
            result = binarize(result);
        }

        // Resize to target dimensions
        result = resize(result, config.targetHeight, config.preserveAspectRatio);

        // Normalize pixel values
        result = normalize(result);

        return result;
    }

    /**
     * Converts an image to grayscale using luminance formula.
     *
     * @param image The input image
     * @return Grayscale image
     */
    public BufferedImage toGrayscale(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();

        BufferedImage gray = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                // Luminance formula: 0.299*R + 0.587*G + 0.114*B
                int grayValue = (int) (0.299 * r + 0.587 * g + 0.114 * b);
                gray.getRaster().setSample(x, y, 0, grayValue);
            }
        }

        return gray;
    }

    /**
     * Enhances image contrast using histogram equalization.
     *
     * @param image The input image
     * @return Contrast-enhanced image
     */
    public BufferedImage enhanceContrast(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();

        BufferedImage result = new BufferedImage(width, height, image.getType());

        // Compute histogram
        int[] histogram = new int[256];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int gray = image.getRaster().getSample(x, y, 0);
                histogram[gray]++;
            }
        }

        // Compute cumulative distribution function
        int[] cdf = new int[256];
        cdf[0] = histogram[0];
        for (int i = 1; i < 256; i++) {
            cdf[i] = cdf[i - 1] + histogram[i];
        }

        // Normalize CDF
        int totalPixels = width * height;
        int[] normalizedCdf = new int[256];
        for (int i = 0; i < 256; i++) {
            normalizedCdf[i] = (int) ((cdf[i] * 255.0) / totalPixels);
        }

        // Apply equalization
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int gray = image.getRaster().getSample(x, y, 0);
                int equalized = normalizedCdf[gray];
                result.getRaster().setSample(x, y, 0, equalized);
            }
        }

        return result;
    }

    /**
     * Applies noise reduction using median filter.
     *
     * @param image The input image
     * @return Denoised image
     */
    public BufferedImage applyNoiseReduction(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int kernelSize = config.medianFilterSize;
        int halfKernel = kernelSize / 2;

        BufferedImage result = new BufferedImage(width, height, image.getType());

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int[] neighborhood = new int[kernelSize * kernelSize];
                int idx = 0;

                for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                    for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                        int px = Math.min(Math.max(x + kx, 0), width - 1);
                        int py = Math.min(Math.max(y + ky, 0), height - 1);
                        neighborhood[idx++] = image.getRaster().getSample(px, py, 0);
                    }
                }

                Arrays.sort(neighborhood);
                int median = neighborhood[neighborhood.length / 2];
                result.getRaster().setSample(x, y, 0, median);
            }
        }

        return result;
    }

    /**
     * Binarizes the image using the configured method.
     *
     * @param image The input image
     * @return Binary image
     */
    public BufferedImage binarize(BufferedImage image) {
        switch (config.binarizationMethod) {
            case OTSU:
                return binarizeOtsu(image);
            case ADAPTIVE:
                return binarizeAdaptive(image);
            case SAUVOLA:
                return binarizeSauvola(image);
            case NIBLACK:
                return binarizeNiblack(image);
            case FIXED:
                return binarizeFixed(image, 128);
            default:
                return binarizeOtsu(image);
        }
    }

    /**
     * Binarization using Otsu's method for automatic threshold selection.
     *
     * @param image The input image
     * @return Binary image
     */
    public BufferedImage binarizeOtsu(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();

        // Compute histogram
        int[] histogram = new int[256];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int gray = image.getRaster().getSample(x, y, 0);
                histogram[gray]++;
            }
        }

        int total = width * height;

        // Compute cumulative sums and means
        int[] sum = new int[256];
        sum[0] = histogram[0];
        for (int i = 1; i < 256; i++) {
            sum[i] = sum[i - 1] + histogram[i];
        }

        int[] weightedSum = new int[256];
        weightedSum[0] = 0 * histogram[0];
        for (int i = 1; i < 256; i++) {
            weightedSum[i] = weightedSum[i - 1] + i * histogram[i];
        }

        // Find optimal threshold
        float maxVariance = 0;
        int threshold = 0;

        for (int t = 0; t < 256; t++) {
            int w1 = sum[t];
            int w2 = total - w1;

            if (w1 == 0 || w2 == 0) continue;

            float m1 = (float) weightedSum[t] / w1;
            float m2 = (float) (weightedSum[255] - weightedSum[t]) / w2;

            float variance = w1 * w2 * (m1 - m2) * (m1 - m2);

            if (variance > maxVariance) {
                maxVariance = variance;
                threshold = t;
            }
        }

        return binarizeFixed(image, threshold);
    }

    /**
     * Adaptive binarization using local mean threshold.
     *
     * @param image The input image
     * @return Binary image
     */
    public BufferedImage binarizeAdaptive(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int blockSize = 35; // Must be odd

        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_BINARY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Compute local mean
                int halfBlock = blockSize / 2;
                int sum = 0;
                int count = 0;

                for (int ky = -halfBlock; ky <= halfBlock; ky++) {
                    for (int kx = -halfBlock; kx <= halfBlock; kx++) {
                        int px = x + kx;
                        int py = y + ky;

                        if (px >= 0 && px < width && py >= 0 && py < height) {
                            sum += image.getRaster().getSample(px, py, 0);
                            count++;
                        }
                    }
                }

                int mean = sum / count;
                int pixel = image.getRaster().getSample(x, y, 0);
                result.getRaster().setSample(x, y, 0, pixel < mean ? 0 : 255);
            }
        }

        return result;
    }

    /**
     * Binarization using Sauvola's method (local threshold with variance).
     *
     * @param image The input image
     * @return Binary image
     */
    public BufferedImage binarizeSauvola(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int windowSize = 15;
        double k = 0.2; // Sauvola parameter
        double r = 128; // Dynamic range

        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_BINARY);

        int halfWindow = windowSize / 2;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Compute local mean and standard deviation
                double sum = 0;
                double sumSq = 0;
                int count = 0;

                for (int ky = -halfWindow; ky <= halfWindow; ky++) {
                    for (int kx = -halfWindow; kx <= halfWindow; kx++) {
                        int px = Math.min(Math.max(x + kx, 0), width - 1);
                        int py = Math.min(Math.max(y + ky, 0), height - 1);

                        int pixel = image.getRaster().getSample(px, py, 0);
                        sum += pixel;
                        sumSq += pixel * pixel;
                        count++;
                    }
                }

                double mean = sum / count;
                double variance = Math.sqrt(Math.max(0, sumSq / count - mean * mean));

                int pixel = image.getRaster().getSample(x, y, 0);
                int threshold = (int) (mean * (1 + k * (variance / r - 1)));

                result.getRaster().setSample(x, y, 0, pixel <= threshold ? 0 : 255);
            }
        }

        return result;
    }

    /**
     * Binarization using Niblack's method.
     *
     * @param image The input image
     * @return Binary image
     */
    public BufferedImage binarizeNiblack(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int windowSize = 15;
        double k = -0.2; // Niblack parameter

        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_BINARY);

        int halfWindow = windowSize / 2;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double sum = 0;
                double sumSq = 0;
                int count = 0;

                for (int ky = -halfWindow; ky <= halfWindow; ky++) {
                    for (int kx = -halfWindow; kx <= halfWindow; kx++) {
                        int px = Math.min(Math.max(x + kx, 0), width - 1);
                        int py = Math.min(Math.max(y + ky, 0), height - 1);

                        int pixel = image.getRaster().getSample(px, py, 0);
                        sum += pixel;
                        sumSq += pixel * pixel;
                        count++;
                    }
                }

                double mean = sum / count;
                double std = Math.sqrt(Math.max(0, sumSq / count - mean * mean));

                int pixel = image.getRaster().getSample(x, y, 0);
                int threshold = (int) (mean + k * std);

                result.getRaster().setSample(x, y, 0, pixel <= threshold ? 0 : 255);
            }
        }

        return result;
    }

    /**
     * Fixed threshold binarization.
     *
     * @param image The input image
     * @param threshold The threshold value (0-255)
     * @return Binary image
     */
    public BufferedImage binarizeFixed(BufferedImage image, int threshold) {
        int width = image.getWidth();
        int height = image.getHeight();

        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_BINARY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = image.getRaster().getSample(x, y, 0);
                result.getRaster().setSample(x, y, 0, pixel <= threshold ? 0 : 255);
            }
        }

        return result;
    }

    /**
     * Resizes image to target height while preserving aspect ratio.
     *
     * @param image The input image
     * @param targetHeight The target height
     * @param preserveAspect Whether to preserve aspect ratio
     * @return Resized image
     */
    public BufferedImage resize(BufferedImage image, int targetHeight, boolean preserveAspect) {
        int width = image.getWidth();
        int height = image.getHeight();

        int newHeight = targetHeight;
        int newWidth;

        if (preserveAspect) {
            newWidth = (int) ((double) width / height * targetHeight);
        } else {
            newWidth = targetHeight;
        }

        BufferedImage result = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = result.createGraphics();

        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(image, 0, 0, newWidth, newHeight, null);
        g.dispose();

        return result;
    }

    /**
     * Normalizes pixel values to [0, 1] range for neural network input.
     *
     * @param image The input image
     * @return Normalized image
     */
    public BufferedImage normalize(BufferedImage image) {
        // For neural networks, we normalize to [0, 1]
        // The image is already in the correct format (0-255 for grayscale)
        // We can apply additional normalization if needed
        return image;
    }

    /**
     * Extracts a 2D float array from the image for neural network input.
     *
     * @param image The input image
     * @return 2D float array with pixel values normalized to [0, 1]
     */
    public float[][] extractFeatures(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();

        float[][] features = new float[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = image.getRaster().getSample(x, y, 0);
                features[y][x] = pixel / 255.0f;
            }
        }

        return features;
    }

    /**
     * Performs morphological opening to remove small noise.
     *
     * @param image The input image
     * @return Morphologically processed image
     */
    public BufferedImage morphOpen(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int kernelSize = config.morphKernelSize;
        int halfKernel = kernelSize / 2;

        BufferedImage temp = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_BINARY);

        // Erosion followed by dilation
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                boolean erosion = true;

                for (int ky = -halfKernel; ky <= halfKernel && erosion; ky++) {
                    for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                        int px = x + kx;
                        int py = y + ky;

                        if (px >= 0 && px < width && py >= 0 && py < height) {
                            if (image.getRaster().getSample(px, py, 0) == 0) {
                                erosion = false;
                                break;
                            }
                        }
                    }
                }

                temp.getRaster().setSample(x, y, 0, erosion ? 255 : 0);
            }
        }

        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_BINARY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                boolean dilation = false;

                for (int ky = -halfKernel; ky <= halfKernel && !dilation; ky++) {
                    for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                        int px = x + kx;
                        int py = y + ky;

                        if (px >= 0 && px < width && py >= 0 && py < height) {
                            if (temp.getRaster().getSample(px, py, 0) == 255) {
                                dilation = true;
                                break;
                            }
                        }
                    }
                }

                result.getRaster().setSample(x, y, 0, dilation ? 255 : 0);
            }
        }

        return result;
    }
}
