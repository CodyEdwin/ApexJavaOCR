package com.apexocr.engine;

import com.apexocr.core.tensor.Tensor;
import com.apexocr.core.neural.Layer;
import com.apexocr.core.neural.Conv2D;
import com.apexocr.core.neural.MaxPool2D;
import com.apexocr.core.neural.Dense;
import com.apexocr.core.neural.BiLSTM;
import com.apexocr.core.neural.ReshapeLayer;
import com.apexocr.core.ctc.CTCDecoder;
import com.apexocr.preprocessing.ImagePreprocessor;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

/**
 * OcrEngine - Main OCR processing engine for ApexJavaOCR.
 * Integrates image preprocessing, neural network inference, and CTC decoding
 * to provide high-accuracy text recognition from images.
 *
 * This engine is designed to outperform traditional OCR systems like Tesseract
 * through the use of modern deep learning architectures (CRNN) optimized for
 * the JVM, combined with sophisticated preprocessing pipelines.
 *
 * @author ApexOCR Team
 * @version 1.0.0
 */
public class OcrEngine implements AutoCloseable {
    private final EngineConfig config;
    private final ImagePreprocessor preprocessor;
    private final CTCDecoder decoder;
    private final List<Layer> network;

    private final ExecutorService executor;
    private boolean initialized;
    private boolean closed;

    /**
     * Configuration for the OCR engine.
     */
    public static class EngineConfig {
        public int batchSize = 1;
        public boolean useGPU = false;
        public int numThreads = Runtime.getRuntime().availableProcessors();
        public float confidenceThreshold = 0.5f;
        public boolean enablePreprocessing = true;
        public ImagePreprocessor.PreprocessingConfig preprocessingConfig =
            ImagePreprocessor.PreprocessingConfig.createDefault();
        public int beamWidth = 10;
        public String[] vocabulary = null;

        /**
         * Creates a default configuration.
         */
        public static EngineConfig createDefault() {
            return new EngineConfig();
        }

        /**
         * Creates a configuration optimized for high accuracy.
         */
        public static EngineConfig forHighAccuracy() {
            EngineConfig config = new EngineConfig();
            config.beamWidth = 20;
            config.confidenceThreshold = 0.3f;
            return config;
        }

        /**
         * Creates a configuration optimized for speed.
         */
        public static EngineConfig forSpeed() {
            EngineConfig config = new EngineConfig();
            config.batchSize = 4;
            config.beamWidth = 5;
            config.confidenceThreshold = 0.6f;
            return config;
        }
    }

    /**
     * Default vocabulary for English text recognition.
     */
    private static final String[] DEFAULT_VOCABULARY = {
        " ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-",
        ".", "/", ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^",
        "_", "`", "{", "|", "}", "~"
    };

    /**
     * Creates a new OCR engine with the specified configuration.
     *
     * @param config The engine configuration
     */
    public OcrEngine(EngineConfig config) {
        this.config = config;
        this.preprocessor = new ImagePreprocessor(config.preprocessingConfig);
        this.initialized = false;
        this.closed = false;

        // Create thread pool for parallel processing
        this.executor = Executors.newFixedThreadPool(config.numThreads);

        // Initialize vocabulary
        String[] vocab = config.vocabulary != null ? config.vocabulary : DEFAULT_VOCABULARY;
        String[] labels = new String[vocab.length + 1];
        labels[0] = ""; // Blank token
        System.arraycopy(vocab, 0, labels, 1, vocab.length);

        this.decoder = CTCDecoder.createDefault(labels);

        this.network = new ArrayList<>();
    }

    /**
     * Creates an OCR engine with default settings.
     */
    public OcrEngine() {
        this(EngineConfig.createDefault());
    }

    /**
     * Initializes the neural network with pre-trained weights.
     * In a production system, this would load weights from a file.
     */
    public void initialize() {
        if (initialized) {
            return;
        }

        // Adjust preprocessing target height based on network architecture
        // The network has 4 conv layers with stride [3,1], so it reduces height by ~81x
        // We need the preprocessed height to be at least 64 to get non-zero output
        config.preprocessingConfig.targetHeight = 64;

        buildNetwork();

        // Initialize layers that don't depend on input shape
        // Conv2D and MaxPool2D do lazy initialization on first forward pass
        // Dense and BiLSTM will be initialized on first forward pass with actual input shape
        // ReshapeLayer doesn't need initialization
        // So we don't need to call initialize() on any layers here

        // In a real implementation, load pre-trained weights here
        // loadWeights("path/to/weights.bin");

        initialized = true;
    }

    /**
     * Builds the CRNN (Convolutional Recurrent Neural Network) architecture.
     */
    private void buildNetwork() {
        network.clear();

        // Convolutional layers for feature extraction
        // Layer 1: Conv -> MaxPool
        network.add(new Conv2D(64, new int[]{3, 3}, new int[]{3, 1}, new int[]{0, 0}, Dense.ActivationType.RELU));
        network.add(new MaxPool2D(new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0}));

        // Layer 2: Conv -> MaxPool
        network.add(new Conv2D(128, new int[]{3, 3}, new int[]{3, 1}, new int[]{0, 0}, Dense.ActivationType.RELU));
        network.add(new MaxPool2D(new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0}));

        // Layer 3: Conv -> MaxPool
        network.add(new Conv2D(256, new int[]{3, 3}, new int[]{3, 1}, new int[]{0, 0}, Dense.ActivationType.RELU));
        network.add(new MaxPool2D(new int[]{2, 2}, new int[]{2, 1}, new int[]{0, 0}));

        // Layer 4: Conv -> MaxPool
        network.add(new Conv2D(512, new int[]{3, 3}, new int[]{3, 1}, new int[]{0, 0}, Dense.ActivationType.RELU));
        network.add(new MaxPool2D(new int[]{2, 2}, new int[]{2, 1}, new int[]{0, 0}));

        // Reshape 4D tensor [batch, height, width, channels] to 3D tensor [batch, width, height*channels]
        // This flattens the height and channels dimensions to create a sequence for the RNN
        network.add(new ReshapeLayer());

        // Bidirectional LSTM layers for sequence modeling
        // These expect input shape: [batch, timeSteps, features]
        network.add(new BiLSTM(256, true, 0.0f));
        network.add(new BiLSTM(256, true, 0.0f));

        // Final dense layer for classification
        network.add(new Dense(512, Dense.ActivationType.RELU, true));

        // Output layer (vocabulary size + 1 for blank)
        network.add(new Dense(decoder.getVocabularySize(), Dense.ActivationType.SOFTMAX, true));
    }

    /**
     * Processes an image and returns the recognized text.
     *
     * @param image The input image
     * @return The OCR result containing recognized text and metadata
     */
    public OcrResult process(BufferedImage image) {
        if (!initialized) {
            initialize();
        }

        long startTime = System.currentTimeMillis();

        // Preprocess the image
        BufferedImage processed = config.enablePreprocessing ?
            preprocessor.process(image) : image;

        // Log preprocessing info for debugging
        System.err.println("DEBUG: Input image dimensions: " + image.getWidth() + "x" + image.getHeight());
        System.err.println("DEBUG: Preprocessed image dimensions: " + processed.getWidth() + "x" + processed.getHeight());

        // Extract features as tensor
        Tensor input = imageToTensor(processed);

        // Run neural network inference
        Tensor output = runInference(input);

        // Decode output to text
        String text = decoder.decode(output);
        float confidence = computeConfidence(output, text);

        long processingTime = System.currentTimeMillis() - startTime;

        return new OcrResult(text, confidence, processingTime, processed.getWidth(), processed.getHeight());
    }

    /**
     * Processes multiple images in parallel for higher throughput.
     *
     * @param images The input images
     * @return List of OCR results
     */
    public List<OcrResult> processBatch(List<BufferedImage> images) {
        if (!initialized) {
            initialize();
        }

        List<Future<OcrResult>> futures = new ArrayList<>();

        for (BufferedImage image : images) {
            futures.add(executor.submit(() -> process(image)));
        }

        List<OcrResult> results = new ArrayList<>();
        for (Future<OcrResult> future : futures) {
            try {
                results.add(future.get(30, TimeUnit.SECONDS));
            } catch (Exception e) {
                results.add(new OcrResult("", 0, 0, 0, 0));
            }
        }

        return results;
    }

    /**
     * Converts a buffered image to an input tensor.
     *
     * @param image The input image
     * @return Tensor suitable for neural network input
     */
    private Tensor imageToTensor(BufferedImage image) {
        int height = image.getHeight();
        int width = image.getWidth();

        // Validate minimum dimensions for CRNN network architecture
        // The network has 4 conv layers (3x3 kernels) and 4 max-pool layers (2x2)
        // Minimum height needed: (3-1)*4 + 1 = 9, but we use 32 as target
        // Minimum width needed depends on number of characters, but should be > 8
        int minHeight = 32;  // Target height for the network
        int minWidth = 8;    // Minimum width for at least one character

        if (height < minHeight) {
            throw new IllegalArgumentException(
                String.format("Image height (%d) is too small. Minimum required is %d pixels. " +
                    "The image will be resized to %d pixels height during preprocessing.",
                    height, minHeight, minHeight));
        }

        if (width < minWidth) {
            throw new IllegalArgumentException(
                String.format("Image width (%d) is too small. Minimum required is %d pixels. " +
                    "This may indicate the image is too narrow for text recognition.",
                    width, minWidth));
        }

        // Ensure dimensions are compatible with network
        int timeSteps = width;
        int features = height;

        long[] shape = new long[]{1, height, width, 1};
        Tensor tensor = new Tensor(shape, Tensor.DataType.FLOAT32);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = image.getRaster().getSample(x, y, 0);
                float normalized = pixel / 255.0f;
                tensor.setFloat(normalized, 0, y, x, 0);
            }
        }

        return tensor;
    }

    /**
     * Runs neural network inference on the input tensor.
     *
     * @param input The input tensor
     * @return Output tensor containing character probabilities
     */
    private Tensor runInference(Tensor input) {
        Tensor current = input;

        // Set all layers to evaluation mode
        for (Layer layer : network) {
            layer.eval();
        }

        // Forward pass through all layers
        for (Layer layer : network) {
            current = layer.forward(current, false);
        }

        return current;
    }

    /**
     * Computes confidence score for the recognition result.
     *
     * @param output Network output tensor
     * @param text Recognized text
     * @return Confidence score between 0 and 1
     */
    private float computeConfidence(Tensor output, String text) {
        // Compute average probability of predicted characters
        long[] shape = output.getShape();
        int timeSteps = (int) shape[0];
        int numClasses = (int) shape[1];

        // Simple confidence: average of max probabilities per time step
        float totalProb = 0;
        for (int t = 0; t < timeSteps; t++) {
            float maxProb = 0;
            for (int c = 0; c < numClasses; c++) {
                maxProb = Math.max(maxProb, output.getFloat(t, c));
            }
            totalProb += maxProb;
        }

        return totalProb / timeSteps;
    }

    /**
     * Processes an image file and returns the recognized text.
     *
     * @param filePath Path to the image file
     * @return The OCR result
     */
    public OcrResult processFile(String filePath) throws IOException {
        BufferedImage image = ImageIO.read(new File(filePath));
        if (image == null) {
            throw new IOException("Failed to load image: " + filePath);
        }
        return process(image);
    }

    /**
     * Processes multiple image files in batch.
     *
     * @param filePaths Paths to image files
     * @return List of OCR results
     */
    public List<OcrResult> processFiles(List<String> filePaths) throws IOException {
        List<BufferedImage> images = new ArrayList<>();
        for (String path : filePaths) {
            BufferedImage image = ImageIO.read(new File(path));
            if (image == null) {
                throw new IOException("Failed to load image: " + path);
            }
            images.add(image);
        }
        return processBatch(images);
    }

    /**
     * Gets the preprocessing pipeline.
     *
     * @return The image preprocessor
     */
    public ImagePreprocessor getPreprocessor() {
        return preprocessor;
    }

    /**
     * Gets the CTC decoder.
     *
     * @return The CTC decoder
     */
    public CTCDecoder getDecoder() {
        return decoder;
    }

    /**
     * Checks if the engine is initialized.
     *
     * @return True if initialized
     */
    public boolean isInitialized() {
        return initialized;
    }

    /**
     * Gets the current configuration.
     *
     * @return The engine configuration
     */
    public EngineConfig getConfig() {
        return config;
    }

    /**
     * Gets the number of network layers.
     *
     * @return Layer count
     */
    public int getLayerCount() {
        return network.size();
    }

    /**
     * Gets the total number of parameters in the network.
     *
     * @return Total parameter count
     */
    public long getParameterCount() {
        return network.stream().mapToLong(Layer::getParameterCount).sum();
    }

    /**
     * Gets a summary of the network architecture.
     *
     * @return Architecture summary string
     */
    public String getArchitectureSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("ApexJavaOCR Network Architecture\n");
        sb.append("================================\n\n");

        long totalParams = 0;
        for (Layer layer : network) {
            sb.append(layer.summary()).append("\n");
            totalParams += layer.getParameterCount();
        }

        sb.append("\nTotal Parameters: ").append(String.format("%,d", totalParams));
        sb.append("\nVocabulary Size: ").append(decoder.getVocabularySize());

        return sb.toString();
    }

    /**
     * Resets any internal state (useful for processing multiple unrelated images).
     */
    public void resetState() {
        for (Layer layer : network) {
            layer.resetState();
        }
    }

    /**
     * Enables evaluation mode.
     */
    public void eval() {
        for (Layer layer : network) {
            layer.eval();
        }
    }

    /**
     * Enables training mode.
     */
    public void train() {
        for (Layer layer : network) {
            layer.train();
        }
    }
    
    /**
     * Loads pre-trained weights from a binary file.
     * The weight file format supports version 1 of ApexOCR model format.
     * 
     * @param weightFilePath Path to the weight file
     * @return true if weights were loaded successfully
     */
    public boolean loadWeights(String weightFilePath) {
        try {
            Path path = Paths.get(weightFilePath);
            byte[] data = Files.readAllBytes(path);
            return loadWeightsFromBytes(data);
        } catch (IOException e) {
            System.err.println("Error loading weights from " + weightFilePath + ": " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Loads weights from a byte array.
     * 
     * @param data Serialized weight data
     * @return true if weights were loaded successfully
     */
    public boolean loadWeightsFromBytes(byte[] data) {
        if (data == null || data.length == 0) {
            return false;
        }
        
        ByteBuffer buffer = ByteBuffer.wrap(data);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        try {
            // Read magic number
            int magic = buffer.getInt();
            if (magic != 0x41504558) { // "APEX" in little endian
                System.err.println("Invalid weight file format (bad magic number)");
                return false;
            }
            
            int version = buffer.getInt();
            if (version != 1) {
                System.err.println("Unsupported weight file version: " + version);
                return false;
            }
            
            // Read number of layers with weights
            int numLayers = buffer.getInt();
            
            int layerIndex = 0;
            for (Layer layer : network) {
                long paramCount = layer.getParameterCount();
                if (paramCount > 0 && layerIndex < numLayers) {
                    int paramSize = buffer.getInt();
                    byte[] layerData = new byte[paramSize];
                    buffer.get(layerData);
                    layer.deserializeParameters(layerData);
                    layerIndex++;
                }
            }
            
            System.out.println("Successfully loaded " + layerIndex + " layers of weights");
            return true;
            
        } catch (Exception e) {
            System.err.println("Error deserializing weights: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Saves the current network weights to a binary file.
     * 
     * @param weightFilePath Path to save the weight file
     * @return true if weights were saved successfully
     */
    public boolean saveWeights(String weightFilePath) {
        try {
            // First pass: calculate total size needed
            int headerSize = 16; // magic (4) + version (4) + numLayers (4) + layer count entries (4)
            int layerCountEntries = 0;
            int totalDataSize = 0;
            
            for (Layer layer : network) {
                if (layer.getParameterCount() > 0) {
                    byte[] data = layer.serializeParameters();
                    if (data.length > 0) {
                        layerCountEntries++;
                        totalDataSize += 4 + data.length; // size prefix + data
                    }
                }
            }
            
            // Allocate buffer with dynamic size
            int totalSize = headerSize + totalDataSize;
            ByteBuffer buffer = ByteBuffer.allocate(totalSize);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            
            // Write magic number "APEX"
            buffer.putInt(0x41504558);
            
            // Write version
            buffer.putInt(1);
            
            // Write number of layers
            buffer.putInt(layerCountEntries);
            
            // Write each layer's data
            for (Layer layer : network) {
                if (layer.getParameterCount() > 0) {
                    byte[] data = layer.serializeParameters();
                    if (data.length > 0) {
                        buffer.putInt(data.length);
                        buffer.put(data);
                    }
                }
            }
            
            // Write to file
            Files.write(Paths.get(weightFilePath), buffer.array());
            
            System.out.println("Successfully saved " + layerCountEntries + " layers of weights (" + totalSize + " bytes) to " + weightFilePath);
            return true;
            
        } catch (IOException e) {
            System.err.println("Error saving weights to " + weightFilePath + ": " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Generates demo weights that produce consistent (though not meaningful) output.
     * This is useful for testing the pipeline with deterministic results.
     */
    public void generateDemoWeights() {
        // Generate deterministic weights based on layer index
        int layerNum = 0;
        for (Layer layer : network) {
            layerNum++;
            Tensor weights = layer.getWeights();
            Tensor biases = layer.getBiases();
            
            if (weights != null) {
                // Fill with deterministic values using proper multi-dimensional indexing
                long[] shape = weights.getShape();
                long totalSize = weights.getSize();
                long counter = 0;
                
                // Use iterative approach for multi-dimensional tensors
                int rank = shape.length;
                int[] indices = new int[rank];
                
                for (long i = 0; i < totalSize; i++) {
                    // Calculate value based on layer and position
                    float val = (float) (Math.sin(layerNum * 1000 + counter * 0.01) * 0.1);
                    
                    // Set value at current indices
                    switch (rank) {
                        case 1:
                            weights.setFloat(val, indices[0]);
                            break;
                        case 2:
                            weights.setFloat(val, indices[0], indices[1]);
                            break;
                        case 3:
                            weights.setFloat(val, indices[0], indices[1], indices[2]);
                            break;
                        case 4:
                            weights.setFloat(val, indices[0], indices[1], indices[2], indices[3]);
                            break;
                        default:
                            weights.setFloat(val, i);
                    }
                    
                    // Increment counter and update indices
                    counter++;
                    
                    // Update multi-dimensional indices
                    for (int d = rank - 1; d >= 0; d--) {
                        indices[d]++;
                        if (d > 0 && indices[d] >= shape[d]) {
                            indices[d] = 0;
                        } else {
                            break;
                        }
                    }
                }
            }
            
            if (biases != null) {
                // Fill biases with deterministic values
                long size = biases.getSize();
                for (long i = 0; i < size; i++) {
                    float val = (float) (Math.cos(layerNum * 100 + i * 0.1) * 0.05f);
                    biases.setFloat(val, (int) i);
                }
            }
        }
    }

    @Override
    public void close() {
        if (closed) return;

        // Clean up network layers
        for (Layer layer : network) {
            layer.close();
        }
        network.clear();

        // Shutdown executor
        executor.shutdown();
        try {
            if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }

        closed = true;
    }

    /**
     * Main entry point for quick testing.
     */
    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Usage: java OcrEngine <image_path>");
            System.out.println("Or use the ApexOCR CLI for full functionality.");
            return;
        }

        try (OcrEngine engine = new OcrEngine()) {
            engine.initialize();

            long start = System.currentTimeMillis();
            OcrResult result = engine.processFile(args[0]);
            long time = System.currentTimeMillis() - start;

            System.out.println("Recognized Text:");
            System.out.println(result.getText());
            System.out.println("\nConfidence: " + String.format("%.2f%%", result.getConfidence() * 100));
            System.out.println("Processing Time: " + time + "ms");
        } catch (IOException e) {
            System.err.println("Error processing image: " + e.getMessage());
        }
    }
}
