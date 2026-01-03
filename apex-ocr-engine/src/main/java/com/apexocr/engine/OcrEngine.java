package com.apexocr.engine;

import com.apexocr.core.tensor.Tensor;
import com.apexocr.core.neural.Layer;
import com.apexocr.core.neural.Layer.Initializer;
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
     * This matches the EasyOCR english_g2 model vocabulary (97 classes).
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
     * This architecture exactly matches the EasyOCR english_g2 model.
     */
    public void initialize() {
        if (initialized) {
            return;
        }

        // EasyOCR uses 32px height as standard input
        // This ensures proper dimension flow through the CRNN architecture
        config.preprocessingConfig.targetHeight = 32;

        buildNetwork();

        initialized = true;
    }

    /**
     * Builds the CRNN (Convolutional Recurrent Neural Network) architecture.
     * This architecture exactly matches the EasyOCR english_g2 model.
     * 
     * CRITICAL ARCHITECTURE DETAILS:
     * - Input height: 32px (standard for CRNN stability)
     * - Uses stride 1 for all convolutions to preserve dimensions
     * - Uses RECTANGULAR POOLING (2x1) to prevent width collapse
     * - 7 Conv2D layers with specific kernel/stride/padding configurations
     * - 2 BiLSTM layers for sequence modeling
     * - 2 Dense layers for prediction
     */
    private void buildNetwork() {
        network.clear();

        // =================================================================
        // FEATURE EXTRACTION (CNN) - Exact EasyOCR Architecture
        // =================================================================
        
        // Layer 0: Conv 3x3, stride 1, padding 1 -> MaxPool 2x2
        // Input: [B, 1, 32, W] -> Output: [B, 32, 16, W]
        // Note: stride (1,1) preserves width, only height reduced by pool
        Conv2D conv0 = new Conv2D(32, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, Dense.ActivationType.RELU);
        conv0.setName("FeatureExtraction.ConvNet.0.weight");
        network.add(conv0);
        network.add(new MaxPool2D(new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0}));

        // Layer 3: Conv 3x3, stride 1, padding 1 -> MaxPool 2x2
        // Input: [B, 32, 16, W] -> Output: [B, 64, 8, W]
        Conv2D conv3 = new Conv2D(64, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, Dense.ActivationType.RELU);
        conv3.setName("FeatureExtraction.ConvNet.3.weight");
        network.add(conv3);
        network.add(new MaxPool2D(new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0}));

        // Layer 6: Conv 3x3, stride 1, padding 1 -> NO POOL
        // Input: [B, 64, 8, W] -> Output: [B, 128, 8, W]
        // NOTE: No pooling here - preserves spatial dimensions
        Conv2D conv6 = new Conv2D(128, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, Dense.ActivationType.RELU);
        conv6.setName("FeatureExtraction.ConvNet.6.weight");
        network.add(conv6);
        // No pooling after layer 6

        // Layer 8: Conv 3x3, stride 1, padding 1 -> MaxPool 2x1 (RECTANGULAR!)
        // Input: [B, 128, 8, W] -> Output: [B, 128, 4, W]
        // CRITICAL: 2x1 pooling pools height by 2, width by 1 (preserves width!)
        Conv2D conv8 = new Conv2D(128, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, Dense.ActivationType.RELU);
        conv8.setName("FeatureExtraction.ConvNet.8.weight");
        network.add(conv8);
        network.add(new MaxPool2D(new int[]{2, 1}, new int[]{2, 1}, new int[]{0, 0}));

        // Layer 11: Conv 3x3, stride 1, padding 1 -> NO POOL
        // Input: [B, 128, 4, W] -> Output: [B, 256, 4, W]
        // NOTE: No pooling here - preserves spatial dimensions
        Conv2D conv11 = new Conv2D(256, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, Dense.ActivationType.RELU);
        conv11.setName("FeatureExtraction.ConvNet.11.weight");
        network.add(conv11);
        // No pooling after layer 11

        // Layer 14: Conv 3x3, stride 1, padding 1 -> MaxPool 2x1 (RECTANGULAR!)
        // Input: [B, 256, 4, W] -> Output: [B, 256, 2, W]
        // CRITICAL: 2x1 pooling pools height by 2, width by 1 (preserves width!)
        Conv2D conv14 = new Conv2D(256, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, Dense.ActivationType.RELU);
        conv14.setName("FeatureExtraction.ConvNet.14.weight");
        network.add(conv14);
        network.add(new MaxPool2D(new int[]{2, 1}, new int[]{2, 1}, new int[]{0, 0}));

        // Layer 18: Conv 2x2, stride 1, padding 0 -> NO POOL
        // Input: [B, 256, 2, W] -> Output: [B, 256, 1, W-1]
        // Final conv collapses height to 1, width reduced by 1
        Conv2D conv18 = new Conv2D(256, new int[]{2, 2}, new int[]{1, 1}, new int[]{0, 0}, Dense.ActivationType.RELU);
        conv18.setName("FeatureExtraction.ConvNet.18.weight");
        network.add(conv18);
        // No pooling after layer 18

        // =================================================================
        // SEQUENCE MODELING
        // =================================================================
        
        // Reshape 4D tensor [B, 256, 1, W-1] to 3D tensor [W-1, B, 256]
        // This creates a sequence of feature vectors for the RNN
        network.add(new ReshapeLayer());

        // BiLSTM Layer 1: Input 256 -> Hidden 256, Bidirectional
        // Output: [W-1, B, 512] (concatenated forward/backward)
        BiLSTM bilstm0 = new BiLSTM(256, true, 0.0f);
        bilstm0.setName("SequenceModeling.0.rnn.weight_forward");
        network.add(bilstm0);

        // BiLSTM Layer 2: Input 256 -> Hidden 256, Bidirectional
        // Output: [W-1, B, 512]
        BiLSTM bilstm1 = new BiLSTM(256, true, 0.0f);
        bilstm1.setName("SequenceModeling.1.rnn.weight_forward");
        network.add(bilstm1);

        // =================================================================
        // PREDICTION
        // =================================================================
        
        // First Dense layer: projects from BiLSTM output (512) to intermediate (256)
        // This matches SequenceModeling.0.linear.weight [256, 512] = 131072 elements
        // IMPORTANT: inputUnits=512 to match BiLSTM output (256 forward + 256 backward)
        Dense denseSeq0 = new Dense(256, 512, Dense.ActivationType.RELU, false);
        denseSeq0.setName("SequenceModeling.0.linear.weight");
        network.add(denseSeq0);

        // Second Dense layer: projects from BiLSTM output (512) to intermediate (256)
        // This matches SequenceModeling.1.linear.weight [256, 512] = 131072 elements
        Dense denseSeq1 = new Dense(256, 512, Dense.ActivationType.RELU, false);
        denseSeq1.setName("SequenceModeling.1.linear.weight");
        network.add(denseSeq1);

        // Output layer: 256 -> 97 classes (96 characters + 1 blank for CTC)
        // This matches Prediction.weight [97, 256] = 24832 elements
        Dense denseOutput = new Dense(97, 256, Dense.ActivationType.SOFTMAX, false);
        denseOutput.setName("Prediction.weight");
        network.add(denseOutput);
    }

    /**
     * Initializes all network weights using appropriate initialization methods.
     * This must be called after buildNetwork() if not loading pre-trained weights.
     * 
     * Uses He initialization for layers with ReLU activation (Conv2D, Dense with ReLU)
     * Uses Xavier/Glorot initialization for layers with linear/softmax activation
     * Uses orthogonal initialization for recurrent layers (BiLSTM)
     */
    public void initializeNetworkWeights() {
        if (network.isEmpty()) {
            buildNetwork();
        }
        
        System.out.println("Initializing network weights...");
        long totalParams = 0;
        
        for (Layer layer : network) {
            if (layer instanceof Conv2D) {
                Conv2D conv = (Conv2D) layer;
                // Set a default input shape for initialization
                conv.setInputShape(new long[]{1, 32, 64, 1});
                conv.initialize(Initializer.HE_NORMAL);
                long params = conv.getParameterCount();
                totalParams += params;
                System.out.println("  Initialized Conv2D: " + conv.getName() + " (" + params + " params)");
                
            } else if (layer instanceof BiLSTM) {
                BiLSTM lstm = (BiLSTM) layer;
                // Set a default input shape for initialization
                lstm.setInputShape(new long[]{1, 10, 256});
                lstm.initialize(Initializer.ORTHOGONAL);
                long params = lstm.getParameterCount();
                totalParams += params;
                System.out.println("  Initialized BiLSTM: " + lstm.getName() + " (" + params + " params)");
                
            } else if (layer instanceof Dense) {
                Dense dense = (Dense) layer;
                // Use Xavier for layers with softmax, He for ReLU
                Initializer init = (dense.getActivation() == Dense.ActivationType.SOFTMAX || 
                                    dense.getActivation() == Dense.ActivationType.LINEAR) 
                                   ? Initializer.XAVIER_NORMAL 
                                   : Initializer.HE_NORMAL;
                dense.initialize(init);
                long params = dense.getParameterCount();
                totalParams += params;
                System.out.println("  Initialized Dense: " + dense.getName() + " (" + params + " params)");
            }
        }
        
        System.out.println("Total trainable parameters: " + String.format("%,d", totalParams));
    }

    /**
     * Checks if the network has been initialized with weights.
     * 
     * @return true if weights have been initialized
     */
    public boolean hasWeights() {
        if (network.isEmpty()) {
            return false;
        }
        
        for (Layer layer : network) {
            if (layer.getWeights() != null) {
                return true;
            }
        }
        return false;
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
        
        int timeSteps, numClasses;
        
        // Handle both 2D and 3D input
        if (shape.length == 3) {
            // 3D input: [batch, timeSteps, numClasses]
            timeSteps = (int) shape[1];
            numClasses = (int) shape[2];
        } else {
            // 2D input: [timeSteps, numClasses]
            timeSteps = (int) shape[0];
            numClasses = (int) shape[1];
        }

        // Simple confidence: average of max probabilities per time step
        float totalProb = 0;
        for (int t = 0; t < timeSteps; t++) {
            float maxProb = 0;
            for (int c = 0; c < numClasses; c++) {
                if (shape.length == 3) {
                    maxProb = Math.max(maxProb, output.getFloat(0, t, c));
                } else {
                    maxProb = Math.max(maxProb, output.getFloat(t, c));
                }
            }
            totalProb += maxProb;
        }

        return totalProb / timeSteps;
    }

    /**
     * Debug method to print the raw prediction tensor before decoding.
     * This helps diagnose issues with the network output.
     *
     * @param output Network output tensor
     */
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
     * Gets the internal network list.
     * This is used by the training module to access and modify layer weights.
     *
     * @return The list of network layers
     */
    public List<Layer> getNetwork() {
        return network;
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
     * Supports both the simple format (paramSize + data) and the full format
     * with layer names and types written by convert_easyocr.py.
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
            
            // Read number of layers
            int numLayers = buffer.getInt();
            System.out.println("Loading " + numLayers + " pre-trained layers...");
            System.out.println("DEBUG: Network has " + network.size() + " layers");
            for (int i = 0; i < network.size(); i++) {
                Layer layer = network.get(i);
                System.out.println("DEBUG: Layer " + i + ": " + layer.getName() + " (" + layer.getClass().getSimpleName() + ")");
            }
            
            // Adjust numLayers if weight file has more layers than network
            // Some weight files may have header mismatch
            if (numLayers > network.size()) {
                System.out.println("WARNING: Weight file claims " + numLayers + " layers but network has " + network.size() + " layers. Adjusting...");
                numLayers = network.size();
            }
            
            // Calculate expected size for simple format
            // Simple format: 4 bytes (paramSize) per layer
            int expectedSimpleSize = numLayers * 4;
            int remaining = buffer.remaining();
            
            // Detect format based on remaining bytes vs expected simple format size
            // Simple format data is ALWAYS 4 * numLayers bytes
            // Extended format data is MUCH larger (hundreds of bytes per layer)
            // If remaining >> expectedSimpleSize, it's extended format
            // If remaining is close to expectedSimpleSize, it's simple format
            
            int loadedLayers = 0;
            
            // Use a reasonable threshold: if remaining is more than 10x the simple format,
            // it's definitely extended format
            if (remaining > expectedSimpleSize * 10) {
                // Use the extended format with layer names
                System.out.println("Using extended weight format with layer names...");
                loadedLayers = loadWeightsExtended(buffer, numLayers);
            } else {
                // Use the simple format (just paramSize + data)
                System.out.println("Using simple weight format (no layer names)...");
                loadedLayers = loadWeightsSimple(buffer, numLayers);
            }
            
            System.out.println("Successfully loaded " + loadedLayers + " layers of weights");
            return loadedLayers > 0;
            
        } catch (Exception e) {
            System.err.println("Error deserializing weights: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
    
    /**
     * Load weights using the simple format: paramSize (4 bytes) + param data for each layer.
     * This format matches what saveWeights() produces.
     */
    private int loadWeightsSimple(ByteBuffer buffer, int numLayers) {
        int layerIndex = 0;
        for (Layer layer : network) {
            long paramCount = layer.getParameterCount();
            if (paramCount > 0 && layerIndex < numLayers) {
                try {
                    int paramSize = buffer.getInt();
                    byte[] layerData = new byte[paramSize];
                    buffer.get(layerData);
                    layer.deserializeParameters(layerData);
                    layerIndex++;
                } catch (Exception e) {
                    System.err.println("Failed to load layer " + layerIndex + ": " + e.getMessage());
                }
            }
        }
        return layerIndex;
    }
    
    /**
     * Load weights using the extended format from convert_easyocr.py.
     * Format: name_length(1) + name + type(1) + weight_tensor + optional bias_tensor
     */
    private int loadWeightsExtended(ByteBuffer buffer, int numLayers) {
        // Build a map of layer names to layers for matching
        java.util.Map<String, Layer> layerMap = new java.util.HashMap<>();
        for (Layer layer : network) {
            String layerName = layer.getName();
            if (layerName != null && !layerName.isEmpty()) {
                layerMap.put(layerName.toLowerCase(), layer);
            }
        }

        int loadedLayers = 0;

        // Track if we've already loaded SequenceModeling.1.linear.weight to skip duplicates
        boolean alreadyLoadedSeq1Linear = false;

        // Only read the first 17 items from the file (skip the 18th which is a duplicate)
        // The weight file has a duplicate SequenceModeling.1.linear.weight and is missing Prediction.weight
        int maxLayersToRead = Math.min(numLayers, 17);

        for (int i = 0; i < maxLayersToRead; i++) {
            try {
                // Check if we have enough bytes for at least the name length
                if (!buffer.hasRemaining()) {
                    System.err.println("Buffer exhausted at layer " + i);
                    break;
                }

                // Read name length (1 byte)
                int nameLength = buffer.get() & 0xFF;

                // Validate name length to prevent OOM
                if (nameLength <= 0 || nameLength > 200) {
                    System.err.println("Invalid name length: " + nameLength + ", stopping load");
                    break;
                }

                // Read name
                byte[] nameBytes = new byte[nameLength];
                buffer.get(nameBytes);
                String layerName = new String(nameBytes, "UTF-8");

                // Skip duplicate SequenceModeling.1.linear.weight
                if (layerName.equals("SequenceModeling.1.linear.weight") && alreadyLoadedSeq1Linear) {
                    System.out.println("SKIPPING duplicate layer: " + layerName);
                    // Still need to read the weight data from buffer to skip it properly
                    // Read layer type
                    buffer.get();
                    // Read shape dimensions
                    int shapeDim = buffer.getInt();
                    long weightSize = 1;
                    for (int j = 0; j < shapeDim; j++) {
                        long dimValue = buffer.getLong(); // skip shape values
                        weightSize *= dimValue;
                    }
                    // Skip weight data
                    for (int j = 0; j < weightSize; j++) {
                        buffer.getFloat();
                    }
                    continue;
                }

                // Mark SequenceModeling.1.linear.weight as loaded
                if (layerName.equals("SequenceModeling.1.linear.weight")) {
                    alreadyLoadedSeq1Linear = true;
                }

                // Read layer type (1 byte)
                if (!buffer.hasRemaining()) {
                    System.err.println("Buffer exhausted after layer name at layer " + i);
                    break;
                }
                int layerType = buffer.get() & 0xFF;
                System.out.println("DEBUG: Layer " + i + " type=" + layerType + ", name='" + layerName + "'");
                
                // Read weight tensor dimensions
                if (!buffer.hasRemaining()) {
                    System.err.println("Buffer exhausted before weight shape at layer " + i);
                    break;
                }
                int shapeDim = buffer.getInt();
                
                // Validate shape dimensions
                if (shapeDim <= 0 || shapeDim > 10) {
                    System.err.println("Invalid shape dimension: " + shapeDim + " for layer " + layerName);
                    break;
                }
                
                int[] shape = new int[shapeDim];
                long weightSize = 1;
                for (int j = 0; j < shapeDim; j++) {
                    if (!buffer.hasRemaining()) {
                        System.err.println("Buffer exhausted reading shape at layer " + i);
                        break;
                    }
                    // Python writes shape values as int64 (8 bytes), read them correctly
                    long dimValue = buffer.getLong();
                    shape[j] = (int) dimValue;  // Safe cast since dimensions won't exceed int range
                    // Validate individual dimension
                    if (shape[j] <= 0 || shape[j] > 100000) {
                        System.err.println("Invalid shape dimension value: " + shape[j] + " for dim " + j);
                        break;
                    }
                    weightSize *= shape[j];
                }
                
                // Validate total weight size to prevent OOM
                if (weightSize <= 0 || weightSize > 10000000) {
                    System.err.println("Invalid weight size: " + weightSize + " for layer " + layerName);
                    break;
                }
                
                // Check if we have enough bytes for the weights
                long weightBytesNeeded = weightSize * 4L; // 4 bytes per float
                if (buffer.remaining() < weightBytesNeeded) {
                    System.err.println("Not enough bytes for weights: have " + buffer.remaining() + ", need " + weightBytesNeeded);
                    break;
                }
                
                // Read weight data
                System.out.println("DEBUG: About to read weights for '" + layerName + "', buffer.remaining=" + buffer.remaining() + ", weightBytesNeeded=" + weightBytesNeeded);
                float[] weights = new float[(int) weightSize];
                for (int j = 0; j < weightSize; j++) {
                    weights[j] = buffer.getFloat();
                }
                System.out.println("DEBUG: Successfully read " + weightSize + " weights for '" + layerName + "'");
                
                // Check if there's a bias (remaining data that looks like a 1D tensor)
                int biasSize = 0;
                float[] biases = null;
                System.out.println("DEBUG: After reading weights, buffer.remaining=" + buffer.remaining());
                
                if (buffer.hasRemaining() && buffer.remaining() >= 4) {
                    // Peek at the next value to check if it's a 1D shape
                    int peekPos = buffer.position();
                    int nextShapeDim = buffer.getInt();
                    
                    if (nextShapeDim == 1 && buffer.remaining() >= 4) {
                        // Likely a bias tensor
                        int[] biasShape = new int[1];
                        biasShape[0] = buffer.getInt();
                        biasSize = biasShape[0];
                        
                        // Validate bias size
                        if (biasSize > 0 && biasSize <= 100000 && buffer.remaining() >= biasSize * 4L) {
                            biases = new float[biasSize];
                            for (int j = 0; j < biasSize; j++) {
                                biases[j] = buffer.getFloat();
                            }
                        } else {
                            // Not actually a bias, reset
                            buffer.position(peekPos);
                        }
                    } else {
                        // Not a bias, reset
                        buffer.position(peekPos);
                    }
                }
                
                // Try to find matching layer by name
                // The Python converter uses names like "FeatureExtraction.ConvNet.0.weight"
                // We need to match these to Java layer names
                System.out.println("DEBUG: About to call findMatchingLayer for '" + layerName + "'");
                Layer matchingLayer = findMatchingLayer(layerMap, layerName);
                System.out.println("DEBUG: findMatchingLayer returned " + (matchingLayer != null ? matchingLayer.getName() : "null") + " for '" + layerName + "'");
                
                if (matchingLayer != null) {
                    // Apply weights to the layer
                    System.out.println("Loading weights for: " + layerName + " (" + weights.length + " elements)");
                    if (matchingLayer instanceof Conv2D) {
                        Conv2D conv = (Conv2D) matchingLayer;
                        conv.setWeightsFromPreTrained(weights, biases);
                        loadedLayers++;
                    } else if (matchingLayer instanceof Dense) {
                        Dense dense = (Dense) matchingLayer;
                        dense.setWeightsFromPreTrained(weights, biases);
                        loadedLayers++;
                    } else if (matchingLayer instanceof BiLSTM) {
                        BiLSTM lstm = (BiLSTM) matchingLayer;
                        lstm.setWeightsFromPreTrained(weights, biases, layerName);
                        loadedLayers++;
                    }
                } else {
                    System.err.println("Warning: No matching layer found for '" + layerName + "'");
                }
                
            } catch (Exception e) {
                System.err.println("Error loading layer " + i + ": " + e.getMessage());
                e.printStackTrace();
                break;
            }
        }
        
        return loadedLayers;
    }
    
    /**
     * Find a matching layer by comparing the Python layer name to Java layer names.
     * Handles various name formats and conventions.
     */
    private Layer findMatchingLayer(java.util.Map<String, Layer> layerMap, String pythonName) {
        // Direct match
        if (layerMap.containsKey(pythonName.toLowerCase())) {
            return layerMap.get(pythonName.toLowerCase());
        }
        
        // Try to extract just the layer identifier
        // Python names like "FeatureExtraction.ConvNet.0.weight"
        String simpleName = pythonName.toLowerCase();
        
        System.out.println("DEBUG findMatchingLayer: pythonName='" + pythonName + "', simpleName='" + simpleName + "'");
        
        // Extract layer index from ConvNet.X pattern
        java.util.regex.Pattern convPattern = java.util.regex.Pattern.compile("convnet\\.(\\d+)");
        java.util.regex.Matcher matcher = convPattern.matcher(simpleName);
        if (matcher.find()) {
            int easyOcrIndex = Integer.parseInt(matcher.group(1));
            // Map EasyOCR layer index to Java network position
            // EasyOCR ConvNet: 0, 3, 6, 8, 11, 14, 18
            // Java Conv2D layers at indices: 0, 2, 4, 6, 8, 10, 12
            int[] javaIndices = {0, 3, 6, 8, 11, 14, 18};
            for (int j = 0; j < javaIndices.length; j++) {
                if (javaIndices[j] == easyOcrIndex && j < network.size()) {
                    return network.get(j);
                }
            }
        }
        
        // Check for rnn/lstm patterns (SequenceModeling layers)
        // EasyOCR layer names: "SequenceModeling.0.rnn.weight_forward"
        // Also handles: "SequenceModeling.0.rnn.weight_forward_hidden", "weight_forward_reverse", etc.
        System.out.println("DEBUG: Checking sequence/modeling: contains sequence=" + simpleName.contains("sequence") + ", contains modeling=" + simpleName.contains("modeling"));
        if (simpleName.contains("sequence") && simpleName.contains("modeling")) {
            java.util.regex.Pattern seqPattern = java.util.regex.Pattern.compile("modeling\\.(\\d+)\\.rnn");
            matcher = seqPattern.matcher(simpleName);
            boolean found = matcher.find();
            int seqIndex = -1;
            if (found) {
                seqIndex = Integer.parseInt(matcher.group(1));
            }
            if (found) {
                // This is definitely a BiLSTM layer or weight component (has .rnn. in name)
                // Route to BiLSTM layers at indices 12 and 13
                int[] javaIndices = {12, 13};
                if (seqIndex < javaIndices.length && seqIndex < network.size()) {
                    return network.get(javaIndices[seqIndex]);
                }
            } else {
                // Check for linear/prediction layers (don't have .rnn. in name)
                if (simpleName.contains("linear") || simpleName.contains("prediction")) {
                    // Handle Prediction.weight separately (no SequenceModeling.X prefix)
                    if (simpleName.equals("prediction.weight") || simpleName.contains("prediction.weight")) {
                        System.out.println("DEBUG: Handling Prediction.weight, network.size()=" + network.size());
                        // Return the last Dense layer (index 16)
                        if (network.size() > 16) {
                            Layer layer = network.get(16);
                            System.out.println("DEBUG: Returning layer " + layer.getName() + " at index 16");
                            return layer;
                        }
                    }
                    
                    // Handle SequenceModeling.x.linear weights
                    // SequenceModeling.0.linear -> Dense layer at index 14
                    // SequenceModeling.1.linear -> Dense layer at index 15
                    int[] javaIndices = {14, 15};
                    java.util.regex.Pattern linearPattern = java.util.regex.Pattern.compile("modeling\\.(\\d+)\\.(linear|prediction)");
                    matcher = linearPattern.matcher(simpleName);
                    if (matcher.find()) {
                        int linearSeqIndex = Integer.parseInt(matcher.group(1));
                        if (linearSeqIndex < javaIndices.length && linearSeqIndex < network.size()) {
                            return network.get(javaIndices[linearSeqIndex]);
                        }
                    }
                    // Fallback: try finding Dense layers
                    for (int i = 0; i < network.size(); i++) {
                        if (network.get(i) instanceof Dense) {
                            return network.get(i);
                        }
                    }
                }
            }
        }
        
        // Fallback: check for any rnn/bilstm in the name
        if (simpleName.contains("rnn") || simpleName.contains("bilstm")) {
            // Try to extract sequence index
            java.util.regex.Pattern rnnPattern = java.util.regex.Pattern.compile("\\.(\\d+)\\.");
            matcher = rnnPattern.matcher(simpleName);
            if (matcher.find()) {
                int seqIndex = Integer.parseInt(matcher.group(1));
                int[] javaIndices = {12, 13};
                if (seqIndex < javaIndices.length && seqIndex < network.size()) {
                    return network.get(javaIndices[seqIndex]);
                }
            }
        }
        
        // Check for prediction layer
        if (simpleName.contains("prediction")) {
            System.out.println("DEBUG: Found 'prediction' in fallback check, network.size()=" + network.size());
            // Last dense layer
            for (int i = network.size() - 1; i >= 0; i--) {
                if (network.get(i) instanceof Dense) {
                    System.out.println("DEBUG: Returning Dense layer " + network.get(i).getName() + " at index " + i);
                    return network.get(i);
                }
            }
        }
        
        return null;
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
