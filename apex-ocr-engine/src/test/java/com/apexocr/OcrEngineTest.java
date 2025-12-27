package com.apexocr;

import com.apexocr.core.tensor.Tensor;
import com.apexocr.core.tensor.TensorOperations;
import com.apexocr.core.neural.Dense;
import com.apexocr.core.neural.Layer;
import com.apexocr.core.ctc.CTCDecoder;
import com.apexocr.preprocessing.ImagePreprocessor;
import com.apexocr.engine.OcrEngine;
import com.apexocr.engine.OcrResult;

import org.junit.Test;
import static org.junit.Assert.*;

import java.awt.image.BufferedImage;

/**
 * Unit tests for the ApexJavaOCR engine.
 * Tests core tensor operations, neural network layers, preprocessing,
 * and the main OCR pipeline.
 */
public class OcrEngineTest {

    // ==================== Tensor Tests ====================

    @Test
    public void testTensorCreation() {
        long[] shape = new long[]{2, 3, 4};
        Tensor tensor = new Tensor(shape, Tensor.DataType.FLOAT32);

        assertEquals("Tensor rank should be 3", 3, tensor.getRank());
        assertEquals("Tensor size should be 24", 24, tensor.getSize());
        assertArrayEquals("Tensor shape should match", shape, tensor.getShape());
        assertEquals("Tensor data type should be FLOAT32", Tensor.DataType.FLOAT32, tensor.getDataType());
    }

    @Test
    public void testTensorFloatOperations() {
        Tensor tensor = new Tensor(new long[]{10}, Tensor.DataType.FLOAT32);

        for (long i = 0; i < 10; i++) {
            tensor.setFloat(i, (float) i * 1.5f);
        }

        for (long i = 0; i < 10; i++) {
            assertEquals((float) i * 1.5f, tensor.getFloat(i), 0.0001f);
        }
    }

    @Test
    public void testTensorMultiDimensionalAccess() {
        Tensor tensor = new Tensor(new long[]{3, 4, 5}, Tensor.DataType.FLOAT32);

        // Set value at specific coordinates
        tensor.setFloat(1.5f, 1, 2, 3);

        // Retrieve using same coordinates
        assertEquals(1.5f, tensor.getFloat(1, 2, 3), 0.0001f);

        // Test linear index conversion
        long linearIndex = tensor.linearIndex(new long[]{1, 2, 3});
        assertEquals(1.5f, tensor.getFloat(linearIndex), 0.0001f);
    }

    @Test
    public void testTensorFill() {
        Tensor tensor = new Tensor(new long[]{100}, Tensor.DataType.FLOAT32);

        tensor.fill(42.0f);

        for (long i = 0; i < 100; i++) {
            assertEquals(42.0f, tensor.getFloat(i), 0.0001f);
        }
    }

    @Test
    public void testTensorCopyAndView() {
        Tensor original = new Tensor(new long[]{5, 5}, Tensor.DataType.FLOAT32);
        for (long i = 0; i < 25; i++) {
            original.setFloat(i, i);
        }

        Tensor copy = original.copy();
        Tensor view = original.view();

        // Modify copy
        copy.setFloat(0, 999);

        // Original should be unchanged
        assertEquals(0, original.getFloat(0), 0.0001f);

        // Copy should be changed
        assertEquals(999, copy.getFloat(0), 0.0001f);

        // View shares memory
        view.setFloat(0, 888);
        assertEquals(888, original.getFloat(0), 0.0001f);
    }

    // ==================== TensorOperations Tests ====================

    @Test
    public void testElementWiseOperations() {
        Tensor a = new Tensor(new long[]{10}, Tensor.DataType.FLOAT32);
        Tensor b = new Tensor(new long[]{10}, Tensor.DataType.FLOAT32);

        for (long i = 0; i < 10; i++) {
            a.setFloat(i, (float) i);
            b.setFloat(i, (float) (10 - i));
        }

        // Test addition
        Tensor sum = TensorOperations.add(a, b);
        for (long i = 0; i < 10; i++) {
            assertEquals(10.0f, sum.getFloat(i), 0.0001f);
        }

        // Test multiplication
        Tensor product = TensorOperations.multiply(a, b);
        for (long i = 0; i < 10; i++) {
            assertEquals(i * (10 - i), product.getFloat(i), 0.0001f);
        }
    }

    @Test
    public void testMatrixMultiplication() {
        // Create matrices
        Tensor a = new Tensor(new long[]{2, 3}, Tensor.DataType.FLOAT32);
        Tensor b = new Tensor(new long[]{3, 4}, Tensor.DataType.FLOAT32);

        // Set values
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                a.setFloat((float)(i + j), i, j);
            }
        }

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                b.setFloat((float)(i * j), i, j);
            }
        }

        // Compute product
        Tensor result = TensorOperations.matmul(a, b);

        // Verify shape
        assertEquals("Result rows should be 2", 2, result.getShape()[0]);
        assertEquals("Result cols should be 4", 4, result.getShape()[1]);

        // Verify some values
        assertEquals("Element (0,0) should be 0", 0, result.getFloat(0, 0), 0.0001f);
        assertEquals("Element (0,1) should be 3", 3, result.getFloat(0, 1), 0.0001f);
    }

    @Test
    public void testReLUActivation() {
        Tensor input = new Tensor(new long[]{5}, Tensor.DataType.FLOAT32);
        input.setFloat(-2.0f, 0);
        input.setFloat(-1.0f, 1);
        input.setFloat(0.0f, 2);
        input.setFloat(1.0f, 3);
        input.setFloat(2.0f, 4);

        // Test ReLU
        Tensor relu = TensorOperations.relu(input);
        assertEquals("ReLU of -2 should be 0", 0, relu.getFloat(0), 0.0001f);
        assertEquals("ReLU of -1 should be 0", 0, relu.getFloat(1), 0.0001f);
        assertEquals("ReLU of 0 should be 0", 0, relu.getFloat(2), 0.0001f);
        assertEquals("ReLU of 1 should be 1", 1, relu.getFloat(3), 0.0001f);
        assertEquals("ReLU of 2 should be 2", 2, relu.getFloat(4), 0.0001f);
    }

    @Test
    public void testSigmoidActivation() {
        Tensor input = new Tensor(new long[]{5}, Tensor.DataType.FLOAT32);
        input.setFloat(-2.0f, 0);
        input.setFloat(0.0f, 1);
        input.setFloat(2.0f, 2);

        // Test Sigmoid
        Tensor sigmoid = TensorOperations.sigmoid(input);
        assertTrue("Sigmoid of -2 should be between 0 and 0.2",
            sigmoid.getFloat(0) > 0 && sigmoid.getFloat(0) < 0.2);
        assertTrue("Sigmoid of 0 should be around 0.5",
            sigmoid.getFloat(1) > 0.4 && sigmoid.getFloat(1) < 0.6);
        assertTrue("Sigmoid of 2 should be between 0.8 and 1.0",
            sigmoid.getFloat(2) > 0.8 && sigmoid.getFloat(2) < 1.0);
    }

    @Test
    public void testSoftmax() {
        Tensor input = new Tensor(new long[]{1, 3}, Tensor.DataType.FLOAT32);
        input.setFloat(1.0f, 0, 0);
        input.setFloat(2.0f, 0, 1);
        input.setFloat(3.0f, 0, 2);

        Tensor softmax = TensorOperations.softmax(input);

        // Check probabilities sum to 1
        float sum = 0;
        for (int i = 0; i < 3; i++) {
            sum += softmax.getFloat(0, i);
        }
        assertEquals("Softmax probabilities should sum to 1", 1.0f, sum, 0.0001f);

        // Check probabilities are in valid range
        for (int i = 0; i < 3; i++) {
            assertTrue("Probability should be in [0, 1]",
                softmax.getFloat(0, i) >= 0 && softmax.getFloat(0, i) <= 1);
        }
    }

    // ==================== Neural Network Layer Tests ====================

    @Test
    public void testDenseLayer() {
        Dense layer = new Dense(10, Dense.ActivationType.RELU, true);
        layer.setInputShape(new long[]{-1, 100});
        layer.initialize(Layer.Initializer.RANDOM_NORMAL);

        Tensor input = new Tensor(new long[]{1, 100}, Tensor.DataType.FLOAT32);
        input.randomNormal(0, 0.1f);

        Tensor output = layer.forward(input, false);

        assertEquals("Dense layer output should have 10 units", 10, output.getShape()[1]);
        assertNotNull("Output should not be null", output);
    }

    @Test
    public void testDenseLayerActivations() {
        Dense.ActivationType[] activations = {
            Dense.ActivationType.RELU,
            Dense.ActivationType.SIGMOID,
            Dense.ActivationType.TANH
        };

        for (Dense.ActivationType activation : activations) {
            Dense layer = new Dense(5, activation, true);
            layer.setInputShape(new long[]{-1, 10});
            layer.initialize(Layer.Initializer.RANDOM_NORMAL);

            Tensor input = new Tensor(new long[]{1, 10}, Tensor.DataType.FLOAT32);
            input.randomUniform(-1, 1);

            Tensor output = layer.forward(input, false);

            assertEquals("Dense layer output should have 5 units", 5, output.getShape()[1]);
        }
    }

    // ==================== CTC Decoder Tests ====================

    @Test
    public void testCTCDecoder() {
        String[] labels = {"", "a", "b", "c", "d"};
        CTCDecoder decoder = CTCDecoder.createDefault(labels);

        // Create a simple probability matrix
        Tensor input = new Tensor(new long[]{5, 5}, Tensor.DataType.FLOAT32);

        // Set high probability for 'a' at steps 0,1, then blank at 2, then 'b' at 3,4
        input.setFloat(0.1f, 0, 0); // blank
        input.setFloat(0.8f, 0, 1); // 'a'
        input.setFloat(0.1f, 0, 2); // 'b'

        input.setFloat(0.1f, 1, 0); // blank
        input.setFloat(0.7f, 1, 1); // 'a'
        input.setFloat(0.2f, 1, 2); // 'b'

        input.setFloat(0.9f, 2, 0); // blank
        input.setFloat(0.05f, 2, 1); // 'a'
        input.setFloat(0.05f, 2, 2); // 'b'

        input.setFloat(0.1f, 3, 0); // blank
        input.setFloat(0.1f, 3, 1); // 'a'
        input.setFloat(0.8f, 3, 2); // 'b'

        input.setFloat(0.1f, 4, 0); // blank
        input.setFloat(0.1f, 4, 1); // 'a'
        input.setFloat(0.8f, 4, 2); // 'b'

        String result = decoder.decode(input);

        // Should decode to "ab"
        assertEquals("CTC decoder should output 'ab'", "ab", result);
    }

    @Test
    public void testCTCGreedyDecoder() {
        String[] labels = {"", "a", "b", "c"};
        CTCDecoder decoder = CTCDecoder.createDefault(labels);

        Tensor input = new Tensor(new long[]{4, 4}, Tensor.DataType.FLOAT32);

        // Set clear pattern: a, a, blank, b
        input.setFloat(0.1f, 0, 0);
        input.setFloat(0.8f, 0, 1);
        input.setFloat(0.05f, 0, 2);
        input.setFloat(0.05f, 0, 3);

        input.setFloat(0.1f, 1, 0);
        input.setFloat(0.8f, 1, 1);
        input.setFloat(0.05f, 1, 2);
        input.setFloat(0.05f, 1, 3);

        input.setFloat(0.8f, 2, 0);
        input.setFloat(0.1f, 2, 1);
        input.setFloat(0.05f, 2, 2);
        input.setFloat(0.05f, 2, 3);

        input.setFloat(0.1f, 3, 0);
        input.setFloat(0.1f, 3, 1);
        input.setFloat(0.05f, 3, 2);
        input.setFloat(0.8f, 3, 3);

        String result = decoder.decodeGreedy(input);

        // Should produce "ab"
        assertEquals("Greedy decoder should output 'ab'", "ab", result);
    }

    // ==================== Image Preprocessing Tests ====================

    @Test
    public void testGrayscaleConversion() {
        BufferedImage colorImage = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB);

        // Create a colored image
        for (int y = 0; y < 100; y++) {
            for (int x = 0; x < 100; x++) {
                int r = x % 256;
                int g = y % 256;
                int b = (x + y) % 256;
                colorImage.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        }

        ImagePreprocessor preprocessor = new ImagePreprocessor();
        BufferedImage grayImage = preprocessor.toGrayscale(colorImage);

        assertEquals("Grayscale image should be TYPE_BYTE_GRAY",
            BufferedImage.TYPE_BYTE_GRAY, grayImage.getType());

        // Verify conversion is reasonable
        for (int y = 0; y < 10; y++) {
            for (int x = 0; x < 10; x++) {
                int gray = grayImage.getRaster().getSample(x, y, 0);
                assertTrue("Grayscale value should be in valid range", gray >= 0 && gray <= 255);
            }
        }
    }

    @Test
    public void testBinarizationOtsu() {
        BufferedImage image = new BufferedImage(50, 50, BufferedImage.TYPE_BYTE_GRAY);

        // Create gradient image
        for (int y = 0; y < 50; y++) {
            for (int x = 0; x < 50; x++) {
                int value = (x + y) * 2 % 256;
                image.getRaster().setSample(x, y, 0, value);
            }
        }

        ImagePreprocessor preprocessor = new ImagePreprocessor();

        // Test Otsu
        BufferedImage otsu = preprocessor.binarizeOtsu(image);
        assertEquals("Otsu binarization should produce TYPE_BYTE_BINARY",
            BufferedImage.TYPE_BYTE_BINARY, otsu.getType());
    }

    @Test
    public void testBinarizationFixed() {
        BufferedImage image = new BufferedImage(50, 50, BufferedImage.TYPE_BYTE_GRAY);

        // Fill with known values
        for (int y = 0; y < 50; y++) {
            for (int x = 0; x < 50; x++) {
                image.getRaster().setSample(x, y, 0, 100);
            }
        }

        ImagePreprocessor preprocessor = new ImagePreprocessor();

        // Test Fixed threshold
        BufferedImage binary = preprocessor.binarizeFixed(image, 128);
        assertEquals("Fixed threshold should produce TYPE_BYTE_BINARY",
            BufferedImage.TYPE_BYTE_BINARY, binary.getType());

        // All pixels should be 0 (black) since 100 < 128
        int sample = binary.getRaster().getSample(0, 0, 0);
        assertEquals("Pixels below threshold should be black (0)", 0, sample);
    }

    // ==================== OCR Engine Tests ====================

    @Test
    public void testEngineInitialization() {
        OcrEngine engine = new OcrEngine();
        assertFalse("Engine should not be initialized initially", engine.isInitialized());

        engine.initialize();
        assertTrue("Engine should be initialized after initialize()", engine.isInitialized());

        assertTrue("Engine should have at least one layer", engine.getLayerCount() > 0);
        // Note: Parameter count is 0 until layers are initialized during first forward pass
        // This is by design for lazy initialization

        engine.close();
    }

    @Test
    public void testEngineConfigurations() {
        // Test default config
        OcrEngine engine1 = new OcrEngine(OcrEngine.EngineConfig.createDefault());
        engine1.initialize();
        assertTrue("Engine with default config should initialize", engine1.isInitialized());
        engine1.close();

        // Test accuracy config
        OcrEngine engine2 = new OcrEngine(OcrEngine.EngineConfig.forHighAccuracy());
        engine2.initialize();
        assertTrue("Engine with accuracy config should initialize", engine2.isInitialized());
        engine2.close();

        // Test speed config
        OcrEngine engine3 = new OcrEngine(OcrEngine.EngineConfig.forSpeed());
        engine3.initialize();
        assertTrue("Engine with speed config should initialize", engine3.isInitialized());
        engine3.close();
    }

    @Test
    public void testOcrResult() {
        OcrResult result = new OcrResult("Hello World", 0.95f, 150, 640, 480);

        assertEquals("Text should match", "Hello World", result.getText());
        assertEquals("Confidence should match", 0.95f, result.getConfidence(), 0.001f);
        assertEquals("Confidence percent should be 95", 95.0f, result.getConfidencePercent(), 0.1f);
        assertEquals("Processing time should match", 150, result.getProcessingTimeMs());
        assertEquals("Word count should be 2", 2, result.getWordCount());
        assertEquals("Character count should be 11", 11, result.getCharacterCount());
    }

    @Test
    public void testOcrResultBuilder() {
        OcrResult result = new OcrResult.Builder()
            .setText("Test text")
            .setConfidence(0.85f)
            .setProcessingTime(100)
            .setImageDimensions(800, 600)
            .addRegion(new OcrResult.TextRegion(10, 20, 100, 50, 0.9f))
            .addWord(new OcrResult.Word("Test", 10, 20, 50, 30, 0.95f))
            .build();

        assertEquals("Text should match", "Test text", result.getText());
        assertEquals("Confidence should match", 0.85f, result.getConfidence(), 0.001f);
        assertEquals("Should have 1 region", 1, result.getRegions().size());
        assertEquals("Should have 1 word", 1, result.getWords().size());
    }

    // ==================== Integration Tests ====================

    @Test
    public void testFullPreprocessingPipeline() {
        BufferedImage image = new BufferedImage(200, 100, BufferedImage.TYPE_INT_RGB);

        // Create a test pattern
        for (int y = 0; y < 100; y++) {
            for (int x = 0; x < 200; x++) {
                int gray = (x + y) % 256;
                image.setRGB(x, y, (gray << 16) | (gray << 8) | gray);
            }
        }

        ImagePreprocessor.PreprocessingConfig config =
            ImagePreprocessor.PreprocessingConfig.createDefault();
        ImagePreprocessor preprocessor = new ImagePreprocessor(config);

        BufferedImage processed = preprocessor.process(image);

        // Verify output properties
        assertEquals("Processed image height should be 32", 32, processed.getHeight());
        assertTrue("Processed image width should be >= 32", processed.getWidth() >= 32);
    }
}
