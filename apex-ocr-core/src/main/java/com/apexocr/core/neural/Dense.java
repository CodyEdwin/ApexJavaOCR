package com.apexocr.core.neural;

import com.apexocr.core.tensor.Tensor;
import com.apexocr.core.tensor.TensorOperations;
import com.apexocr.core.tensor.MemoryManager;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Dense - Fully connected layer implementation for neural network operations.
 * Also known as a dense layer or linear layer, this component computes
 * output = activation(input * weights + bias) for each element.
 *
 * This implementation is optimized for high-performance inference on the JVM,
 * utilizing efficient memory access patterns and parallel computation.
 *
 * @author ApexOCR Team
 * @version 1.0.0
 */
public class Dense implements Layer {
    private String name;
    private final int units;
    private final ActivationType activation;
    private final boolean useBias;

    private Tensor weights;
    private Tensor bias;
    private long[] inputShape;
    private long[] outputShape;

    private boolean training;
    private boolean initialized;
    private Tensor inputCache;
    private Tensor outputCache;

    /**
     * Enumeration of supported activation functions.
     */
    public enum ActivationType {
        LINEAR,
        RELU,
        LEAKY_RELU,
        SIGMOID,
        TANH,
        SOFTMAX,
        SWISH,
        MISH
    }

    /**
     * Creates a new dense layer.
     *
     * @param units The number of output units
     * @param activation The activation function to use
     * @param useBias Whether to include bias terms
     */
    public Dense(int units, ActivationType activation, boolean useBias) {
        this.name = "dense_" + units;
        this.units = units;
        this.activation = activation;
        this.useBias = useBias;
        this.training = false;
    }

    /**
     * Creates a new dense layer with specified input units.
     *
     * @param units The number of output units
     * @param inputUnits The number of input units (for pre-trained weight loading)
     * @param activation The activation function to use
     * @param useBias Whether to include bias terms
     */
    public Dense(int units, int inputUnits, ActivationType activation, boolean useBias) {
        this(units, activation, useBias);
        this.name = "dense_" + units;
        // Set up input shape for pre-trained weight validation
        // But don't initialize weights - they will be loaded from pre-trained model
        if (inputUnits > 0 && units > 0) {
            this.inputShape = new long[]{inputUnits};
            this.outputShape = new long[]{units};
            this.initialized = true;
        }
    }

    /**
     * Creates a dense layer with default settings (ReLU activation, with bias).
     *
     * @param units The number of output units
     */
    public Dense(int units) {
        this(units, ActivationType.RELU, true);
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public void setName(String name) {
        this.name = name;
    }

    @Override
    public LayerType getType() {
        return LayerType.DENSE;
    }

    @Override
    public long[] getInputShape() {
        return inputShape != null ? inputShape.clone() : null;
    }

    @Override
    public long[] getOutputShape() {
        return outputShape != null ? outputShape.clone() : null;
    }

    @Override
    public long getParameterCount() {
        if (inputShape == null) return 0;

        long inputSize = 1;
        for (int i = 0; i < inputShape.length; i++) {
            inputSize *= inputShape[i];
        }

        long weightParams = inputSize * units;
        long biasParams = useBias ? units : 0;

        return weightParams + biasParams;
    }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        this.training = training;
        return forward(input);
    }

    @Override
    public Tensor forward(Tensor input) {
        // Cache input for potential gradient computation
        if (training) {
            inputCache = input.copy();
        }

        long[] inputShape = input.getShape();
        long batchSize = inputShape[0];

        // Lazy initialization on first forward pass (only if weights not already set)
        // Also preserve 1D inputShape from constructor (used for pre-trained models)
        // Only overwrite inputShape if it's null or 1D (not if it's 3D from warm-up)
        if (weights == null || weights.getSize() == 0) {
            // Only set inputShape if it's null (never set) or 1D (from constructor)
            // Don't overwrite 3D inputShape from previous warm-up runs
            if (this.inputShape == null || this.inputShape.length == 1) {
                setInputShape(inputShape);
            }

            // For 3D input [batch, timeSteps, features], initialize weights for features dimension only
            // For 2D input [batch, features], use the full input size
            if (inputShape.length == 3) {
                // Create weights tensor with shape [features, units]
                long features = inputShape[2];
                this.weights = new Tensor(new long[]{features, units}, Tensor.DataType.FLOAT32);
                initializeWeightsTensor(this.weights);

                // Create bias if needed
                if (useBias) {
                    this.bias = new Tensor(new long[]{units}, Tensor.DataType.FLOAT32);
                    this.bias.fill(0);
                }
            } else {
                // Original initialization for 2D input
                initialize(Initializer.HE_NORMAL);
            }
        }

        // Handle 3D input [batch, timeSteps, features] from BiLSTM
        // Apply Dense transformation to each time step independently
        if (inputShape.length == 3) {
            long timeSteps = inputShape[1];
            long features = inputShape[2];

            // Output shape: [batch, timeSteps, units]
            outputShape = new long[]{batchSize, timeSteps, units};
            Tensor output = new Tensor(outputShape, Tensor.DataType.FLOAT32);

            // Apply Dense to each time step
            for (long t = 0; t < timeSteps; t++) {
                // Extract time step t: [batch, features]
                Tensor timeStepInput = new Tensor(new long[]{batchSize, features}, Tensor.DataType.FLOAT32);
                for (long b = 0; b < batchSize; b++) {
                    for (long f = 0; f < features; f++) {
                        timeStepInput.setFloat(input.getFloat(b, t, f), b, f);
                    }
                }

                // Compute: output = input * weights + bias
                Tensor timeStepOutput = TensorOperations.matmul(timeStepInput, weights);

                if (useBias) {
                    for (long b = 0; b < batchSize; b++) {
                        for (int u = 0; u < units; u++) {
                            float val = timeStepOutput.getFloat(b, u) + bias.getFloat(u);
                            timeStepOutput.setFloat(val, b, u);
                        }
                    }
                }

                // Copy to output
                for (long b = 0; b < batchSize; b++) {
                    for (int u = 0; u < units; u++) {
                        output.setFloat(timeStepOutput.getFloat(b, u), b, t, u);
                    }
                }

                timeStepInput.close();
                timeStepOutput.close();
            }

            // Apply activation
            Tensor activated = applyActivation(output);

            // Cache output for training
            if (training) {
                outputCache = activated;
            }

            return activated;
        }

        // Handle 2D input [batch, inputSize] - original behavior
        long inputSize = 1;
        for (int i = 1; i < inputShape.length; i++) {
            inputSize *= inputShape[i];
        }

        // Reshape input to [batch, inputSize]
        Tensor inputFlat = input.reshape(batchSize, inputSize);

        // Compute: output = input * weights + bias
        Tensor output = TensorOperations.matmul(inputFlat, weights);

        if (useBias) {
            // Broadcast bias across batch
            for (long b = 0; b < batchSize; b++) {
                for (int u = 0; u < units; u++) {
                    float val = output.getFloat(b, u) + bias.getFloat(u);
                    output.setFloat(val, b, u);
                }
            }
        }

        // Apply activation
        Tensor activated = applyActivation(output);

        // Cache output for training
        if (training) {
            outputCache = activated;
        }

        return activated;
    }

    /**
     * Applies the specified activation function.
     */
    private Tensor applyActivation(Tensor input) {
        switch (activation) {
            case LINEAR:
                return input;
            case RELU:
                return TensorOperations.relu(input);
            case LEAKY_RELU:
                return TensorOperations.leakyRelu(input, 0.01f);
            case SIGMOID:
                return TensorOperations.sigmoid(input);
            case TANH:
                return TensorOperations.tanh(input);
            case SOFTMAX:
                return TensorOperations.softmax(input);
            case SWISH:
                return swish(input);
            case MISH:
                return mish(input);
            default:
                throw new IllegalArgumentException("Unsupported activation: " + activation);
        }
    }

    /**
     * Swish activation: x * sigmoid(x)
     */
    private Tensor swish(Tensor input) {
        Tensor sigmoid = TensorOperations.sigmoid(input);
        return TensorOperations.multiply(input, sigmoid);
    }

    /**
     * Mish activation: x * tanh(softplus(x))
     */
    private Tensor mish(Tensor input) {
        Tensor softplus = softplus(input);
        Tensor tanh = TensorOperations.tanh(softplus);
        return TensorOperations.multiply(input, tanh);
    }

    /**
     * Softplus activation: log(1 + exp(x))
     */
    private Tensor softplus(Tensor input) {
        long size = input.getSize();
        Tensor result = new Tensor(input.getShape(), Tensor.DataType.FLOAT32);

        for (long i = 0; i < size; i++) {
            float x = input.getFloat(i);
            result.setFloat(i, (float) Math.log(1.0 + Math.exp(x)));
        }

        return result;
    }

    @Override
    public void resetState() {
        inputCache = null;
        outputCache = null;
    }

    @Override
    public void eval() {
        this.training = false;
    }

    @Override
    public void train() {
        this.training = true;
    }

    @Override
    public boolean isTraining() {
        return training;
    }

    @Override
    public Tensor getWeights() {
        return weights;
    }

    @Override
    public Tensor getBiases() {
        return bias;
    }

    /**
     * Gets the activation type for this layer.
     * Used by external classes to determine initialization strategy.
     *
     * @return The activation type
     */
    public ActivationType getActivation() {
        return activation;
    }

    @Override
    public void setWeights(Tensor weights) {
        this.weights = weights;
    }

    @Override
    public void setBiases(Tensor biases) {
        this.bias = biases;
    }

    /**
     * Sets weights from pre-trained model (EasyOCR format).
     * EasyOCR/PyTorch stores Dense weights as [output, input].
     * Java expects weights as [input, output] for matmul(input, weights).
     * This method handles the conversion correctly.
     *
     * @param weights Flattened weight array from pre-trained model
     * @param biases Flattened bias array from pre-trained model (can be null)
     */
    public void setWeightsFromPreTrained(float[] weights, float[] biases) {
        if (weights == null || weights.length == 0) {
            return;
        }

        // Get expected dimensions from current layer configuration
        int expectedInputSize;
        int expectedOutputSize = this.units;

        if (inputShape != null && inputShape.length >= 1) {
            // Only trust inputShape if it's a 1D array from the constructor [inputUnits]
            // Don't trust 3D inputShape from runtime warm-up [batch, timeSteps, features]
            if (inputShape.length == 1) {
                // Use the constructor-configured input size
                expectedInputSize = (int) inputShape[0];
            } else {
                // inputShape is from runtime (3D), infer input size from weight array
                // Weight format from EasyOCR: [output, input] = total elements
                expectedInputSize = weights.length / expectedOutputSize;
            }
        } else {
            // Infer input units from weight array: input = total / output
            // Weight format from EasyOCR: [output, input] = total elements
            expectedInputSize = weights.length / expectedOutputSize;
        }

        int expectedTotalWeights = expectedInputSize * expectedOutputSize;

        // Validate that weights array size matches expected dimensions
        if (weights.length != expectedTotalWeights) {
            throw new IllegalArgumentException(
                String.format("Weight array size mismatch: expected %d elements (input=%d, output=%d), got %d elements. " +
                              "This usually means the BiLSTM output dimension (%d) doesn't match the Dense layer input expectation.",
                              expectedTotalWeights, expectedInputSize, expectedOutputSize, weights.length, expectedInputSize));
        }

        // EasyOCR stores weights as [output, input] (PyTorch convention)
        // We need to convert to Java's [input, output] format for matmul
        // Reuse existing weights tensor if dimensions match, otherwise create new one
        if (this.weights == null || this.weights.getShape()[0] != expectedInputSize ||
            this.weights.getShape()[1] != expectedOutputSize) {
            this.weights = new Tensor(new long[]{expectedInputSize, expectedOutputSize}, Tensor.DataType.FLOAT32);
        }

        // Copy weights, transposing from [output, input] to [input, output]
        for (int out = 0; out < expectedOutputSize; out++) {
            for (int in = 0; in < expectedInputSize; in++) {
                // EasyOCR: [output, input] -> flat index = out * inputSize + in
                // Java: [input, output] -> flat index = in * outputSize + out
                float val = weights[out * expectedInputSize + in];
                this.weights.setFloat(val, in, out);
            }
        }

        // Set bias if provided
        if (biases != null && biases.length >= expectedOutputSize && useBias) {
            if (this.bias == null || this.bias.getShape()[0] != expectedOutputSize) {
                this.bias = new Tensor(new long[]{expectedOutputSize}, Tensor.DataType.FLOAT32);
            }
            for (int i = 0; i < expectedOutputSize; i++) {
                this.bias.setFloat(biases[i], i);
            }
        } else if (useBias) {
            // Initialize zero bias
            if (this.bias == null || this.bias.getShape()[0] != expectedOutputSize) {
                this.bias = new Tensor(new long[]{expectedOutputSize}, Tensor.DataType.FLOAT32);
            }
            this.bias.fill(0);
        }

        // Update output shape for 3D input
        if (inputShape != null && inputShape.length == 3) {
            outputShape = new long[]{inputShape[0], inputShape[1], expectedOutputSize};
        } else {
            outputShape = new long[]{-1, expectedOutputSize};
        }
    }

    @Override
    public void initialize(Initializer initializer) {
        if (inputShape == null) {
            throw new IllegalStateException("Input shape must be set before initialization");
        }

        long inputSize = 1;
        // Skip batch dimension (first dimension) when calculating input size
        for (int i = 1; i < inputShape.length; i++) {
            inputSize *= inputShape[i];
        }

        weights = new Tensor(new long[]{inputSize, units}, Tensor.DataType.FLOAT32);
        initializeWeights(initializer);

        if (useBias) {
            bias = new Tensor(new long[]{units}, Tensor.DataType.FLOAT32);
            initializeBias(initializer);
        }

        outputShape = new long[]{-1, units};
    }

    /**
     * Initializes weights according to the specified method.
     */
    private void initializeWeights(Initializer initializer) {
        switch (initializer) {
            case ZEROS:
                weights.fill(0);
                break;
            case ONES:
                weights.fill(1);
                break;
            case CONSTANT:
                weights.fill(0.01f);
                break;
            case RANDOM_UNIFORM:
                weights.randomUniform(-0.1f, 0.1f);
                break;
            case RANDOM_NORMAL:
                weights.randomNormal(0, 0.02f);
                break;
            case XAVIER_UNIFORM:
                xavierUniform(weights);
                break;
            case XAVIER_NORMAL:
                xavierNormal(weights);
                break;
            case HE_UNIFORM:
                heUniform(weights);
                break;
            case HE_NORMAL:
                heNormal(weights);
                break;
            case ORTHOGONAL:
                orthogonal(weights);
                break;
        }
    }

    /**
     * Helper method to initialize a weights tensor directly with He normal initialization.
     * Used for 3D input handling where we need to initialize based on features dimension only.
     */
    private void initializeWeightsTensor(Tensor tensor) {
        long fanIn = tensor.getShape()[0];
        float std = (float) Math.sqrt(2.0 / fanIn);
        tensor.randomNormal(0, std);
    }

    /**
     * Initializes bias according to the specified method.
     */
    private void initializeBias(Initializer initializer) {
        if (!useBias) return;

        switch (initializer) {
            case ZEROS:
                bias.fill(0);
                break;
            case ONES:
                bias.fill(1);
                break;
            case CONSTANT:
                bias.fill(0.01f);
                break;
            case RANDOM_UNIFORM:
            case RANDOM_NORMAL:
                bias.fill(0);
                break;
            case XAVIER_UNIFORM:
            case XAVIER_NORMAL:
            case HE_UNIFORM:
            case HE_NORMAL:
            case ORTHOGONAL:
                bias.fill(0);
                break;
        }
    }

    /**
     * Xavier/Glorot uniform initialization.
     */
    private void xavierUniform(Tensor tensor) {
        long fanIn = tensor.getShape()[0];
        long fanOut = tensor.getShape()[1];
        float limit = (float) Math.sqrt(6.0 / (fanIn + fanOut));
        tensor.randomUniform(-limit, limit);
    }

    /**
     * Xavier/Glorot normal initialization.
     */
    private void xavierNormal(Tensor tensor) {
        long fanIn = tensor.getShape()[0];
        long fanOut = tensor.getShape()[1];
        float std = (float) Math.sqrt(2.0 / (fanIn + fanOut));
        tensor.randomNormal(0, std);
    }

    /**
     * He/Kaiming uniform initialization.
     */
    private void heUniform(Tensor tensor) {
        long fanIn = tensor.getShape()[0];
        float limit = (float) Math.sqrt(6.0 / fanIn);
        tensor.randomUniform(-limit, limit);
    }

    /**
     * He/Kaiming normal initialization.
     */
    private void heNormal(Tensor tensor) {
        long fanIn = tensor.getShape()[0];
        float std = (float) Math.sqrt(2.0 / fanIn);
        tensor.randomNormal(0, std);
    }

    /**
     * Orthogonal initialization for recurrent layers.
     */
    private void orthogonal(Tensor tensor) {
        long[] shape = tensor.getShape();
        int rows = (int) shape[0];
        int cols = (int) shape[1];

        // Simplified orthogonal initialization
        tensor.randomNormal(0, 0.02f);
        normalizeOrthogonal(tensor, rows, cols);
    }

    /**
     * Normalizes a matrix to be orthogonal.
     */
    private void normalizeOrthogonal(Tensor tensor, int rows, int cols) {
        // Simplified: just normalize rows for now
        for (int i = 0; i < rows; i++) {
            float norm = 0;
            for (int j = 0; j < cols; j++) {
                norm += tensor.getFloat(i, j) * tensor.getFloat(i, j);
            }
            norm = (float) Math.sqrt(norm) + 1e-6f;
            for (int j = 0; j < cols; j++) {
                tensor.setFloat(tensor.getFloat(i, j) / norm, i, j);
            }
        }
    }

    /**
     * Sets the input shape for this layer.
     *
     * @param inputShape The expected input shape
     */
    public void setInputShape(long[] inputShape) {
        this.inputShape = inputShape.clone();
        this.outputShape = new long[]{-1, units};
    }

    @Override
    public byte[] serializeParameters() {
        if (weights == null) {
            return new byte[0];
        }
        
        // Format: [weights data][bias data]
        long[] weightsShape = weights.getShape();
        long[] biasShape = bias != null ? bias.getShape() : new long[0];
        
        int weightsSizeBytes = (int) (weights.getSize() * Float.BYTES);
        int biasSizeBytes = (int) (bias != null ? bias.getSize() * Float.BYTES : 0);
        
        int totalBytes = 4 + weightsShape.length * 8 + weightsSizeBytes + 
                         4 + biasShape.length * 8 + biasSizeBytes;
        
        java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocate(totalBytes);
        buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN);
        
        // Write weights shape
        buffer.putInt(weightsShape.length);
        for (long dim : weightsShape) {
            buffer.putLong(dim);
        }
        
        // Write weights data
        for (long i = 0; i < weights.getSize(); i++) {
            buffer.putFloat(weights.getFloat(i));
        }
        
        // Write bias shape
        buffer.putInt(biasShape.length);
        for (long dim : biasShape) {
            buffer.putLong(dim);
        }
        
        // Write bias data
        if (bias != null) {
            for (long i = 0; i < bias.getSize(); i++) {
                buffer.putFloat(bias.getFloat(i));
            }
        }
        
        return buffer.array();
    }

    @Override
    public void deserializeParameters(byte[] data) {
        if (data == null || data.length == 0) {
            return;
        }
        
        ByteBuffer buffer = ByteBuffer.wrap(data);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        // Read weights shape
        int weightsShapeLen = buffer.getInt();
        long[] weightsShape = new long[weightsShapeLen];
        for (int i = 0; i < weightsShapeLen; i++) {
            weightsShape[i] = buffer.getLong();
        }
        
        // Read weights data (2D tensor: [inputSize, units])
        long weightsSize = 1;
        for (long dim : weightsShape) weightsSize *= dim;
        weights = new Tensor(weightsShape, Tensor.DataType.FLOAT32);
        int rank = weightsShape.length;
        int[] indices = new int[rank];
        for (long i = 0; i < weightsSize; i++) {
            float val = buffer.getFloat();
            switch (rank) {
                case 1: weights.setFloat(val, indices[0]); break;
                case 2: weights.setFloat(val, indices[0], indices[1]); break;
                default: weights.setFloat(val, i);
            }
            for (int d = rank - 1; d >= 0; d--) {
                indices[d]++;
                if (d > 0 && indices[d] >= weightsShape[d]) {
                    indices[d] = 0;
                } else {
                    break;
                }
            }
        }
        
        // Read bias shape
        int biasShapeLen = buffer.getInt();
        if (biasShapeLen > 0) {
            long[] biasShape = new long[biasShapeLen];
            for (int i = 0; i < biasShapeLen; i++) {
                biasShape[i] = buffer.getLong();
            }
            
            // Read bias data (1D tensor)
            long biasSize = 1;
            for (long dim : biasShape) biasSize *= dim;
            bias = new Tensor(biasShape, Tensor.DataType.FLOAT32);
            for (long i = 0; i < biasSize; i++) {
                bias.setFloat(buffer.getFloat(), (int) i);
            }
        } else {
            bias = null;
        }
    }

    @Override
    public String summary() {
        return String.format("%-20s %-20s %-15s %,d",
            name,
            Arrays.toString(outputShape),
            activation.name().toLowerCase(),
            getParameterCount());
    }

    @Override
    public void close() {
        if (weights != null) {
            weights.close();
            weights = null;
        }
        if (bias != null) {
            bias.close();
            bias = null;
        }
        if (inputCache != null) {
            inputCache.close();
            inputCache = null;
        }
        if (outputCache != null) {
            outputCache.close();
            outputCache = null;
        }
    }
}
