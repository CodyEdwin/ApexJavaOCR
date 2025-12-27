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
    private final String name;
    private final int units;
    private final ActivationType activation;
    private final boolean useBias;

    private Tensor weights;
    private Tensor bias;
    private long[] inputShape;
    private long[] outputShape;

    private boolean training;
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
        // Lazy initialization on first forward pass
        if (weights == null) {
            setInputShape(input.getShape());
            initialize(Initializer.HE_NORMAL);
        }

        // Cache input for potential gradient computation
        if (training) {
            inputCache = input.copy();
        }

        long[] inputShape = input.getShape();
        long batchSize = inputShape[0];
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

    @Override
    public void setWeights(Tensor weights) {
        this.weights = weights;
    }

    @Override
    public void setBiases(Tensor biases) {
        this.bias = biases;
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
