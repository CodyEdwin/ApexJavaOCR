package com.apexocr.core.neural;

import com.apexocr.core.tensor.Tensor;
import com.apexocr.core.tensor.TensorOperations;

import java.util.Arrays;

/**
 * MaxPool2D - 2D Max Pooling layer implementation for neural network operations.
 * Performs downsampling by taking the maximum value within each pooling region.
 *
 * This layer is commonly used in convolutional neural networks for spatial
 * downsampling, reducing computational load and providing translation invariance.
 *
 * @author ApexOCR Team
 * @version 1.0.0
 */
public class MaxPool2D implements Layer {
    private String name;
    private final int[] poolSize;
    private final int[] strides;
    private final int[] padding;

    private Tensor inputCache;
    private Tensor outputCache;

    private long[] inputShape;
    private long[] outputShape;
    private boolean training;
    private boolean initialized;

    /**
     * Creates a new 2D max pooling layer.
     *
     * @param poolSize Size of the pooling window [height, width]
     * @param strides Stride of the pooling operation [height, width]
     * @param padding Padding to apply before pooling
     */
    public MaxPool2D(int[] poolSize, int[] strides, int[] padding) {
        this.name = "maxpool2d_" + poolSize[0] + "x" + poolSize[1];
        this.poolSize = poolSize != null ? poolSize.clone() : new int[]{2, 2};
        this.strides = strides != null ? strides.clone() : this.poolSize.clone();
        this.padding = padding != null ? padding.clone() : new int[]{0, 0};
        this.training = false;
        this.initialized = false;
    }

    /**
     * Creates a max pooling layer with default settings (2x2 pooling).
     */
    public MaxPool2D() {
        this(new int[]{2, 2}, null, null);
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
        return LayerType.MAX_POOL2D;
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
        // Max pooling has no learnable parameters
        return 0;
    }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        this.training = training;
        return forward(input);
    }

    @Override
    public Tensor forward(Tensor input) {
        if (!initialized) {
            initializeWithInputShape(input.getShape());
        }

        if (training) {
            inputCache = input.copy();
        }

        Tensor output = performMaxPooling(input);

        if (training) {
            outputCache = output;
        }

        return output;
    }

    /**
     * Performs the 2D max pooling operation.
     */
    private Tensor performMaxPooling(Tensor input) {
        long[] inputShape = input.getShape();
        int batchSize = (int) inputShape[0];
        int inputHeight = (int) inputShape[1];
        int inputWidth = (int) inputShape[2];
        int channels = (int) inputShape[3];

        int outputHeight = calculateOutputSize(inputHeight, poolSize[0], strides[0], padding[0]);
        int outputWidth = calculateOutputSize(inputWidth, poolSize[1], strides[1], padding[1]);

        long[] outputShape = new long[]{batchSize, outputHeight, outputWidth, channels};
        Tensor output = new Tensor(outputShape, Tensor.DataType.FLOAT32);

        int poolHeight = poolSize[0];
        int poolWidth = poolSize[1];
        int strideH = strides[0];
        int strideW = strides[1];
        int padH = padding[0];
        int padW = padding[1];

        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        float max = Float.NEGATIVE_INFINITY;

                        for (int ph = 0; ph < poolHeight; ph++) {
                            for (int pw = 0; pw < poolWidth; pw++) {
                                int ih = oh * strideH + ph - padH;
                                int iw = ow * strideW + pw - padW;

                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                    float value = input.getFloat(b, ih, iw, c);
                                    if (value > max) {
                                        max = value;
                                    }
                                }
                            }
                        }

                        output.setFloat(max, b, oh, ow, c);
                    }
                }
            }
        }

        return output;
    }

    /**
     * Calculates the output dimension for a pooling operation.
     */
    private int calculateOutputSize(int inputSize, int poolSize, int stride, int padding) {
        return (inputSize + 2 * padding - poolSize) / stride + 1;
    }

    /**
     * Initializes the layer with the given input shape.
     */
    private void initializeWithInputShape(long[] shape) {
        this.inputShape = shape.clone();

        int outputHeight = calculateOutputSize((int) shape[1], poolSize[0], strides[0], padding[0]);
        int outputWidth = calculateOutputSize((int) shape[2], poolSize[1], strides[1], padding[1]);
        outputShape = new long[]{-1, outputHeight, outputWidth, shape[3]};

        initialized = true;
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
        // Max pooling has no weights
        return null;
    }

    @Override
    public Tensor getBiases() {
        // Max pooling has no biases
        return null;
    }

    @Override
    public void setWeights(Tensor weights) {
        // Not applicable for max pooling
    }

    @Override
    public void setBiases(Tensor biases) {
        // Not applicable for max pooling
    }

    @Override
    public void initialize(Initializer initializer) {
        // No parameters to initialize for max pooling
    }

    /**
     * Sets the input shape for this layer.
     */
    public void setInputShape(long[] inputShape) {
        this.inputShape = inputShape.clone();
    }

    @Override
    public byte[] serializeParameters() {
        // No parameters to serialize
        return new byte[0];
    }

    @Override
    public void deserializeParameters(byte[] data) {
        // No parameters to deserialize
    }

    @Override
    public String summary() {
        return String.format("%-20s %-20s %-15s %,d",
            name,
            outputShape != null ? Arrays.toString(outputShape) : "pending",
            "max",
            getParameterCount());
    }

    @Override
    public void close() {
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
