package com.apexocr.core.neural;

import com.apexocr.core.tensor.Tensor;

import java.util.Arrays;

/**
 * ReshapeLayer - Layer that reshapes tensors between different dimensionalities.
 * Essential for converting 4D feature maps from CNN layers into 3D sequences
 * for RNN processing in CRNN architectures.
 *
 * This layer transforms [batch, height, width, channels] to [batch, timeSteps, features]
 * where timeSteps = width and features = height * channels.
 *
 * @author ApexOCR Team
 * @version 1.0.0
 */
public class ReshapeLayer implements Layer {
    private final String name = "reshape";
    private long[] inputShape;
    private long[] outputShape;

    /**
     * Creates a new reshape layer.
     */
    public ReshapeLayer() {
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public LayerType getType() {
        return LayerType.RESHAPE;
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
        return 0;  // Reshape layers have no parameters
    }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        return forward(input);
    }

    @Override
    public Tensor forward(Tensor input) {
        // Store input shape for reference
        inputShape = input.getShape().clone();

        long[] shape = input.getShape();
        long batchSize = shape[0];
        long height = shape[1];
        long width = shape[2];
        long channels = shape[3];

        // Reshape from [batch, height, width, channels] to [batch, width, height*channels]
        long features = height * channels;
        long timeSteps = width;

        outputShape = new long[]{batchSize, timeSteps, features};

        // Create the reshaped tensor
        Tensor output = new Tensor(outputShape, Tensor.DataType.FLOAT32);

        // Copy data with proper reordering
        for (int b = 0; b < batchSize; b++) {
            for (int w = 0; w < width; w++) {
                for (int h = 0; h < height; h++) {
                    for (int c = 0; c < channels; c++) {
                        // Flatten (h, c) into single feature dimension
                        long featureIndex = h * channels + c;
                        float value = input.getFloat(b, h, w, c);
                        output.setFloat(value, b, w, featureIndex);
                    }
                }
            }
        }

        return output;
    }

    @Override
    public void resetState() {
        // No state to reset for reshape layer
    }

    @Override
    public void eval() {
        // No mode setting needed
    }

    @Override
    public void train() {
        // No mode setting needed
    }

    @Override
    public boolean isTraining() {
        return false;
    }

    @Override
    public Tensor getWeights() {
        return null;
    }

    @Override
    public Tensor getBiases() {
        return null;
    }

    @Override
    public void setWeights(Tensor weights) {
        // Not applicable
    }

    @Override
    public void setBiases(Tensor biases) {
        // Not applicable
    }

    public void initialize(Initializer initializer) {
        // No initialization needed
    }

    public void setInputShape(long[] inputShape) {
        this.inputShape = inputShape.clone();
    }

    @Override
    public byte[] serializeParameters() {
        return new byte[0];
    }

    @Override
    public void deserializeParameters(byte[] data) {
        // No parameters to deserialize
    }

    @Override
    public String summary() {
        return String.format("%-20s %-20s %-10s %,d",
            name,
            outputShape != null ? Arrays.toString(outputShape) : "pending",
            "none",
            0);
    }

    @Override
    public void close() {
        // No resources to clean up
    }
}
