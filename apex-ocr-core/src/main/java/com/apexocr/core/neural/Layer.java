package com.apexocr.core.neural;

import com.apexocr.core.tensor.Tensor;

/**
 * Layer - Base interface for all neural network layers in the ApexOCR engine.
 * Defines the contract that all layer implementations must follow,
 * providing a consistent API for building and executing neural networks.
 *
 * This interface enables composition of complex neural network architectures
 * through a simple, unified interface. Each layer is responsible for
 * computing its output given an input tensor and maintaining any necessary
 * parameters (weights, biases, etc.).
 *
 * @author ApexOCR Team
 * @version 1.0.0
 */
public interface Layer {
    /**
     * Gets the name of this layer for identification and debugging.
     *
     * @return The layer name
     */
    String getName();

    /**
     * Gets the type identifier for this layer.
     *
     * @return The layer type
     */
    LayerType getType();

    /**
     * Gets the input shape expected by this layer.
     *
     * @return Array containing input dimensions [batch, ..., channels]
     */
    long[] getInputShape();

    /**
     * Gets the output shape produced by this layer.
     *
     * @return Array containing output dimensions [batch, ..., channels]
     */
    long[] getOutputShape();

    /**
     * Gets the number of trainable parameters in this layer.
     *
     * @return The parameter count
     */
    long getParameterCount();

    /**
     * Performs the forward pass computation.
     * Takes an input tensor and produces an output tensor.
     *
     * @param input The input tensor
     * @return The output tensor after applying this layer's transformation
     */
    Tensor forward(Tensor input);

    /**
     * Performs the forward pass with optional training mode.
     *
     * @param input The input tensor
     * @param training Whether this is a training forward pass
     * @return The output tensor
     */
    Tensor forward(Tensor input, boolean training);

    /**
     * Resets any layer-specific state (e.g., dropout masks, batch norm statistics).
     */
    void resetState();

    /**
     * Sets the layer into evaluation mode (disables dropout, etc.).
     */
    void eval();

    /**
     * Sets the layer into training mode (enables dropout, etc.).
     */
    void train();

    /**
     * Checks if this layer is in training mode.
     *
     * @return True if in training mode
     */
    boolean isTraining();

    /**
     * Gets the weights tensor if this layer has trainable parameters.
     *
     * @return The weights tensor, or null if no weights
     */
    Tensor getWeights();

    /**
     * Gets the biases tensor if this layer has trainable parameters.
     *
     * @return The biases tensor, or null if no biases
     */
    Tensor getBiases();

    /**
     * Sets the weights for this layer.
     *
     * @param weights The weights tensor
     */
    void setWeights(Tensor weights);

    /**
     * Sets the biases for this layer.
     *
     * @param biases The biases tensor
     */
    void setBiases(Tensor biases);

    /**
     * Initializes layer parameters with the specified initialization method.
     *
     * @param initializer The initialization method to use
     */
    void initialize(Initializer initializer);

    /**
     * Serializes the layer's parameters to a byte array.
     *
     * @return Byte array containing serialized parameters
     */
    byte[] serializeParameters();

    /**
     * Deserializes parameters from a byte array.
     *
     * @param data The serialized parameters
     */
    void deserializeParameters(byte[] data);

    /**
     * Gets a summary of this layer for logging/debugging.
     *
     * @return Summary string
     */
    String summary();

    /**
     * Closes this layer and releases any resources.
     */
    void close();

    /**
     * Enumeration of supported layer types.
     */
    enum LayerType {
        CONV2D("Conv2D"),
        DENSE("Dense"),
        MAX_POOL2D("MaxPool2D"),
        AVG_POOL2D("AvgPool2D"),
        LSTM("LSTM"),
        BI_LSTM("BiLSTM"),
        ACTIVATION("Activation"),
        BATCH_NORM("BatchNorm"),
        DROPOUT("Dropout"),
        FLATTEN("Flatten"),
        RESHAPE("Reshape"),
        PERMUTE("Permute"),
        REPEAT_VECTOR("RepeatVector"),
        ACTIVATIONS("Activations"),
        TIME_DISTRIBUTED("TimeDistributed"),
        BIDIRECTIONAL("Bidirectional"),
        Wrapper("Wrapper"),
        Ctc("CTC");

        private final String displayName;

        LayerType(String displayName) {
            this.displayName = displayName;
        }

        public String getDisplayName() {
            return displayName;
        }
    }

    /**
     * Enumeration of weight initialization methods.
     */
    enum Initializer {
        ZEROS,
        ONES,
        CONSTANT,
        RANDOM_UNIFORM,
        RANDOM_NORMAL,
        XAVIER_UNIFORM,
        XAVIER_NORMAL,
        HE_UNIFORM,
        HE_NORMAL,
        ORTHOGONAL
    }
}
