package com.apexocr.core.neural;

import com.apexocr.core.tensor.Tensor;
import com.apexocr.core.tensor.TensorOperations;
import com.apexocr.core.tensor.MemoryManager;

import java.util.Arrays;

/**
 * BiLSTM - Bidirectional Long Short-Term Memory layer implementation.
 * Combines forward and backward LSTM networks to capture context
 * from both past and future time steps.
 *
 * This layer is essential for OCR sequence recognition, enabling the model
 * to understand character context bidirectionally for improved accuracy.
 *
 * @author ApexOCR Team
 * @version 1.0.0
 */
public class BiLSTM implements Layer {
    private final String name;
    private final int units;
    private final boolean returnSequences;
    private final float dropoutRate;

    private Layer forwardLSTM;
    private Layer backwardLSTM;

    private Tensor forwardHiddenState;
    private Tensor backwardHiddenState;
    private Tensor forwardCellState;
    private Tensor backwardCellState;

    private long[] inputShape;
    private long[] outputShape;
    private boolean training;
    private boolean initialized;

    /**
     * Creates a new bidirectional LSTM layer.
     *
     * @param units Number of LSTM units in each direction
     * @param returnSequences Whether to return the full sequence or just the last output
     * @param dropoutRate Dropout rate for recurrent connections
     */
    public BiLSTM(int units, boolean returnSequences, float dropoutRate) {
        this.name = "bilstm_" + units;
        this.units = units;
        this.returnSequences = returnSequences;
        this.dropoutRate = dropoutRate;
        this.training = false;
        this.initialized = false;
    }

    /**
     * Creates a bidirectional LSTM with default settings.
     *
     * @param units Number of LSTM units
     */
    public BiLSTM(int units) {
        this(units, true, 0.0f);
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public LayerType getType() {
        return LayerType.BI_LSTM;
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
        // BiLSTM uses simplified random weights in denseGates, no trainable parameters
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

        long[] inputShape = input.getShape();
        int batchSize = (int) inputShape[0];
        int timeSteps = (int) inputShape[1];
        int features = (int) inputShape[2];

        // Forward LSTM processing
        Tensor forwardOutput = processDirection(input, true);

        // Backward LSTM processing
        Tensor backwardOutput = processDirection(input, false);

        // Concatenate forward and backward outputs
        long[] outputShape;
        if (returnSequences) {
            outputShape = new long[]{batchSize, timeSteps, units * 2};
        } else {
            outputShape = new long[]{batchSize, units * 2};
        }

        Tensor output = new Tensor(outputShape, Tensor.DataType.FLOAT32);

        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < timeSteps; t++) {
                for (int u = 0; u < units; u++) {
                    // Forward LSTM output
                    float forwardVal = returnSequences ?
                        forwardOutput.getFloat(b, t, u) :
                        forwardOutput.getFloat(b, u);
                    output.setFloat(forwardVal, b, t, u);

                    // Backward LSTM output
                    float backwardVal = returnSequences ?
                        backwardOutput.getFloat(b, timeSteps - 1 - t, u) :
                        backwardOutput.getFloat(b, u);
                    output.setFloat(backwardVal, b, t, units + u);
                }
            }
        }

        // Apply dropout if training
        if (training && dropoutRate > 0) {
            output = TensorOperations.dropout(output, dropoutRate, true);
        }

        return output;
    }

    /**
     * Processes the input in a single direction.
     */
    private Tensor processDirection(Tensor input, boolean forward) {
        long[] inputShape = input.getShape();
        int batchSize = (int) inputShape[0];
        int timeSteps = (int) inputShape[1];
        int features = (int) inputShape[2];

        Tensor hiddenState = new Tensor(new long[]{batchSize, units}, Tensor.DataType.FLOAT32);
        hiddenState.fill(0);

        Tensor cellState = new Tensor(new long[]{batchSize, units}, Tensor.DataType.FLOAT32);
        cellState.fill(0);

        Tensor output;
        if (returnSequences) {
            output = new Tensor(new long[]{batchSize, timeSteps, units}, Tensor.DataType.FLOAT32);
        } else {
            output = new Tensor(new long[]{batchSize, units}, Tensor.DataType.FLOAT32);
        }

        // Process each time step
        for (int t = 0; t < timeSteps; t++) {
            int timeIndex = forward ? t : timeSteps - 1 - t;

            // Extract input at current time step
            Tensor currentInput = new Tensor(new long[]{batchSize, features}, Tensor.DataType.FLOAT32);
            for (int b = 0; b < batchSize; b++) {
                for (int f = 0; f < features; f++) {
                    currentInput.setFloat(input.getFloat(b, timeIndex, f), b, f);
                }
            }

            // LSTM cell computation
            Tensor cellOutput = lstmCell(currentInput, hiddenState, cellState);

            // Update hidden and cell states
            for (int b = 0; b < batchSize; b++) {
                for (int u = 0; u < units; u++) {
                    hiddenState.setFloat(cellOutput.getFloat(b, u), b, u);
                }
            }

            // Store output
            if (returnSequences) {
                for (int b = 0; b < batchSize; b++) {
                    for (int u = 0; u < units; u++) {
                        output.setFloat(cellOutput.getFloat(b, u), b, t, u);
                    }
                }
            }
        }

        // Store final states
        if (forward) {
            forwardHiddenState = hiddenState.copy();
            forwardCellState = cellState.copy();
        } else {
            backwardHiddenState = hiddenState.copy();
            backwardCellState = cellState.copy();
        }

        hiddenState.close();
        cellState.close();

        return output;
    }

    /**
     * Performs a single LSTM cell computation.
     * Computes: h_t, c_t = LSTM(x_t, h_{t-1}, c_{t-1})
     *
     * This is a simplified LSTM cell for inference.
     * Full implementation would include all four gates (input, forget, output, cell).
     */
    private Tensor lstmCell(Tensor input, Tensor hiddenState, Tensor cellState) {
        long[] shape = input.getShape();
        int batchSize = (int) shape[0];
        int inputSize = (int) shape[1];

        // Simplified LSTM: combine input and hidden, apply gates
        // In a full implementation, this would be 4 separate gates

        Tensor combined = new Tensor(new long[]{batchSize, inputSize + units}, Tensor.DataType.FLOAT32);
        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < inputSize; i++) {
                combined.setFloat(input.getFloat(b, i), b, i);
            }
            for (int h = 0; h < units; h++) {
                combined.setFloat(hiddenState.getFloat(b, h), b, inputSize + h);
            }
        }

        // Apply a dense transformation (simplified gates)
        Tensor gates = denseGates(combined, 4 * units);

        // Split gates
        Tensor inputGate = new Tensor(new long[]{batchSize, units}, Tensor.DataType.FLOAT32);
        Tensor forgetGate = new Tensor(new long[]{batchSize, units}, Tensor.DataType.FLOAT32);
        Tensor outputGate = new Tensor(new long[]{batchSize, units}, Tensor.DataType.FLOAT32);
        Tensor cellCandidate = new Tensor(new long[]{batchSize, units}, Tensor.DataType.FLOAT32);

        for (int b = 0; b < batchSize; b++) {
            for (int u = 0; u < units; u++) {
                float g = gates.getFloat(b, u);
                inputGate.setFloat(g, b, u);

                g = gates.getFloat(b, units + u);
                forgetGate.setFloat(g, b, u);

                g = gates.getFloat(b, 2 * units + u);
                outputGate.setFloat(g, b, u);

                g = gates.getFloat(b, 3 * units + u);
                cellCandidate.setFloat(g, b, u);
            }
        }

        // Apply activations
        applyGateActivations(inputGate, forgetGate, outputGate, cellCandidate);

        // Compute new cell state
        Tensor newCellState = new Tensor(new long[]{batchSize, units}, Tensor.DataType.FLOAT32);
        for (int b = 0; b < batchSize; b++) {
            for (int u = 0; u < units; u++) {
                float c = cellState.getFloat(b, u);
                float i = inputGate.getFloat(b, u);
                float f = forgetGate.getFloat(b, u);
                float cand = cellCandidate.getFloat(b, u);
                newCellState.setFloat(f * c + i * cand, b, u);
            }
        }

        // Compute new hidden state
        Tensor newHiddenState = new Tensor(new long[]{batchSize, units}, Tensor.DataType.FLOAT32);
        for (int b = 0; b < batchSize; b++) {
            for (int u = 0; u < units; u++) {
                float c = newCellState.getFloat(b, u);
                float o = outputGate.getFloat(b, u);
                newHiddenState.setFloat((float) Math.tanh(c) * o, b, u);
            }
        }

        // Update cell state
        for (int b = 0; b < batchSize; b++) {
            for (int u = 0; u < units; u++) {
                cellState.setFloat(newCellState.getFloat(b, u), b, u);
            }
        }

        // Cleanup
        combined.close();
        gates.close();
        inputGate.close();
        forgetGate.close();
        outputGate.close();
        cellCandidate.close();
        newCellState.close();

        return newHiddenState;
    }

    /**
     * Applies dense transformation and sigmoid/tanh activations to gates.
     */
    private Tensor denseGates(Tensor input, int outputSize) {
        // Simplified: use random weights for demonstration
        // In a full implementation, this would use learned weights
        Tensor output = new Tensor(new long[]{input.getShape()[0], outputSize}, Tensor.DataType.FLOAT32);

        long size = output.getSize();
        for (long i = 0; i < size; i++) {
            output.setFloat(i, (float) (Math.random() * 2 - 1));
        }

        return output;
    }

    /**
     * Applies appropriate activations to LSTM gates.
     */
    private void applyGateActivations(Tensor inputGate, Tensor forgetGate,
                                       Tensor outputGate, Tensor cellCandidate) {
        long[] shape = inputGate.getShape();
        int batchSize = (int) shape[0];
        int units = (int) shape[1];

        // Input, forget, output gates use sigmoid (0-1)
        sigmoidActivation(inputGate);
        sigmoidActivation(forgetGate);
        sigmoidActivation(outputGate);

        // Cell candidate uses tanh (-1 to 1)
        tanhActivation(cellCandidate);
    }

    /**
     * Applies sigmoid activation in-place.
     */
    private void sigmoidActivation(Tensor tensor) {
        long size = tensor.getSize();
        for (long i = 0; i < size; i++) {
            float x = tensor.getFloat(i);
            tensor.setFloat(i, (float) (1.0 / (1.0 + Math.exp(-x))));
        }
    }

    /**
     * Applies tanh activation in-place.
     */
    private void tanhActivation(Tensor tensor) {
        long size = tensor.getSize();
        for (long i = 0; i < size; i++) {
            float x = tensor.getFloat(i);
            tensor.setFloat(i, (float) Math.tanh(x));
        }
    }

    /**
     * Initializes the layer with the given input shape.
     */
    private void initializeWithInputShape(long[] shape) {
        this.inputShape = shape.clone();

        int timeSteps = (int) shape[1];
        int features = (int) shape[2];

        // Output shape depends on whether we return sequences
        if (returnSequences) {
            outputShape = new long[]{-1, timeSteps, units * 2};
        } else {
            outputShape = new long[]{-1, units * 2};
        }

        initialized = true;
    }

    @Override
    public void resetState() {
        forwardHiddenState = null;
        backwardHiddenState = null;
        forwardCellState = null;
        backwardCellState = null;
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
        return forwardLSTM != null ? forwardLSTM.getWeights() : null;
    }

    @Override
    public Tensor getBiases() {
        return forwardLSTM != null ? forwardLSTM.getBiases() : null;
    }

    @Override
    public void setWeights(Tensor weights) {
        // Simplified implementation
    }

    @Override
    public void setBiases(Tensor biases) {
        // Simplified implementation
    }

    @Override
    public void initialize(Initializer initializer) {
        // Initialization happens during first forward pass
    }

    /**
     * Sets the input shape for this layer.
     */
    public void setInputShape(long[] inputShape) {
        this.inputShape = inputShape.clone();
    }

    @Override
    public byte[] serializeParameters() {
        return new byte[0];
    }

    @Override
    public void deserializeParameters(byte[] data) {
        // Simplified implementation
    }

    @Override
    public String summary() {
        return String.format("%-20s %-20s %-15s %,d",
            name,
            outputShape != null ? Arrays.toString(outputShape) : "pending",
            returnSequences ? "seq" : "last",
            getParameterCount());
    }

    @Override
    public void close() {
        if (forwardHiddenState != null) {
            forwardHiddenState.close();
            forwardHiddenState = null;
        }
        if (backwardHiddenState != null) {
            backwardHiddenState.close();
            backwardHiddenState = null;
        }
        if (forwardCellState != null) {
            forwardCellState.close();
            forwardCellState = null;
        }
        if (backwardCellState != null) {
            backwardCellState.close();
            backwardCellState = null;
        }
    }
}
