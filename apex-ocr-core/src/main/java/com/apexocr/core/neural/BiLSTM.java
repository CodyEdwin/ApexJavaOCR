package com.apexocr.core.neural;

import com.apexocr.core.tensor.Tensor;
import com.apexocr.core.tensor.TensorOperations;
import com.apexocr.core.tensor.MemoryManager;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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

    // Stored weights for the LSTM gates
    private Tensor forwardGateWeights;
    private Tensor forwardGateBias;
    private Tensor backwardGateWeights;
    private Tensor backwardGateBias;
    
    private Tensor forwardHiddenState;
    private Tensor backwardHiddenState;
    private Tensor forwardCellState;
    private Tensor backwardCellState;

    private long[] inputShape;
    private long[] outputShape;
    private boolean training;
    private boolean initialized;
    private int inputFeatures;

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
        if (forwardGateWeights == null) return 0;
        
        // Count parameters for both directions
        long params = forwardGateWeights.getSize();
        if (forwardGateBias != null) params += forwardGateBias.getSize();
        if (backwardGateWeights != null) params += backwardGateWeights.getSize();
        if (backwardGateBias != null) params += backwardGateBias.getSize();
        
        return params;
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
        Tensor gates = denseGates(combined, true);

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
     * Applies dense transformation using stored gate weights.
     */
    private Tensor denseGates(Tensor input, boolean forward) {
        // Use the stored gate weights instead of generating random ones
        Tensor gateWeights = forward ? forwardGateWeights : backwardGateWeights;
        Tensor gateBias = forward ? forwardGateBias : backwardGateBias;
        
        if (gateWeights == null) {
            // Fallback to random if weights not initialized yet
            Tensor output = new Tensor(new long[]{input.getShape()[0], 4 * units}, Tensor.DataType.FLOAT32);
            long size = output.getSize();
            for (long i = 0; i < size; i++) {
                output.setFloat(i, (float) (Math.random() * 2 - 1));
            }
            return output;
        }
        
        // Compute: output = input * weights + bias
        Tensor output = TensorOperations.matmul(input, gateWeights);
        
        // Add bias
        long[] outputShape = output.getShape();
        int batchSize = (int) outputShape[0];
        int gateSize = (int) outputShape[1];
        
        for (int b = 0; b < batchSize; b++) {
            for (int g = 0; g < gateSize; g++) {
                float val = output.getFloat(b, g) + gateBias.getFloat(g);
                output.setFloat(val, b, g);
            }
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
        this.inputFeatures = (int) shape[2];

        // Initialize gate weights for both directions
        // Gate weights shape: [inputFeatures + units, 4 * units]
        int combinedSize = inputFeatures + units;
        int gateSize = 4 * units;
        
        // Forward direction
        forwardGateWeights = new Tensor(new long[]{combinedSize, gateSize}, Tensor.DataType.FLOAT32);
        forwardGateWeights.randomNormal(0, 0.02f);
        forwardGateBias = new Tensor(new long[]{gateSize}, Tensor.DataType.FLOAT32);
        forwardGateBias.fill(0);
        
        // Backward direction
        backwardGateWeights = new Tensor(new long[]{combinedSize, gateSize}, Tensor.DataType.FLOAT32);
        backwardGateWeights.randomNormal(0, 0.02f);
        backwardGateBias = new Tensor(new long[]{gateSize}, Tensor.DataType.FLOAT32);
        backwardGateBias.fill(0);

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
        return forwardGateWeights;
    }

    @Override
    public Tensor getBiases() {
        return forwardGateBias;
    }

    @Override
    public void setWeights(Tensor weights) {
        if (weights != null) {
            this.forwardGateWeights = weights;
        }
    }

    @Override
    public void setBiases(Tensor biases) {
        if (biases != null) {
            this.forwardGateBias = biases;
        }
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
        if (forwardGateWeights == null) {
            return new byte[0];
        }
        
        // Format: [forward weights][forward bias][backward weights][backward bias]
        long[] fwShape = forwardGateWeights.getShape();
        long[] fbShape = forwardGateBias != null ? forwardGateBias.getShape() : new long[0];
        long[] bwShape = backwardGateWeights != null ? backwardGateWeights.getShape() : new long[0];
        long[] bbShape = backwardGateBias != null ? backwardGateBias.getShape() : new long[0];
        
        int fwBytes = (int) (forwardGateWeights.getSize() * Float.BYTES);
        int fbBytes = (int) (forwardGateBias != null ? forwardGateBias.getSize() * Float.BYTES : 0);
        int bwBytes = (int) (backwardGateWeights != null ? backwardGateWeights.getSize() * Float.BYTES : 0);
        int bbBytes = (int) (backwardGateBias != null ? backwardGateBias.getSize() * Float.BYTES : 0);
        
        int totalBytes = 4 + fwShape.length * 8 + fwBytes +
                         4 + fbShape.length * 8 + fbBytes +
                         4 + bwShape.length * 8 + bwBytes +
                         4 + bbShape.length * 8 + bbBytes;
        
        java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocate(totalBytes);
        buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN);
        
        // Write forward weights
        buffer.putInt(fwShape.length);
        for (long dim : fwShape) buffer.putLong(dim);
        for (long i = 0; i < forwardGateWeights.getSize(); i++) {
            buffer.putFloat(forwardGateWeights.getFloat(i));
        }
        
        // Write forward bias
        buffer.putInt(fbShape.length);
        for (long dim : fbShape) buffer.putLong(dim);
        if (forwardGateBias != null) {
            for (long i = 0; i < forwardGateBias.getSize(); i++) {
                buffer.putFloat(forwardGateBias.getFloat(i));
            }
        }
        
        // Write backward weights
        buffer.putInt(bwShape.length);
        for (long dim : bwShape) buffer.putLong(dim);
        if (backwardGateWeights != null) {
            for (long i = 0; i < backwardGateWeights.getSize(); i++) {
                buffer.putFloat(backwardGateWeights.getFloat(i));
            }
        }
        
        // Write backward bias
        buffer.putInt(bbShape.length);
        for (long dim : bbShape) buffer.putLong(dim);
        if (backwardGateBias != null) {
            for (long i = 0; i < backwardGateBias.getSize(); i++) {
                buffer.putFloat(backwardGateBias.getFloat(i));
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
        
        // Read forward weights (2D tensor)
        int fwLen = buffer.getInt();
        long[] fwShape = new long[fwLen];
        for (int i = 0; i < fwLen; i++) fwShape[i] = buffer.getLong();
        long fwSize = 1;
        for (long dim : fwShape) fwSize *= dim;
        forwardGateWeights = new Tensor(fwShape, Tensor.DataType.FLOAT32);
        long counter = 0;
        int rank = fwShape.length;
        int[] indices = new int[rank];
        for (long i = 0; i < fwSize; i++) {
            float val = buffer.getFloat();
            switch (rank) {
                case 1: forwardGateWeights.setFloat(val, indices[0]); break;
                case 2: forwardGateWeights.setFloat(val, indices[0], indices[1]); break;
                default: forwardGateWeights.setFloat(val, i);
            }
            counter++;
            for (int d = rank - 1; d >= 0; d--) {
                indices[d]++;
                if (d > 0 && indices[d] >= fwShape[d]) {
                    indices[d] = 0;
                } else {
                    break;
                }
            }
        }
        
        // Read forward bias (1D tensor)
        int fbLen = buffer.getInt();
        long[] fbShape = new long[fbLen];
        for (int i = 0; i < fbLen; i++) fbShape[i] = buffer.getLong();
        if (fbLen > 0) {
            long fbSize = 1;
            for (long dim : fbShape) fbSize *= dim;
            forwardGateBias = new Tensor(fbShape, Tensor.DataType.FLOAT32);
            for (long i = 0; i < fbSize; i++) {
                forwardGateBias.setFloat(buffer.getFloat(), (int) i);
            }
        }
        
        // Read backward weights (2D tensor)
        int bwLen = buffer.getInt();
        long[] bwShape = new long[bwLen];
        for (int i = 0; i < bwLen; i++) bwShape[i] = buffer.getLong();
        if (bwLen > 0) {
            long bwSize = 1;
            for (long dim : bwShape) bwSize *= dim;
            backwardGateWeights = new Tensor(bwShape, Tensor.DataType.FLOAT32);
            counter = 0;
            rank = bwShape.length;
            indices = new int[rank];
            for (long i = 0; i < bwSize; i++) {
                float val = buffer.getFloat();
                switch (rank) {
                    case 1: backwardGateWeights.setFloat(val, indices[0]); break;
                    case 2: backwardGateWeights.setFloat(val, indices[0], indices[1]); break;
                    default: backwardGateWeights.setFloat(val, i);
                }
                counter++;
                for (int d = rank - 1; d >= 0; d--) {
                    indices[d]++;
                    if (d > 0 && indices[d] >= bwShape[d]) {
                        indices[d] = 0;
                    } else {
                        break;
                    }
                }
            }
        }
        
        // Read backward bias (1D tensor)
        int bbLen = buffer.getInt();
        long[] bbShape = new long[bbLen];
        for (int i = 0; i < bbLen; i++) bbShape[i] = buffer.getLong();
        if (bbLen > 0) {
            long bbSize = 1;
            for (long dim : bbShape) bbSize *= dim;
            backwardGateBias = new Tensor(bbShape, Tensor.DataType.FLOAT32);
            for (long i = 0; i < bbSize; i++) {
                backwardGateBias.setFloat(buffer.getFloat(), (int) i);
            }
        }
        
        initialized = true;
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
        if (forwardGateWeights != null) {
            forwardGateWeights.close();
            forwardGateWeights = null;
        }
        if (forwardGateBias != null) {
            forwardGateBias.close();
            forwardGateBias = null;
        }
        if (backwardGateWeights != null) {
            backwardGateWeights.close();
            backwardGateWeights = null;
        }
        if (backwardGateBias != null) {
            backwardGateBias.close();
            backwardGateBias = null;
        }
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
