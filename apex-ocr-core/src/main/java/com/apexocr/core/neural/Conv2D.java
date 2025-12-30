package com.apexocr.core.neural;

import com.apexocr.core.tensor.Tensor;
import com.apexocr.core.tensor.TensorOperations;
import com.apexocr.core.tensor.MemoryManager;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Conv2D - 2D Convolutional layer implementation for neural network operations.
 * Performs spatial convolution over input feature maps using learned filters.
 *
 * This implementation is optimized for high-performance inference on the JVM,
 * supporting various configurations including dilation, groups, and padding modes.
 *
 * @author ApexOCR Team
 * @version 1.0.0
 */
public class Conv2D implements Layer {
    private String name;
    private final int filters;
    private final int[] kernelSize;
    private final int[] strides;
    private final int[] padding;
    private final int[] dilation;
    private final int groups;
    private final Dense.ActivationType activation;

    private Tensor kernels;
    private Tensor bias;
    private Tensor inputCache;
    private Tensor outputCache;

    private long[] inputShape;
    private long[] outputShape;
    private boolean training;
    private boolean initialized;

    /**
     * Padding modes supported by the convolution layer.
     */
    public enum PaddingMode {
        VALID,
        SAME,
        FULL
    }

    /**
     * Creates a new 2D convolutional layer.
     *
     * @param filters Number of output filters
     * @param kernelSize Size of the convolution kernel [height, width]
     * @param strides Stride of the convolution [height, width]
     * @param padding Padding mode or explicit padding
     * @param activation Activation function to apply
     */
    public Conv2D(int filters, int[] kernelSize, int[] strides,
                  int[] padding, Dense.ActivationType activation) {
        this.name = "conv2d_" + filters;
        this.filters = filters;
        this.kernelSize = kernelSize != null ? kernelSize.clone() : new int[]{3, 3};
        this.strides = strides != null ? strides.clone() : new int[]{1, 1};
        this.padding = padding != null ? padding.clone() : new int[]{0, 0};
        this.dilation = new int[]{1, 1};
        this.groups = 1;
        this.activation = activation;
        this.training = false;
        this.initialized = false;
    }

    /**
     * Creates a convolutional layer with default settings.
     *
     * @param filters Number of output filters
     * @param kernelSize Size of the convolution kernel
     */
    public Conv2D(int filters, int[] kernelSize) {
        this(filters, kernelSize, new int[]{1, 1}, new int[]{0, 0}, Dense.ActivationType.RELU);
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
        return LayerType.CONV2D;
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
        if (!initialized) return 0;
        
        // If kernels tensor exists, calculate from its shape
        if (kernels != null) {
            long[] shape = kernels.getShape();
            long kernelParams = 1;
            for (long dim : shape) {
                kernelParams *= dim;
            }
            long biasParams = useBias() ? filters : 0;
            return kernelParams + biasParams;
        }
        
        // Fallback: calculate from inputShape (may be null)
        if (inputShape == null || inputShape.length < 4) {
            return 0;
        }
        
        long kernelParams = kernelSize[0] * kernelSize[1] * inputShape[3] / groups * filters;
        long biasParams = useBias() ? filters : 0;

        return kernelParams + biasParams;
    }

    private boolean useBias() {
        return bias != null;
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

        Tensor output = performConvolution(input);

        if (useBias()) {
            addBias(output);
        }

        Tensor activated = applyActivation(output);

        if (training) {
            outputCache = activated;
        }

        return activated;
    }

    /**
     * Performs the 2D convolution operation.
     */
    private Tensor performConvolution(Tensor input) {
        long[] inputShape = input.getShape();
        int batchSize = (int) inputShape[0];
        int inputHeight = (int) inputShape[1];
        int inputWidth = (int) inputShape[2];
        int inChannels = (int) inputShape[3];

        int outputHeight = calculateOutputSize(inputHeight, kernelSize[0], strides[0], padding[0], dilation[0]);
        int outputWidth = calculateOutputSize(inputWidth, kernelSize[1], strides[1], padding[1], dilation[1]);

        // Validate output dimensions
        if (outputHeight <= 0 || outputWidth <= 0) {
            throw new IllegalArgumentException(
                String.format("Invalid convolution output size: %dx%d for input %dx%d with kernel %dx%d, stride %dx%d, padding %dx%d",
                    outputHeight, outputWidth, inputHeight, inputWidth,
                    kernelSize[0], kernelSize[1], strides[0], strides[1], padding[0], padding[1]));
        }

        long[] outputShape = new long[]{batchSize, outputHeight, outputWidth, filters};
        Tensor output = new Tensor(outputShape, Tensor.DataType.FLOAT32);

        int kernelHeight = kernelSize[0];
        int kernelWidth = kernelSize[1];
        int padH = padding[0];
        int padW = padding[1];
        int strideH = strides[0];
        int strideW = strides[1];
        int dilationH = dilation[0];
        int dilationW = dilation[1];

        for (int b = 0; b < batchSize; b++) {
            for (int oh = 0; oh < outputHeight; oh++) {
                for (int ow = 0; ow < outputWidth; ow++) {
                    for (int f = 0; f < filters; f++) {
                        float sum = 0;

                        for (int kh = 0; kh < kernelHeight; kh++) {
                            for (int kw = 0; kw < kernelWidth; kw++) {
                                int ih = oh * strideH + kh * dilationH - padH;
                                int iw = ow * strideW + kw * dilationW - padW;

                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                    for (int ic = 0; ic < inChannels; ic++) {
                                        sum += input.getFloat(b, ih, iw, ic) *
                                               kernels.getFloat(kh, kw, ic, f);
                                    }
                                }
                            }
                        }

                        output.setFloat(sum, b, oh, ow, f);
                    }
                }
            }
        }

        return output;
    }

    /**
     * Calculates the output dimension for a convolution.
     */
    private int calculateOutputSize(int inputSize, int kernelSize, int stride, int padding, int dilation) {
        int dilatedKernel = (kernelSize - 1) * dilation + 1;
        return (inputSize + 2 * padding - dilatedKernel) / stride + 1;
    }

    /**
     * Adds bias terms to the output.
     */
    private void addBias(Tensor output) {
        long[] shape = output.getShape();
        int batchSize = (int) shape[0];
        int height = (int) shape[1];
        int width = (int) shape[2];

        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    for (int f = 0; f < filters; f++) {
                        float val = output.getFloat(b, h, w, f) + bias.getFloat(f);
                        output.setFloat(val, b, h, w, f);
                    }
                }
            }
        }
    }

    /**
     * Applies the activation function.
     */
    private Tensor applyActivation(Tensor input) {
        if (activation == null || activation == Dense.ActivationType.LINEAR) {
            return input;
        }

        switch (activation) {
            case RELU:
                return TensorOperations.relu(input);
            case LEAKY_RELU:
                return TensorOperations.leakyRelu(input, 0.01f);
            case SIGMOID:
                return TensorOperations.sigmoid(input);
            case TANH:
                return TensorOperations.tanh(input);
            default:
                return input;
        }
    }

    /**
     * Initializes the layer with the given input shape.
     */
    private void initializeWithInputShape(long[] shape) {
        this.inputShape = shape.clone();

        int inChannels = (int) shape[3];
        int kernelHeight = kernelSize[0];
        int kernelWidth = kernelSize[1];

        // Initialize kernels
        long[] kernelShape = new long[]{kernelHeight, kernelWidth, inChannels / groups, filters};
        kernels = new Tensor(kernelShape, Tensor.DataType.FLOAT32);
        initializeKernels(Initializer.HE_NORMAL);

        // Initialize bias
        bias = new Tensor(new long[]{filters}, Tensor.DataType.FLOAT32);
        bias.fill(0);

        // Calculate output shape
        int outputHeight = calculateOutputSize((int) shape[1], kernelSize[0], strides[0], padding[0], dilation[0]);
        int outputWidth = calculateOutputSize((int) shape[2], kernelSize[1], strides[1], padding[1], dilation[1]);
        outputShape = new long[]{-1, outputHeight, outputWidth, filters};

        initialized = true;
    }

    /**
     * Initializes kernel weights.
     */
    private void initializeKernels(Initializer initializer) {
        int kernelHeight = kernelSize[0];
        int kernelWidth = kernelSize[1];
        int inChannels = (int) inputShape[3];

        long fanIn = kernelHeight * kernelWidth * inChannels / groups;
        long fanOut = kernelHeight * kernelWidth * filters / groups;

        switch (initializer) {
            case HE_NORMAL:
                float std = (float) Math.sqrt(2.0 / fanIn);
                kernels.randomNormal(0, std);
                break;
            case HE_UNIFORM:
                float limit = (float) Math.sqrt(6.0 / fanIn);
                kernels.randomUniform(-limit, limit);
                break;
            case XAVIER_NORMAL:
                float stdX = (float) Math.sqrt(2.0 / (fanIn + fanOut));
                kernels.randomNormal(0, stdX);
                break;
            default:
                kernels.randomNormal(0, 0.02f);
        }
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
        return kernels;
    }

    @Override
    public Tensor getBiases() {
        return bias;
    }

    @Override
    public void setWeights(Tensor weights) {
        this.kernels = weights;
        this.initialized = true;
    }

    @Override
    public void setBiases(Tensor biases) {
        this.bias = biases;
    }

    /**
     * Sets weights from pre-trained model (EasyOCR format).
     * The Python converter provides weights in [h, w, in_ch, out_ch] format.
     * This method converts to and expects [out_ch, in_ch, h, w] for internal use.
     * 
     * @param weights Flattened weight array from pre-trained model
     * @param biases Flattened bias array from pre-trained model (can be null)
     */
    public void setWeightsFromPreTrained(float[] weights, float[] biases) {
        if (weights == null || weights.length == 0) {
            return;
        }

        int kh = kernelSize[0];
        int kw = kernelSize[1];
        int outCh = filters;

        // Determine input channels
        int inCh;
        if (this.inputShape != null && this.inputShape.length >= 4) {
            inCh = (int) this.inputShape[3];
        } else {
            inCh = weights.length / (kh * kw * outCh);
        }

        int expectedSize = kh * kw * inCh * outCh;

        if (weights.length == expectedSize) {
            // Weights are in EasyOCR format [h, w, in_ch, out_ch]
            long[] kernelShape = new long[]{kh, kw, inCh, outCh};
            kernels = new Tensor(kernelShape, Tensor.DataType.FLOAT32);

            // Convert from [h, w, in_ch, out_ch] flat array to tensor
            for (int h = 0; h < kh; h++) {
                for (int w = 0; w < kw; w++) {
                    for (int ic = 0; ic < inCh; ic++) {
                        for (int oc = 0; oc < outCh; oc++) {
                            int srcIndex = ((h * kw + w) * inCh + ic) * outCh + oc;
                            kernels.setFloat(weights[srcIndex], h, w, ic, oc);
                        }
                    }
                }
            }
        } else if (weights.length == kh * kw * outCh * inCh) {
            // Weights might be in PyTorch format [out_ch, in_ch, h, w]
            // where the calculation kh*kw*outCh*inCh gives the same result
            // but we need to handle it differently
            long[] kernelShape = new long[]{kh, kw, inCh, outCh};
            kernels = new Tensor(kernelShape, Tensor.DataType.FLOAT32);

            // Transpose from [out_ch, in_ch, h, w] to [h, w, in_ch, out_ch]
            for (int h = 0; h < kh; h++) {
                for (int w = 0; w < kw; w++) {
                    for (int ic = 0; ic < inCh; ic++) {
                        for (int oc = 0; oc < outCh; oc++) {
                            int srcIndex = ((oc * inCh + ic) * kh + h) * kw + w;
                            kernels.setFloat(weights[srcIndex], h, w, ic, oc);
                        }
                    }
                }
            }
        } else {
            // Weight size doesn't match expected - try recovery
            System.out.println("WARNING: " + name + " weight size mismatch. Expected " + expectedSize +
                               " but got " + weights.length);

            // Try treating as PyTorch format with swapped in_ch/out_ch
            int swappedInCh = outCh;
            int swappedOutCh = inCh;
            int swappedExpected = kh * kw * swappedInCh * swappedOutCh;

            if (weights.length == swappedExpected) {
                System.out.println("Attempting recovery with swapped channels...");
                long[] kernelShape = new long[]{kh, kw, swappedInCh, swappedOutCh};
                kernels = new Tensor(kernelShape, Tensor.DataType.FLOAT32);

                for (int h = 0; h < kh; h++) {
                    for (int w = 0; w < kw; w++) {
                        for (int ic = 0; ic < swappedInCh; ic++) {
                            for (int oc = 0; oc < swappedOutCh; oc++) {
                                int srcIndex = ((oc * swappedInCh + ic) * kh + h) * kw + w;
                                kernels.setFloat(weights[srcIndex], h, w, ic, oc);
                            }
                        }
                    }
                }
                System.out.println("Recovered " + name + " as PyTorch format with inCh=" + swappedInCh);
            } else {
                System.out.println("ERROR: Cannot recover " + name + " - weight size doesn't match any known format");
            }
        }
        
        // Set bias if provided
        if (biases != null && biases.length == outCh) {
            this.bias = new Tensor(new long[]{outCh}, Tensor.DataType.FLOAT32);
            for (int i = 0; i < outCh; i++) {
                this.bias.setFloat(biases[i], i);
            }
        } else if (outCh > 0) {
            // Initialize zero bias
            this.bias = new Tensor(new long[]{outCh}, Tensor.DataType.FLOAT32);
            this.bias.fill(0);
        }
        
        // Set output shape
        if (inputShape != null && inputShape.length >= 4) {
            int outputHeight = calculateOutputSize((int) inputShape[1], kernelSize[0], strides[0], padding[0], dilation[0]);
            int outputWidth = calculateOutputSize((int) inputShape[2], kernelSize[1], strides[1], padding[1], dilation[1]);
            outputShape = new long[]{-1, outputHeight, outputWidth, filters};
        } else {
            // Default output shape when inputShape is unknown
            outputShape = new long[]{-1, -1, -1, filters};
        }
        
        initialized = true;
    }

    @Override
    public void initialize(Initializer initializer) {
        if (!initialized && inputShape == null) {
            throw new IllegalStateException("Input shape must be set before initialization");
        }
        // Full initialization happens when first forward pass is called
    }

    /**
     * Sets the input shape for this layer.
     */
    public void setInputShape(long[] inputShape) {
        this.inputShape = inputShape.clone();
    }

    @Override
    public byte[] serializeParameters() {
        if (kernels == null) {
            return new byte[0];
        }
        
        // Format: [kernel data][bias data]
        // Each tensor: [shape dims][shape values][data bytes]
        long[] kernelShape = kernels.getShape();
        long[] biasShape = bias != null ? bias.getShape() : new long[0];
        
        int kernelSizeBytes = (int) (kernels.getSize() * Float.BYTES);
        int biasSizeBytes = (int) (bias != null ? bias.getSize() * Float.BYTES : 0);
        
        int totalBytes = 4 + kernelShape.length * 8 + kernelSizeBytes + 
                         4 + biasShape.length * 8 + biasSizeBytes;
        
        java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocate(totalBytes);
        buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN);
        
        // Write kernel shape
        buffer.putInt(kernelShape.length);
        for (long dim : kernelShape) {
            buffer.putLong(dim);
        }
        
        // Write kernel data
        for (long i = 0; i < kernels.getSize(); i++) {
            buffer.putFloat(kernels.getFloat(i));
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
        
        // Read kernel shape
        int kernelShapeLen = buffer.getInt();
        long[] kernelShape = new long[kernelShapeLen];
        for (int i = 0; i < kernelShapeLen; i++) {
            kernelShape[i] = buffer.getLong();
        }
        
        // Read kernel data (4D tensor: [kernelHeight, kernelWidth, inChannels, filters])
        long kernelSize = 1;
        for (long dim : kernelShape) kernelSize *= dim;
        kernels = new Tensor(kernelShape, Tensor.DataType.FLOAT32);
        int rank = kernelShape.length;
        int[] indices = new int[rank];
        for (long i = 0; i < kernelSize; i++) {
            float val = buffer.getFloat();
            switch (rank) {
                case 1: kernels.setFloat(val, indices[0]); break;
                case 2: kernels.setFloat(val, indices[0], indices[1]); break;
                case 3: kernels.setFloat(val, indices[0], indices[1], indices[2]); break;
                case 4: kernels.setFloat(val, indices[0], indices[1], indices[2], indices[3]); break;
                default: kernels.setFloat(val, i);
            }
            for (int d = rank - 1; d >= 0; d--) {
                indices[d]++;
                if (d > 0 && indices[d] >= kernelShape[d]) {
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
        
        initialized = true;
    }

    @Override
    public String summary() {
        return String.format("%-20s %-20s %-10s %,d",
            name,
            outputShape != null ? Arrays.toString(outputShape) : "pending",
            activation.name().toLowerCase(),
            getParameterCount());
    }

    @Override
    public void close() {
        if (kernels != null) {
            kernels.close();
            kernels = null;
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
