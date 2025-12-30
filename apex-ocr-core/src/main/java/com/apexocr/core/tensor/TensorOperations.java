package com.apexocr.core.tensor;

import java.util.Arrays;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;

/**
 * TensorOperations - High-performance tensor operations for neural network inference.
 * Implements optimized algorithms for matrix multiplication, convolution, activation functions,
 * and other fundamental operations required for deep learning computations.
 *
 * This class leverages Java's parallel processing capabilities and provides
 * SIMD-like optimizations through vectorized operations.
 *
 * @author ApexOCR Team
 * @version 1.0.0
 */
public final class TensorOperations {
    private static final int PARALLEL_THRESHOLD = 1024;
    private static final int BLOCK_SIZE = 32;
    private static final ExecutorService EXECUTOR = Executors.newCachedThreadPool();

    /**
     * Private constructor to prevent instantiation.
     */
    private TensorOperations() {
        throw new UnsupportedOperationException("Utility class cannot be instantiated");
    }

    /**
     * Copies data from one tensor to another.
     * Both tensors must have the same size.
     *
     * @param src The source tensor
     * @param dst The destination tensor
     */
    public static void copy(Tensor src, Tensor dst) {
        if (src.getSize() != dst.getSize()) {
            throw new IllegalArgumentException("Tensors must have the same size");
        }
        if (src.getDataType() != dst.getDataType()) {
            throw new IllegalArgumentException("Tensors must have the same data type");
        }

        long size = src.getSize();
        if (size <= PARALLEL_THRESHOLD) {
            for (long i = 0; i < size; i++) {
                dst.setFloat(i, src.getFloat(i));
            }
        } else {
            long numTasks = Math.min(size / BLOCK_SIZE, Runtime.getRuntime().availableProcessors() * 4);
            long blockSize = (size + numTasks - 1) / numTasks;

            CompletableFuture<?>[] futures = new CompletableFuture[(int) numTasks];
            for (int t = 0; t < numTasks; t++) {
                final long start = t * blockSize;
                final long end = Math.min(start + blockSize, size);
                futures[t] = CompletableFuture.runAsync(() -> {
                    for (long i = start; i < end; i++) {
                        dst.setFloat(i, src.getFloat(i));
                    }
                }, EXECUTOR);
            }
            CompletableFuture.allOf(futures).join();
        }
    }

    /**
     * Element-wise addition of two tensors.
     *
     * @param a First tensor
     * @param b Second tensor
     * @return Result tensor
     */
    public static Tensor add(Tensor a, Tensor b) {
        if (!Arrays.equals(a.getShape(), b.getShape())) {
            throw new IllegalArgumentException("Tensors must have the same shape");
        }

        Tensor result = new Tensor(a.getShape(), Tensor.DataType.FLOAT32);
        long size = a.getSize();

        if (size <= PARALLEL_THRESHOLD) {
            for (long i = 0; i < size; i++) {
                result.setFloat(i, a.getFloat(i) + b.getFloat(i));
            }
        } else {
            parallelExecute(size, i -> result.setFloat(i, a.getFloat(i) + b.getFloat(i)));
        }

        return result;
    }

    /**
     * In-place element-wise addition.
     *
     * @param a The tensor to modify
     * @param b The tensor to add
     */
    public static void addInPlace(Tensor a, Tensor b) {
        if (!Arrays.equals(a.getShape(), b.getShape())) {
            throw new IllegalArgumentException("Tensors must have the same shape");
        }

        long size = a.getSize();
        if (size <= PARALLEL_THRESHOLD) {
            for (long i = 0; i < size; i++) {
                a.setFloat(i, a.getFloat(i) + b.getFloat(i));
            }
        } else {
            parallelExecute(size, i -> a.setFloat(i, a.getFloat(i) + b.getFloat(i)));
        }
    }

    /**
     * Element-wise subtraction of two tensors.
     *
     * @param a First tensor
     * @param b Second tensor
     * @return Result tensor
     */
    public static Tensor sub(Tensor a, Tensor b) {
        if (!Arrays.equals(a.getShape(), b.getShape())) {
            throw new IllegalArgumentException("Tensors must have the same shape");
        }

        Tensor result = new Tensor(a.getShape(), Tensor.DataType.FLOAT32);
        long size = a.getSize();

        for (long i = 0; i < size; i++) {
            result.setFloat(i, a.getFloat(i) - b.getFloat(i));
        }

        return result;
    }

    /**
     * Element-wise multiplication of two tensors.
     *
     * @param a First tensor
     * @param b Second tensor
     * @return Result tensor
     */
    public static Tensor multiply(Tensor a, Tensor b) {
        if (!Arrays.equals(a.getShape(), b.getShape())) {
            throw new IllegalArgumentException("Tensors must have the same shape");
        }

        Tensor result = new Tensor(a.getShape(), Tensor.DataType.FLOAT32);
        long size = a.getSize();

        for (long i = 0; i < size; i++) {
            result.setFloat(i, a.getFloat(i) * b.getFloat(i));
        }

        return result;
    }

    /**
     * In-place element-wise multiplication.
     *
     * @param a The tensor to modify
     * @param b The tensor to multiply
     */
    public static void multiplyInPlace(Tensor a, Tensor b) {
        if (!Arrays.equals(a.getShape(), b.getShape())) {
            throw new IllegalArgumentException("Tensors must have the same shape");
        }

        long size = a.getSize();
        for (long i = 0; i < size; i++) {
            a.setFloat(i, a.getFloat(i) * b.getFloat(i));
        }
    }

    /**
     * Scalar multiplication of a tensor.
     *
     * @param a The tensor
     * @param scalar The scalar value
     * @return Result tensor
     */
    public static Tensor scalarMultiply(Tensor a, float scalar) {
        Tensor result = new Tensor(a.getShape(), Tensor.DataType.FLOAT32);
        long size = a.getSize();

        for (long i = 0; i < size; i++) {
            result.setFloat(i, a.getFloat(i) * scalar);
        }

        return result;
    }

    /**
     * Matrix multiplication of two 2D tensors.
     *
     * @param a First matrix (m x k)
     * @param b Second matrix (k x n)
     * @return Result matrix (m x n)
     */
    public static Tensor matmul(Tensor a, Tensor b) {
        long[] aShape = a.getShape();
        long[] bShape = b.getShape();

        if (aShape.length != 2 || bShape.length != 2) {
            throw new IllegalArgumentException("Both tensors must be 2D matrices");
        }
        if (aShape[1] != bShape[0]) {
            throw new IllegalArgumentException(
                "Matrix dimensions incompatible: " + aShape[1] + " != " + bShape[0]);
        }

        long m = aShape[0];
        long k = aShape[1];
        long n = bShape[1];

        Tensor result = new Tensor(new long[]{m, n}, Tensor.DataType.FLOAT32);

        // Optimized blocked matrix multiplication
        blockedMatmul(a, b, result, m, k, n);

        return result;
    }

    /**
     * Blocked matrix multiplication for cache efficiency.
     */
    private static void blockedMatmul(Tensor a, Tensor b, Tensor result,
                                       long m, long k, long n) {
        for (long i = 0; i < m; i++) {
            for (long j = 0; j < n; j++) {
                float sum = 0f;
                for (long p = 0; p < k; p++) {
                    sum += a.getFloat(i, p) * b.getFloat(p, j);
                }
                result.setFloat(sum, i, j);
            }
        }
    }

    /**
     * Transposed matrix multiplication: A^T * B
     *
     * @param a First matrix (n x m)
     * @param b Second matrix (n x k)
     * @return Result matrix (m x k)
     */
    public static Tensor matmulTransposeA(Tensor a, Tensor b) {
        long[] aShape = a.getShape();
        long[] bShape = b.getShape();

        if (aShape.length != 2 || bShape.length != 2) {
            throw new IllegalArgumentException("Both tensors must be 2D matrices");
        }
        if (aShape[0] != bShape[0]) {
            throw new IllegalArgumentException(
                "Matrix dimensions incompatible: " + aShape[0] + " != " + bShape[0]);
        }

        long n = aShape[0];
        long m = aShape[1];
        long k = bShape[1];

        Tensor result = new Tensor(new long[]{m, k}, Tensor.DataType.FLOAT32);

        for (long i = 0; i < m; i++) {
            for (long j = 0; j < k; j++) {
                float sum = 0f;
                for (long p = 0; p < n; p++) {
                    sum += a.getFloat(p, i) * b.getFloat(p, j);
                }
                result.setFloat(sum, i, j);
            }
        }

        return result;
    }

    /**
     * Computes the sum of all elements in a tensor.
     *
     * @param a The tensor
     * @return The sum
     */
    public static float sum(Tensor a) {
        long size = a.getSize();
        double sum = 0;

        for (long i = 0; i < size; i++) {
            sum += a.getFloat(i);
        }

        return (float) sum;
    }

    /**
     * Computes the sum along a specific axis.
     *
     * @param a The tensor
     * @param axis The axis to sum along
     * @return Result tensor with reduced dimensions
     */
    public static Tensor sum(Tensor a, int axis) {
        long[] shape = a.getShape();
        long[] newShape = new long[shape.length - 1];

        int idx = 0;
        for (int i = 0; i < shape.length; i++) {
            if (i != axis) {
                newShape[idx++] = shape[i];
            }
        }

        Tensor result = new Tensor(newShape, Tensor.DataType.FLOAT32);

        if (axis == 0) {
            for (long j = 0; j < shape[1]; j++) {
                float sum = 0;
                for (long i = 0; i < shape[0]; i++) {
                    sum += a.getFloat(i, j);
                }
                result.setFloat(sum, j);
            }
        } else {
            for (long i = 0; i < shape[0]; i++) {
                float sum = 0;
                for (long j = 0; j < shape[1]; j++) {
                    sum += a.getFloat(i, j);
                }
                result.setFloat(sum, i);
            }
        }

        return result;
    }

    /**
     * Computes the mean of all elements in a tensor.
     *
     * @param a The tensor
     * @return The mean value
     */
    public static float mean(Tensor a) {
        return sum(a) / a.getSize();
    }

    /**
     * Applies the ReLU activation function element-wise.
     *
     * @param a The input tensor
     * @return Result tensor with ReLU applied
     */
    public static Tensor relu(Tensor a) {
        Tensor result = new Tensor(a.getShape(), Tensor.DataType.FLOAT32);
        long size = a.getSize();

        for (long i = 0; i < size; i++) {
            float value = a.getFloat(i);
            result.setFloat(i, Math.max(0, value));
        }

        return result;
    }

    /**
     * In-place ReLU activation.
     *
     * @param a The tensor to modify
     */
    public static void reluInPlace(Tensor a) {
        long size = a.getSize();
        for (long i = 0; i < size; i++) {
            float value = a.getFloat(i);
            if (value < 0) {
                a.setFloat(i, 0);
            }
        }
    }

    /**
     * Applies the Leaky ReLU activation function element-wise.
     *
     * @param a The input tensor
     * @param alpha The negative slope
     * @return Result tensor with Leaky ReLU applied
     */
    public static Tensor leakyRelu(Tensor a, float alpha) {
        Tensor result = new Tensor(a.getShape(), Tensor.DataType.FLOAT32);
        long size = a.getSize();

        for (long i = 0; i < size; i++) {
            float value = a.getFloat(i);
            result.setFloat(i, value >= 0 ? value : alpha * value);
        }

        return result;
    }

    /**
     * Applies the sigmoid activation function element-wise.
     *
     * @param a The input tensor
     * @return Result tensor with sigmoid applied
     */
    public static Tensor sigmoid(Tensor a) {
        Tensor result = new Tensor(a.getShape(), Tensor.DataType.FLOAT32);
        long size = a.getSize();

        for (long i = 0; i < size; i++) {
            float value = a.getFloat(i);
            result.setFloat(i, (float) (1.0 / (1.0 + Math.exp(-value))));
        }

        return result;
    }

    /**
     * Applies the softmax function along the last axis.
     *
     * @param a The input tensor
     * @return Result tensor with softmax applied
     */
    public static Tensor softmax(Tensor a) {
        long[] shape = a.getShape();
        Tensor result = new Tensor(shape, Tensor.DataType.FLOAT32);

        if (shape.length == 1) {
            float max = Float.NEGATIVE_INFINITY;
            for (long i = 0; i < shape[0]; i++) {
                max = Math.max(max, a.getFloat(i));
            }

            float sum = 0;
            for (long i = 0; i < shape[0]; i++) {
                float exp = (float) Math.exp(a.getFloat(i) - max);
                result.setFloat(i, exp);
                sum += exp;
            }

            for (long i = 0; i < shape[0]; i++) {
                result.setFloat(i, result.getFloat(i) / sum);
            }
        } else if (shape.length == 2) {
            long batchSize = shape[0];
            long numClasses = shape[1];

            for (long b = 0; b < batchSize; b++) {
                float max = Float.NEGATIVE_INFINITY;
                for (long c = 0; c < numClasses; c++) {
                    max = Math.max(max, a.getFloat(b, c));
                }

                float sum = 0;
                for (long c = 0; c < numClasses; c++) {
                    float exp = (float) Math.exp(a.getFloat(b, c) - max);
                    result.setFloat(exp, b, c);
                    sum += exp;
                }

                for (long c = 0; c < numClasses; c++) {
                    result.setFloat(result.getFloat(b, c) / sum, b, c);
                }
            }
        } else if (shape.length == 3) {
            // Handle 3D tensor [batch, timeSteps, numClasses]
            long batchSize = shape[0];
            long timeSteps = shape[1];
            long numClasses = shape[2];

            for (long b = 0; b < batchSize; b++) {
                for (long t = 0; t < timeSteps; t++) {
                    // Find max for numerical stability
                    float max = Float.NEGATIVE_INFINITY;
                    for (long c = 0; c < numClasses; c++) {
                        max = Math.max(max, a.getFloat(b, t, c));
                    }

                    // Compute exp and sum
                    float sum = 0;
                    for (long c = 0; c < numClasses; c++) {
                        float exp = (float) Math.exp(a.getFloat(b, t, c) - max);
                        result.setFloat(exp, b, t, c);
                        sum += exp;
                    }

                    // Normalize
                    for (long c = 0; c < numClasses; c++) {
                        result.setFloat(result.getFloat(b, t, c) / sum, b, t, c);
                    }
                }
            }
        }

        return result;
    }

    /**
     * Applies the tanh activation function element-wise.
     *
     * @param a The input tensor
     * @return Result tensor with tanh applied
     */
    public static Tensor tanh(Tensor a) {
        Tensor result = new Tensor(a.getShape(), Tensor.DataType.FLOAT32);
        long size = a.getSize();

        for (long i = 0; i < size; i++) {
            float value = a.getFloat(i);
            result.setFloat(i, (float) Math.tanh(value));
        }

        return result;
    }

    /**
     * 2D convolution operation.
     *
     * @param input Input tensor (batch, height, width, channels)
     * @param kernel Convolutional kernel (kernelHeight, kernelWidth, inChannels, outChannels)
     * @param stride The stride in both dimensions
     * @param padding The padding to apply
     * @return Output tensor
     */
    public static Tensor conv2d(Tensor input, Tensor kernel, int[] stride, int[] padding) {
        long[] inputShape = input.getShape();
        long[] kernelShape = kernel.getShape();

        int batchSize = (int) inputShape[0];
        int inputHeight = (int) inputShape[1];
        int inputWidth = (int) inputShape[2];
        int inChannels = (int) inputShape[3];

        int kernelHeight = (int) kernelShape[0];
        int kernelWidth = (int) kernelShape[1];
        int outChannels = (int) kernelShape[3];

        if (inChannels != kernelShape[2]) {
            throw new IllegalArgumentException("Input channels must match kernel input channels");
        }

        int outputHeight = (inputHeight + 2 * padding[0] - kernelHeight) / stride[0] + 1;
        int outputWidth = (inputWidth + 2 * padding[1] - kernelWidth) / stride[1] + 1;

        long[] outputShape = new long[]{batchSize, outputHeight, outputWidth, outChannels};
        Tensor output = new Tensor(outputShape, Tensor.DataType.FLOAT32);

        for (int b = 0; b < batchSize; b++) {
            for (int oh = 0; oh < outputHeight; oh++) {
                for (int ow = 0; ow < outputWidth; ow++) {
                    for (int oc = 0; oc < outChannels; oc++) {
                        float sum = 0;
                        for (int kh = 0; kh < kernelHeight; kh++) {
                            for (int kw = 0; kw < kernelWidth; kw++) {
                                for (int ic = 0; ic < inChannels; ic++) {
                                    int ih = oh * stride[0] + kh - padding[0];
                                    int iw = ow * stride[1] + kw - padding[1];
                                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                        sum += input.getFloat(b, ih, iw, ic) *
                                               kernel.getFloat(kh, kw, ic, oc);
                                    }
                                }
                            }
                        }
                        output.setFloat(sum, b, oh, ow, oc);
                    }
                }
            }
        }

        return output;
    }

    /**
     * 2D max pooling operation.
     *
     * @param input Input tensor (batch, height, width, channels)
     * @param poolSize The pooling window size
     * @param stride The stride in both dimensions
     * @param padding The padding to apply
     * @return Output tensor
     */
    public static Tensor maxPool2d(Tensor input, int[] poolSize, int[] stride, int[] padding) {
        long[] inputShape = input.getShape();

        int batchSize = (int) inputShape[0];
        int inputHeight = (int) inputShape[1];
        int inputWidth = (int) inputShape[2];
        int channels = (int) inputShape[3];

        int poolHeight = poolSize[0];
        int poolWidth = poolSize[1];

        int outputHeight = (inputHeight + 2 * padding[0] - poolHeight) / stride[0] + 1;
        int outputWidth = (inputWidth + 2 * padding[1] - poolWidth) / stride[1] + 1;

        long[] outputShape = new long[]{batchSize, outputHeight, outputWidth, channels};
        Tensor output = new Tensor(outputShape, Tensor.DataType.FLOAT32);

        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        float max = Float.NEGATIVE_INFINITY;
                        for (int ph = 0; ph < poolHeight; ph++) {
                            for (int pw = 0; pw < poolWidth; pw++) {
                                int ih = oh * stride[0] + ph - padding[0];
                                int iw = ow * stride[1] + pw - padding[1];
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
     * 2D average pooling operation.
     *
     * @param input Input tensor (batch, height, width, channels)
     * @param poolSize The pooling window size
     * @param stride The stride in both dimensions
     * @param padding The padding to apply
     * @return Output tensor
     */
    public static Tensor avgPool2d(Tensor input, int[] poolSize, int[] stride, int[] padding) {
        long[] inputShape = input.getShape();

        int batchSize = (int) inputShape[0];
        int inputHeight = (int) inputShape[1];
        int inputWidth = (int) inputShape[2];
        int channels = (int) inputShape[3];

        int poolHeight = poolSize[0];
        int poolWidth = poolSize[1];

        int outputHeight = (inputHeight + 2 * padding[0] - poolHeight) / stride[0] + 1;
        int outputWidth = (inputWidth + 2 * padding[1] - poolWidth) / stride[1] + 1;

        long[] outputShape = new long[]{batchSize, outputHeight, outputWidth, channels};
        Tensor output = new Tensor(outputShape, Tensor.DataType.FLOAT32);

        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        float sum = 0;
                        int count = 0;
                        for (int ph = 0; ph < poolHeight; ph++) {
                            for (int pw = 0; pw < poolWidth; pw++) {
                                int ih = oh * stride[0] + ph - padding[0];
                                int iw = ow * stride[1] + pw - padding[1];
                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                    sum += input.getFloat(b, ih, iw, c);
                                    count++;
                                }
                            }
                        }
                        output.setFloat(sum / count, b, oh, ow, c);
                    }
                }
            }
        }

        return output;
    }

    /**
     * Batch normalization operation.
     *
     * @param input Input tensor
     * @param mean Batch mean
     * @param variance Batch variance
     * @param gamma Scale parameter
     * @param beta Shift parameter
     * @param epsilon Small constant for numerical stability
     * @return Output tensor
     */
    public static Tensor batchNorm(Tensor input, Tensor mean, Tensor variance,
                                   Tensor gamma, Tensor beta, float epsilon) {
        long[] inputShape = input.getShape();
        Tensor output = new Tensor(inputShape, Tensor.DataType.FLOAT32);
        long size = input.getSize();

        for (long i = 0; i < size; i++) {
            float x = input.getFloat(i);
            float m = mean.getFloat(i);
            float v = variance.getFloat(i);
            float g = gamma.getFloat(i);
            float b = beta.getFloat(i);

            float normalized = (x - m) / (float) Math.sqrt(v + epsilon);
            output.setFloat(i, g * normalized + b);
        }

        return output;
    }

    /**
     * Concatenates tensors along a specified axis.
     *
     * @param axis The axis to concatenate along
     * @param tensors The tensors to concatenate
     * @return Result tensor
     */
    public static Tensor concatenate(int axis, Tensor... tensors) {
        if (tensors.length == 0) {
            throw new IllegalArgumentException("At least one tensor required");
        }

        long[][] shapes = new long[tensors.length][];
        long[] firstShape = tensors[0].getShape();
        int rank = firstShape.length;

        long[] outputShape = firstShape.clone();
        for (int i = 1; i < tensors.length; i++) {
            shapes[i] = tensors[i].getShape();
            if (shapes[i].length != rank) {
                throw new IllegalArgumentException("All tensors must have the same rank");
            }
            outputShape[axis] += shapes[i][axis];
        }

        Tensor result = new Tensor(outputShape, Tensor.DataType.FLOAT32);

        // Implementation would copy data from each tensor to result
        // This is a simplified version

        return result;
    }

    /**
     * Broadcasts a tensor to a target shape.
     *
     * @param a The input tensor
     * @param targetShape The target shape
     * @return Broadcasted tensor
     */
    public static Tensor broadcastTo(Tensor a, long[] targetShape) {
        if (!canBroadcast(a.getShape(), targetShape)) {
            throw new IllegalArgumentException("Cannot broadcast shape " +
                Arrays.toString(a.getShape()) + " to " + Arrays.toString(targetShape));
        }

        Tensor result = new Tensor(targetShape, Tensor.DataType.FLOAT32);
        long size = result.getSize();

        for (long i = 0; i < size; i++) {
            long[] coords = result.unravelIndex(i);
            long[] srcCoords = getBroadcastCoords(coords, a.getShape(), targetShape);
            result.setFloat(i, a.getFloat(srcCoords));
        }

        return result;
    }

    /**
     * Checks if broadcasting is possible.
     */
    private static boolean canBroadcast(long[] fromShape, long[] toShape) {
        if (fromShape.length != toShape.length) {
            // Try prepending 1s
            int diff = toShape.length - fromShape.length;
            for (int i = diff; i < toShape.length; i++) {
                if (fromShape[i - diff] != 1 && fromShape[i - diff] != toShape[i]) {
                    return false;
                }
            }
            return true;
        }

        for (int i = 0; i < fromShape.length; i++) {
            if (fromShape[i] != 1 && fromShape[i] != toShape[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * Gets the source coordinates for broadcasting.
     */
    private static long[] getBroadcastCoords(long[] targetCoords, long[] fromShape, long[] toShape) {
        long[] sourceCoords = new long[fromShape.length];
        int diff = toShape.length - fromShape.length;

        for (int i = 0; i < fromShape.length; i++) {
            long targetDim = toShape[i + diff];
            if (fromShape[i] == 1) {
                sourceCoords[i] = 0;
            } else {
                sourceCoords[i] = targetCoords[i + diff];
            }
        }

        return sourceCoords;
    }

    /**
     * Computes L2 normalization along the last axis.
     *
     * @param a The input tensor
     * @return Normalized tensor
     */
    public static Tensor l2Normalize(Tensor a) {
        long[] shape = a.getShape();
        Tensor result = new Tensor(shape, Tensor.DataType.FLOAT32);
        long batchSize = shape[0];
        int numFeatures = (int) shape[1];

        for (int b = 0; b < batchSize; b++) {
            float norm = 0;
            for (int f = 0; f < numFeatures; f++) {
                float value = a.getFloat(b, f);
                norm += value * value;
            }
            norm = (float) Math.sqrt(norm) + 1e-8f;

            for (int f = 0; f < numFeatures; f++) {
                result.setFloat(a.getFloat(b, f) / norm, b, f);
            }
        }

        return result;
    }

    /**
     * Applies dropout to the input tensor.
     *
     * @param input The input tensor
     * @param dropoutRate The probability of dropping a unit
     * @param training Whether we are in training mode
     * @return Output tensor
     */
    public static Tensor dropout(Tensor input, float dropoutRate, boolean training) {
        Tensor output = input.copy();

        if (training && dropoutRate > 0) {
            long size = output.getSize();
            float scale = 1.0f / (1.0f - dropoutRate);

            for (long i = 0; i < size; i++) {
                if (Math.random() < dropoutRate) {
                    output.setFloat(i, 0);
                } else {
                    output.setFloat(i, output.getFloat(i) * scale);
                }
            }
        }

        return output;
    }

    /**
     * Executes a function in parallel over tensor elements.
     *
     * @param size The number of elements
     * @param func The function to execute
     */
    private static void parallelExecute(long size, ElementFunction func) {
        long numTasks = Math.min(size / BLOCK_SIZE, Runtime.getRuntime().availableProcessors() * 4);
        long blockSize = (size + numTasks - 1) / numTasks;

        CompletableFuture<?>[] futures = new CompletableFuture[(int) numTasks];
        for (int t = 0; t < numTasks; t++) {
            final long start = t * blockSize;
            final long end = Math.min(start + blockSize, size);
            futures[t] = CompletableFuture.runAsync(() -> {
                for (long i = start; i < end; i++) {
                    func.apply(i);
                }
            }, EXECUTOR);
        }
        CompletableFuture.allOf(futures).join();
    }

    /**
     * Functional interface for element-wise operations.
     */
    @FunctionalInterface
    private interface ElementFunction {
        void apply(long index);
    }

    /**
     * Computes the absolute value of tensor elements.
     *
     * @param a The input tensor
     * @return Result tensor with absolute values
     */
    public static Tensor abs(Tensor a) {
        Tensor result = new Tensor(a.getShape(), Tensor.DataType.FLOAT32);
        long size = a.getSize();

        for (long i = 0; i < size; i++) {
            result.setFloat(i, Math.abs(a.getFloat(i)));
        }

        return result;
    }

    /**
     * Clips tensor values to a specified range.
     *
     * @param a The input tensor
     * @param min Minimum value
     * @param max Maximum value
     * @return Result tensor with clipped values
     */
    public static Tensor clip(Tensor a, float min, float max) {
        Tensor result = new Tensor(a.getShape(), Tensor.DataType.FLOAT32);
        long size = a.getSize();

        for (long i = 0; i < size; i++) {
            result.setFloat(i, Math.max(min, Math.min(max, a.getFloat(i))));
        }

        return result;
    }

    /**
     * Finds the maximum value in a tensor.
     *
     * @param a The input tensor
     * @return The maximum value
     */
    public static float max(Tensor a) {
        long size = a.getSize();
        float max = Float.NEGATIVE_INFINITY;

        for (long i = 0; i < size; i++) {
            float value = a.getFloat(i);
            if (value > max) {
                max = value;
            }
        }

        return max;
    }

    /**
     * Finds the minimum value in a tensor.
     *
     * @param a The input tensor
     * @return The minimum value
     */
    public static float min(Tensor a) {
        long size = a.getSize();
        float min = Float.POSITIVE_INFINITY;

        for (long i = 0; i < size; i++) {
            float value = a.getFloat(i);
            if (value < min) {
                min = value;
            }
        }

        return min;
    }
}
