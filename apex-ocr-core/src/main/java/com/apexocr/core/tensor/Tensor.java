package com.apexocr.core.tensor;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Tensor - High-performance multi-dimensional array implementation for neural network operations.
 * Designed to outperform traditional matrix libraries through aggressive optimization and
 * utilization of Java's Vector API for SIMD operations.
 *
 * This class serves as the foundational data structure for all neural network computations
 * in the ApexOCR engine, optimized for both memory efficiency and computational performance.
 *
 * @author ApexOCR Team
 * @version 1.0.0
 */
public class Tensor implements AutoCloseable {
    private final long[] shape;
    private final long[] strides;
    private final long size;
    private final int rank;
    private final DataType dataType;
    private final MemoryRegion memoryRegion;
    private final boolean isView;

    /**
     * Enumeration of supported data types for tensor operations.
     * Each type has specific use cases in neural network inference.
     */
    public enum DataType {
        FLOAT32(4, Float.BYTES),
        FLOAT16(2, 2),
        INT32(4, Integer.BYTES),
        INT8(1, Byte.BYTES),
        UINT8(1, Byte.BYTES);

        private final int byteSize;
        private final int elementSize;

        DataType(int byteSize, int elementSize) {
            this.byteSize = byteSize;
            this.elementSize = elementSize;
        }

        public int getByteSize() {
            return byteSize;
        }

        public int getElementSize() {
            return elementSize;
        }
    }

    /**
     * Memory region abstraction for managing off-heap memory allocation.
     * Enables direct memory operations without garbage collection overhead.
     */
    public static class MemoryRegion implements AutoCloseable {
        private final long address;
        private final long byteSize;
        private final boolean external;

        public MemoryRegion(long address, long byteSize) {
            this.address = address;
            this.byteSize = byteSize;
            this.external = false;
        }

        public MemoryRegion(long address, long byteSize, boolean external) {
            this.address = address;
            this.byteSize = byteSize;
            this.external = external;
        }

        public long getAddress() {
            return address;
        }

        public long getByteSize() {
            return byteSize;
        }

        public boolean isExternal() {
            return external;
        }

        @Override
        public void close() {
            if (!external && address != 0) {
                MemoryManager.free(address);
            }
        }
    }

    /**
     * Creates a new tensor with the specified shape and data type.
     * Allocates contiguous memory for optimal cache performance.
     *
     * @param shape The dimensions of the tensor
     * @param dataType The data type for tensor elements
     */
    public Tensor(long[] shape, DataType dataType) {
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Shape must be non-null and non-empty");
        }
        this.shape = shape.clone();
        this.rank = shape.length;
        this.dataType = dataType;

        this.size = calculateSize(shape);
        this.strides = calculateStrides(shape);
        long byteSize = size * dataType.getByteSize();
        this.memoryRegion = MemoryManager.allocate(byteSize);
        this.isView = false;
    }

    /**
     * Creates a tensor as a view into existing memory.
     * Useful for zero-copy operations and weight sharing.
     *
     * @param shape The dimensions of the tensor
     * @param strides The stride values for each dimension
     * @param memoryRegion The memory region to view into
     */
    public Tensor(long[] shape, long[] strides, MemoryRegion memoryRegion) {
        this.shape = shape.clone();
        this.rank = shape.length;
        this.strides = strides.clone();
        this.size = calculateSize(shape);
        this.dataType = DataType.FLOAT32;
        this.memoryRegion = memoryRegion;
        this.isView = true;
    }

    /**
     * Private constructor for creating tensor views.
     */
    private Tensor(long[] shape, long[] strides, long size,
                   DataType dataType, MemoryRegion memoryRegion, boolean isView) {
        this.shape = shape;
        this.strides = strides;
        this.size = size;
        this.rank = shape.length;
        this.dataType = dataType;
        this.memoryRegion = memoryRegion;
        this.isView = isView;
    }

    /**
     * Calculates the total number of elements in the tensor.
     */
    private long calculateSize(long[] shape) {
        long size = 1;
        for (long dim : shape) {
            size *= dim;
        }
        return size;
    }

    /**
     * Calculates stride values for efficient multi-dimensional access.
     * Uses row-major (C-style) ordering for compatibility.
     */
    private long[] calculateStrides(long[] shape) {
        long[] strides = new long[shape.length];
        long stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    /**
     * Gets the shape of the tensor.
     *
     * @return A copy of the shape array
     */
    public long[] getShape() {
        return shape.clone();
    }

    /**
     * Gets the rank (number of dimensions) of the tensor.
     *
     * @return The rank
     */
    public int getRank() {
        return rank;
    }

    /**
     * Gets the total number of elements in the tensor.
     *
     * @return The size
     */
    public long getSize() {
        return size;
    }

    /**
     * Gets the data type of the tensor.
     *
     * @return The data type
     */
    public DataType getDataType() {
        return dataType;
    }

    /**
     * Checks if this tensor is a view of another tensor.
     *
     * @return True if this is a view tensor
     */
    public boolean isView() {
        return isView;
    }

    /**
     * Gets the memory region backing this tensor.
     *
     * @return The memory region
     */
    public MemoryRegion getMemoryRegion() {
        return memoryRegion;
    }

    /**
     * Gets the underlying byte buffer for direct memory access.
     *
     * @return A direct byte buffer backed by tensor memory
     */
    public ByteBuffer getBuffer() {
        return MemoryManager.getBuffer(memoryRegion.getAddress(), memoryRegion.getByteSize());
    }

    /**
     * Gets a float value at the specified linear index.
     *
     * @param index The linear index
     * @return The float value at the index
     */
    public float getFloat(long index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
        }
        return MemoryManager.getFloat(memoryRegion.getAddress(), index, dataType);
    }

    /**
     * Sets a float value at the specified linear index.
     *
     * @param index The linear index
     * @param value The value to set
     */
    public void setFloat(long index, float value) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
        }
        MemoryManager.setFloat(memoryRegion.getAddress(), index, value, dataType);
    }

    /**
     * Gets a float value at the specified multi-dimensional coordinates.
     *
     * @param coordinates The coordinates for each dimension
     * @return The float value at the coordinates
     */
    public float getFloat(long... coordinates) {
        long index = linearIndex(coordinates);
        return getFloat(index);
    }

    /**
     * Sets a float value at the specified multi-dimensional coordinates.
     *
     * @param value The value to set
     * @param coordinates The coordinates for each dimension
     */
    public void setFloat(float value, long... coordinates) {
        long index = linearIndex(coordinates);
        setFloat(index, value);
    }

    /**
     * Converts multi-dimensional coordinates to a linear index.
     *
     * @param coordinates The coordinates
     * @return The linear index
     */
    public long linearIndex(long[] coordinates) {
        if (coordinates.length != rank) {
            throw new IllegalArgumentException("Coordinate length must match tensor rank");
        }
        long index = 0;
        for (int i = 0; i < rank; i++) {
            if (coordinates[i] < 0 || coordinates[i] >= shape[i]) {
                throw new IndexOutOfBoundsException(
                    "Coordinate[" + i + "]=" + coordinates[i] +
                    " out of bounds for dimension " + i + " with size " + shape[i]);
            }
            index += coordinates[i] * strides[i];
        }
        return index;
    }

    /**
     * Converts a linear index to multi-dimensional coordinates.
     *
     * @param linearIndex The linear index
     * @return The coordinates array
     */
    public long[] unravelIndex(long linearIndex) {
        if (linearIndex < 0 || linearIndex >= size) {
            throw new IndexOutOfBoundsException("Index: " + linearIndex + ", Size: " + size);
        }
        long[] coordinates = new long[rank];
        long remaining = linearIndex;
        for (int i = rank - 1; i >= 0; i--) {
            coordinates[i] = remaining % shape[i];
            remaining /= shape[i];
        }
        return coordinates;
    }

    /**
     * Fills the tensor with the specified value using parallel execution.
     *
     * @param value The value to fill
     */
    public void fill(float value) {
        if (size <= 10000) {
            // Sequential fill for small tensors
            for (long i = 0; i < size; i++) {
                setFloat(i, value);
            }
        } else {
            // Parallel fill for larger tensors
            ForkJoinPool.commonPool().invoke(new FillTask(0, size, value));
        }
    }

    /**
     * Parallel fill task for large tensor operations.
     */
    private class FillTask extends RecursiveAction {
        private static final long serialVersionUID = 1L;
        private final long start;
        private final long end;
        private final float value;

        FillTask(long start, long end, float value) {
            this.start = start;
            this.end = end;
            this.value = value;
        }

        @Override
        protected void compute() {
            if (end - start <= 10000) {
                for (long i = start; i < end; i++) {
                    setFloat(i, value);
                }
            } else {
                long mid = (start + end) / 2;
                invokeAll(
                    new FillTask(start, mid, value),
                    new FillTask(mid, end, value)
                );
            }
        }
    }

    /**
     * Initializes the tensor with random values from a normal distribution.
     *
     * @param mean The mean of the distribution
     * @param std The standard deviation
     */
    public void randomNormal(float mean, float std) {
        for (long i = 0; i < size; i++) {
            float value = (float) (mean + std * ThreadLocalRandom.current().nextGaussian());
            setFloat(i, value);
        }
    }

    /**
     * Initializes the tensor with random values from a uniform distribution.
     *
     * @param min The minimum value
     * @param max The maximum value
     */
    public void randomUniform(float min, float max) {
        for (long i = 0; i < size; i++) {
            float value = min + (float) ThreadLocalRandom.current().nextDouble() * (max - min);
            setFloat(i, value);
        }
    }

    /**
     * Creates a shallow copy of this tensor (shares underlying memory).
     *
     * @return A new tensor that shares memory with this tensor
     */
    public Tensor view() {
        return new Tensor(shape, strides, size, dataType, memoryRegion, true);
    }

    /**
     * Creates a copy of this tensor with new allocated memory.
     *
     * @return A new tensor with independent memory
     */
    public Tensor copy() {
        Tensor result = new Tensor(shape, dataType);
        TensorOperations.copy(this, result);
        return result;
    }

    /**
     * Reshapes the tensor without changing underlying data.
     * The new shape must have the same total size.
     *
     * @param newShape The new shape
     * @return A reshaped view of this tensor
     */
    public Tensor reshape(long... newShape) {
        long newSize = calculateSize(newShape);
        if (newSize != this.size) {
            throw new IllegalArgumentException(
                "Cannot reshape tensor of size " + this.size +
                " into shape " + Arrays.toString(newShape));
        }
        long[] newStrides = calculateStrides(newShape);
        return new Tensor(newShape, newStrides, size, dataType, memoryRegion, true);
    }

    /**
     * Extracts a subtensor using the specified ranges.
     *
     * @param ranges The ranges for each dimension [start, end)
     * @return A subtensor view
     */
    public Tensor slice(long[]... ranges) {
        if (ranges.length != rank) {
            throw new IllegalArgumentException("Number of ranges must match tensor rank");
        }

        long[] newShape = new long[rank];
        long[] offsetCoordinates = new long[rank];

        for (int i = 0; i < rank; i++) {
            long start = ranges[i][0];
            long end = ranges[i][1];
            if (end < start) {
                throw new IllegalArgumentException("End must be >= start for dimension " + i);
            }
            newShape[i] = end - start;
            offsetCoordinates[i] = start;
        }

        long offset = linearIndex(offsetCoordinates);
        long newByteOffset = offset * dataType.getByteSize();

        MemoryRegion newRegion = new MemoryRegion(
            memoryRegion.getAddress() + newByteOffset,
            calculateSize(newShape) * dataType.getByteSize()
        );

        return new Tensor(newShape, strides, newRegion);
    }

    /**
     * Transposes the tensor by reversing dimensions.
     *
     * @return A transposed view of this tensor
     */
    public Tensor transpose() {
        if (rank != 2) {
            throw new UnsupportedOperationException(
                "Transpose only supported for 2D tensors, got rank " + rank);
        }
        long[] newShape = new long[]{shape[1], shape[0]};
        long[] newStrides = new long[]{strides[1], strides[0]};
        return new Tensor(newShape, newStrides, size, dataType, memoryRegion, true);
    }

    /**
     * Squeezes singleton dimensions from the tensor.
     *
     * @return A new tensor with singleton dimensions removed
     */
    public Tensor squeeze() {
        int newRank = 0;
        for (long dim : shape) {
            if (dim != 1) newRank++;
        }

        if (newRank == rank) {
            return this;
        }

        long[] newShape = new long[newRank];
        long[] newStrides = new long[newRank];
        int j = 0;
        for (int i = 0; i < rank; i++) {
            if (shape[i] != 1) {
                newShape[j] = shape[i];
                newStrides[j] = strides[i];
                j++;
            }
        }
        return new Tensor(newShape, newStrides, size, dataType, memoryRegion, true);
    }

    /**
     * Unsqueezes a dimension at the specified position.
     *
     * @param position The position to insert the new dimension
     * @return A new tensor with an added singleton dimension
     */
    public Tensor unsqueeze(int position) {
        if (position < 0 || position > rank) {
            throw new IllegalArgumentException("Position must be in [0, " + rank + "]");
        }

        long[] newShape = new long[rank + 1];
        long[] newStrides = new long[rank + 1];
        long stride = 1;
        int j = 0;

        for (int i = 0; i <= rank; i++) {
            if (i == position) {
                newShape[i] = 1;
                newStrides[i] = stride;
            } else {
                newShape[i] = shape[j];
                newStrides[i] = strides[j] * stride;
                j++;
            }
            stride *= newShape[i];
        }

        return new Tensor(newShape, newStrides, size, dataType, memoryRegion, true);
    }

    /**
     * Gets a summary string representation of the tensor.
     *
     * @return A summary string
     */
    public String summary() {
        return String.format("Tensor(shape=%s, size=%d, dtype=%s, isView=%s)",
            Arrays.toString(shape), size, dataType, isView);
    }

    @Override
    public void close() {
        if (!isView) {
            memoryRegion.close();
        }
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof Tensor other)) return false;
        if (this.size != other.size) return false;
        if (this.dataType != other.dataType) return false;

        for (long i = 0; i < size; i++) {
            if (Float.floatToIntBits(this.getFloat(i)) != Float.floatToIntBits(other.getFloat(i))) {
                return false;
            }
        }
        return true;
    }

    @Override
    public int hashCode() {
        int result = 1;
        for (long i = 0; i < size && i < 100; i++) { // Sample for performance
            result = 31 * result + Float.hashCode(getFloat(i));
        }
        return result;
    }

    @Override
    public String toString() {
        if (size > 1000) {
            return String.format("Tensor(shape=%s, size=%d, dtype=%s, ...) {\n  [first 100 elements]: %s\n  ...\n}",
                Arrays.toString(shape), size, dataType, getFloatArray(100));
        }
        return "Tensor(shape=" + Arrays.toString(shape) +
               ", data=" + Arrays.toString(getFloatArray((int) size)) + ")";
    }

    /**
     * Gets the tensor data as a float array.
     * Note: This creates a copy and may be expensive for large tensors.
     *
     * @param limit Maximum number of elements to retrieve
     * @return Float array containing tensor data
     */
    public float[] getFloatArray(int limit) {
        int len = (int) Math.min(size, limit);
        float[] result = new float[len];
        for (int i = 0; i < len; i++) {
            result[i] = getFloat(i);
        }
        return result;
    }

    /**
     * Gets the tensor data as a 2D float array if the tensor is 2D.
     *
     * @return 2D float array
     */
    public float[][] getFloatArray2D() {
        if (rank != 2) {
            throw new IllegalStateException("Tensor must be 2D to convert to 2D array");
        }
        int rows = (int) shape[0];
        int cols = (int) shape[1];
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = getFloat(i, j);
            }
        }
        return result;
    }

    /**
     * Gets the tensor data from a 2D float array.
     *
     * @param data The 2D float array
     * @return This tensor for method chaining
     */
    public Tensor setFloatArray2D(float[][] data) {
        if (rank != 2) {
            throw new IllegalStateException("Tensor must be 2D to set from 2D array");
        }
        int rows = (int) shape[0];
        int cols = (int) shape[1];
        if (data.length != rows || data[0].length != cols) {
            throw new IllegalArgumentException("Array dimensions do not match tensor shape");
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                setFloat(data[i][j], i, j);
            }
        }
        return this;
    }
}
