package com.apexocr.core.tensor;

import sun.misc.Unsafe;

import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.security.AccessController;
import java.security.PrivilegedAction;

/**
 * MemoryManager - High-performance memory allocation and access utility for tensor operations.
 * Provides direct memory access capabilities for neural network inference without
 * the overhead of garbage collection and object allocation.
 *
 * This class leverages sun.misc.Unsafe for direct memory manipulation, enabling
 * efficient off-heap memory management critical for high-performance tensor operations.
 *
 * @author ApexOCR Team
 * @version 1.0.0
 */
public final class MemoryManager {
    private static final Unsafe UNSAFE;
    private static final long BASE_OFFSET;
    private static final int PAGE_SIZE;

    static {
        UNSAFE = AccessController.doPrivileged((PrivilegedAction<Unsafe>) () -> {
            try {
                Field theUnsafe = Unsafe.class.getDeclaredField("theUnsafe");
                theUnsafe.setAccessible(true);
                return (Unsafe) theUnsafe.get(null);
            } catch (Exception e) {
                throw new RuntimeException("Failed to access Unsafe", e);
            }
        });

        BASE_OFFSET = UNSAFE.arrayBaseOffset(float[].class);
        PAGE_SIZE = 4096;

        // Pre-warm the allocator
        warmUp();
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private MemoryManager() {
        throw new UnsupportedOperationException("Utility class cannot be instantiated");
    }

    /**
     * Allocates a contiguous block of off-heap memory.
     *
     * @param byteSize The size in bytes to allocate
     * @return A MemoryRegion containing the allocated memory
     */
    public static Tensor.MemoryRegion allocate(long byteSize) {
        if (byteSize <= 0) {
            throw new IllegalArgumentException("Byte size must be positive");
        }

        // Round up to page size for better performance
        long paddedSize = ((byteSize + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
        long address = UNSAFE.allocateMemory(paddedSize);

        if (address == 0) {
            throw new OutOfMemoryError("Failed to allocate " + paddedSize + " bytes");
        }

        // Initialize memory to zero
        UNSAFE.setMemory(address, paddedSize, (byte) 0);

        return new Tensor.MemoryRegion(address, paddedSize);
    }

    /**
     * Frees previously allocated memory.
     *
     * @param address The memory address to free
     */
    public static void free(long address) {
        if (address != 0) {
            UNSAFE.freeMemory(address);
        }
    }

    /**
     * Reallocates memory to a new size.
     *
     * @param address The current memory address
     * @param oldByteSize The current size in bytes
     * @param newByteSize The new size in bytes
     * @return A new MemoryRegion with the reallocated memory
     */
    public static Tensor.MemoryRegion reallocate(long address, long oldByteSize, long newByteSize) {
        if (newByteSize <= 0) {
            throw new IllegalArgumentException("New byte size must be positive");
        }

        long newAddress = UNSAFE.reallocateMemory(address, newByteSize);

        if (newAddress == 0) {
            throw new OutOfMemoryError("Failed to reallocate to " + newByteSize + " bytes");
        }

        // Initialize new memory to zero
        if (newByteSize > oldByteSize) {
            UNSAFE.setMemory(newAddress + oldByteSize, newByteSize - oldByteSize, (byte) 0);
        }

        return new Tensor.MemoryRegion(newAddress, newByteSize);
    }

    /**
     * Gets a direct ByteBuffer backed by the specified memory address.
     *
     * @param address The memory address
     * @param byteSize The size of the memory region
     * @return A direct ByteBuffer
     */
    public static ByteBuffer getBuffer(long address, long byteSize) {
        return ByteBuffer.allocateDirect((int) byteSize)
                .order(ByteOrder.nativeOrder())
                .put(MemoryManager.getBytes(address, 0, (int) byteSize))
                .flip();
    }

    /**
     * Copies bytes from memory to a byte array.
     *
     * @param address The source memory address
     * @param offset The source offset
     * @param length The number of bytes to copy
     * @return A byte array containing the copied data
     */
    public static byte[] getBytes(long address, long offset, int length) {
        byte[] result = new byte[length];
        for (int i = 0; i < length; i++) {
            result[i] = UNSAFE.getByte(address + offset + i);
        }
        return result;
    }

    /**
     * Reads a float value from memory at the specified index.
     *
     * @param address The memory address
     * @param index The element index
     * @param dataType The data type of the elements
     * @return The float value
     */
    public static float getFloat(long address, long index, Tensor.DataType dataType) {
        long byteOffset = index * dataType.getByteSize();
        switch (dataType) {
            case FLOAT32:
                return UNSAFE.getFloat(address + byteOffset);
            case FLOAT16:
                return halfToFloat(UNSAFE.getShort(address + byteOffset));
            case INT32:
                return (float) UNSAFE.getInt(address + byteOffset);
            case INT8:
                return (float) UNSAFE.getByte(address + byteOffset);
            case UINT8:
                return (float) (UNSAFE.getByte(address + byteOffset) & 0xFF);
            default:
                throw new IllegalArgumentException("Unsupported data type: " + dataType);
        }
    }

    /**
     * Writes a float value to memory at the specified index.
     *
     * @param address The memory address
     * @param index The element index
     * @param value The value to write
     * @param dataType The data type of the elements
     */
    public static void setFloat(long address, long index, float value, Tensor.DataType dataType) {
        long byteOffset = index * dataType.getByteSize();
        switch (dataType) {
            case FLOAT32:
                UNSAFE.putFloat(address + byteOffset, value);
                break;
            case FLOAT16:
                UNSAFE.putShort(address + byteOffset, floatToHalf(value));
                break;
            case INT32:
                UNSAFE.putInt(address + byteOffset, (int) value);
                break;
            case INT8:
                UNSAFE.putByte(address + byteOffset, (byte) value);
                break;
            case UINT8:
                UNSAFE.putByte(address + byteOffset, (byte) (int) value);
                break;
            default:
                throw new IllegalArgumentException("Unsupported data type: " + dataType);
        }
    }

    /**
     * Reads a double value from memory at the specified index.
     *
     * @param address The memory address
     * @param index The element index
     * @return The double value
     */
    public static double getDouble(long address, long index) {
        return UNSAFE.getDouble(address + index * Double.BYTES);
    }

    /**
     * Writes a double value to memory at the specified index.
     *
     * @param address The memory address
     * @param index The element index
     * @param value The value to write
     */
    public static void setDouble(long address, long index, double value) {
        UNSAFE.putDouble(address + index * Double.BYTES, value);
    }

    /**
     * Reads an int value from memory at the specified index.
     *
     * @param address The memory address
     * @param index The element index
     * @return The int value
     */
    public static int getInt(long address, long index) {
        return UNSAFE.getInt(address + index * Integer.BYTES);
    }

    /**
     * Writes an int value to memory at the specified index.
     *
     * @param address The memory address
     * @param index The element index
     * @param value The value to write
     */
    public static void setInt(long address, long index, int value) {
        UNSAFE.putInt(address + index * Integer.BYTES, value);
    }

    /**
     * Copies memory from one address to another.
     *
     * @param srcAddress Source memory address
     * @param dstAddress Destination memory address
     * @param byteCount Number of bytes to copy
     */
    public static void copyMemory(long srcAddress, long dstAddress, long byteCount) {
        UNSAFE.copyMemory(srcAddress, dstAddress, byteCount);
    }

    /**
     * Copies memory from a byte array to an address.
     *
     * @param src The source byte array
     * @param srcOffset Offset in the source array
     * @param dstAddress Destination memory address
     * @param byteCount Number of bytes to copy
     */
    public static void copyToAddress(byte[] src, int srcOffset, long dstAddress, int byteCount) {
        Object base = UNSAFE.getObject(src, BASE_OFFSET);
        long srcBase = UNSAFE.getLong(base, BASE_OFFSET - 8);
        UNSAFE.copyMemory(base, srcBase + srcOffset, null, dstAddress, byteCount);
    }

    /**
     * Converts a float array to a memory region.
     *
     * @param data The float array
     * @return A MemoryRegion containing the array data
     */
    public static Tensor.MemoryRegion fromFloatArray(float[] data) {
        Tensor.MemoryRegion region = allocate(data.length * Float.BYTES);
        for (int i = 0; i < data.length; i++) {
            setFloat(region.getAddress(), i, data[i], Tensor.DataType.FLOAT32);
        }
        return region;
    }

    /**
     * Converts a 2D float array to a memory region.
     *
     * @param data The 2D float array
     * @return A MemoryRegion containing the array data
     */
    public static Tensor.MemoryRegion fromFloatArray2D(float[][] data) {
        int rows = data.length;
        int cols = data[0].length;
        Tensor.MemoryRegion region = allocate(rows * cols * Float.BYTES);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                setFloat(region.getAddress(), i * cols + j, data[i][j], Tensor.DataType.FLOAT32);
            }
        }
        return region;
    }

    /**
     * Creates a FloatBuffer view of a memory region.
     *
     * @param address The memory address
     * @param size The number of floats
     * @return A FloatBuffer
     */
    public static FloatBuffer getFloatBuffer(long address, int size) {
        // Note: Direct NIO buffer access requires module exports in Java 9+
        // For now, return a simple implementation
        FloatBuffer buffer = FloatBuffer.allocate(size);
        for (int i = 0; i < size; i++) {
            buffer.put(i, UNSAFE.getFloat(address + i * Float.BYTES));
        }
        return buffer;
    }

    /**
     * Creates an IntBuffer view of a memory region.
     *
     * @param address The memory address
     * @param size The number of ints
     * @return An IntBuffer
     */
    public static IntBuffer getIntBuffer(long address, int size) {
        // Note: Direct NIO buffer access requires module exports in Java 9+
        // For now, return a simple implementation
        IntBuffer buffer = IntBuffer.allocate(size);
        for (int i = 0; i < size; i++) {
            buffer.put(i, UNSAFE.getInt(address + i * Integer.BYTES));
        }
        return buffer;
    }

    /**
     * Gets the base offset for array access.
     *
     * @return The base offset
     */
    public static long getBaseOffset() {
        return BASE_OFFSET;
    }

    /**
     * Fills memory with a specific byte value.
     *
     * @param address The memory address
     * @param byteSize The number of bytes to fill
     * @param value The byte value to fill with
     */
    public static void setMemory(long address, long byteSize, byte value) {
        UNSAFE.setMemory(address, byteSize, value);
    }

    /**
     * Compares two memory regions for equality.
     *
     * @param addr1 First memory address
     * @param addr2 Second memory address
     * @param byteCount Number of bytes to compare
     * @return True if the regions are equal
     */
    public static boolean compareMemory(long addr1, long addr2, long byteCount) {
        long wordCount = byteCount / Long.BYTES;
        int remainingBytes = (int) (byteCount % Long.BYTES);

        for (long i = 0; i < wordCount; i++) {
            if (UNSAFE.getLong(addr1 + i * Long.BYTES) != UNSAFE.getLong(addr2 + i * Long.BYTES)) {
                return false;
            }
        }

        for (int i = 0; i < remainingBytes; i++) {
            if (UNSAFE.getByte(addr1 + wordCount * Long.BYTES + i) !=
                UNSAFE.getByte(addr2 + wordCount * Long.BYTES + i)) {
                return false;
            }
        }

        return true;
    }

    /**
     * Computes the memory checksum (simple XOR-based).
     *
     * @param address The memory address
     * @param byteSize The number of bytes to checksum
     * @return The checksum value
     */
    public static long checksum(long address, long byteSize) {
        long checksum = 0;
        long wordCount = byteSize / Long.BYTES;
        int remainingBytes = (int) (byteSize % Long.BYTES);

        for (long i = 0; i < wordCount; i++) {
            checksum ^= UNSAFE.getLong(address + i * Long.BYTES);
        }

        for (int i = 0; i < remainingBytes; i++) {
            checksum ^= UNSAFE.getByte(address + wordCount * Long.BYTES + i);
        }

        return checksum;
    }

    /**
     * Warms up the memory allocator to reduce first-run overhead.
     */
    private static void warmUp() {
        // Allocate and free a small block to warm up the allocator
        Tensor.MemoryRegion warmup = allocate(PAGE_SIZE);
        free(warmup.getAddress());

        // Touch pages to ensure they're mapped
        Tensor.MemoryRegion warmup2 = allocate(PAGE_SIZE * 2);
        UNSAFE.setMemory(warmup2.getAddress(), PAGE_SIZE * 2, (byte) 0xFF);
        free(warmup2.getAddress());
    }

    /**
     * Converts a 16-bit half-precision float to 32-bit float.
     * Based on IEEE 754 half-precision format.
     *
     * @param half The 16-bit half value
     * @return The 32-bit float value
     */
    private static short floatToHalf(float value) {
        int floatBits = Float.floatToIntBits(value);
        int sign = (floatBits >>> 16) & 0x8000;
        int exponent = ((floatBits >>> 23) & 0xFF) - 127;
        int mantissa = floatBits & 0x007FFFFF;

        // Handle special cases
        if (exponent == 128) {
            // Infinity or NaN
            return (short) (sign | 0x7C00 | (mantissa >>> 13));
        }
        if (exponent < -14) {
            // Subnormal or zero
            int shift = 14 - exponent;
            int mantissaShifted = mantissa | 0x00800000;
            int rounding = mantissaShifted >>> (shift + 1);
            if ((mantissaShifted & (1 << shift)) != 0) {
                rounding++;
            }
            return (short) (sign | (rounding >>> 1));
        }

        // Normal number
        return (short) (sign | ((exponent + 15) << 10) | (mantissa >>> 13));
    }

    /**
     * Converts a 32-bit float to 16-bit half-precision float.
     *
     * @param half The 16-bit half value
     * @return The 32-bit float value
     */
    private static float halfToFloat(short half) {
        int sign = (half >>> 15) & 0x00000001;
        int exponent = (half >>> 10) & 0x0000001F;
        int mantissa = half & 0x000003FF;

        int floatBits;

        if (exponent == 0) {
            // Subnormal or zero
            floatBits = sign << 31 | (mantissa != 0 ? 0x3F000000 | mantissa << 13 : 0);
        } else if (exponent == 31) {
            // Infinity or NaN
            floatBits = sign << 31 | 0x7F800000 | mantissa;
        } else {
            // Normal number
            floatBits = sign << 31 | (exponent - 15 + 127) << 23 | mantissa << 13;
        }

        return Float.intBitsToFloat(floatBits);
    }

    /**
     * Validates a memory address.
     *
     * @param address The address to validate
     * @return True if the address is valid
     */
    public static boolean isValidAddress(long address) {
        return address != 0;
    }

    /**
     * Gets the memory usage statistics.
     *
     * @return A string containing memory statistics
     */
    public static String getMemoryStats() {
        Runtime runtime = Runtime.getRuntime();
        long maxMemory = runtime.maxMemory() / (1024 * 1024);
        long totalMemory = runtime.totalMemory() / (1024 * 1024);
        long freeMemory = runtime.freeMemory() / (1024 * 1024);
        long usedMemory = totalMemory - freeMemory;

        return String.format(
            "JVM Memory - Max: %d MB, Total: %d MB, Used: %d MB, Free: %d MB",
            maxMemory, totalMemory, usedMemory, freeMemory
        );
    }
}
