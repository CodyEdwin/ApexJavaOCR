package com.apexocr.core.monitoring;

import java.util.List;
import java.util.ArrayList;

/**
 * Description of the neural network architecture for visualization.
 * Contains layer information needed to render the 2D network structure.
 */
public class NetworkArchitecture {
    
    public final String name;
    public final List<LayerInfo> layers;
    public final int totalParameters;
    public final int inputSize;
    public final int outputSize;
    public final float totalComputation;
    
    public NetworkArchitecture(Builder builder) {
        this.name = builder.name;
        this.layers = List.copyOf(builder.layers);
        this.totalParameters = builder.totalParameters;
        this.inputSize = builder.inputSize;
        this.outputSize = builder.outputSize;
        this.totalComputation = builder.totalComputation;
    }
    
    /**
     * Create a builder for constructing network architecture.
     * @return New builder instance
     */
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * Get the number of layers in the network.
     * @return Number of layers
     */
    public int getLayerCount() {
        return layers.size();
    }
    
    /**
     * Get a specific layer by index.
     * @param index Layer index
     * @return Layer info if index is valid
     */
    public LayerInfo getLayer(int index) {
        if (index >= 0 && index < layers.size()) {
            return layers.get(index);
        }
        return null;
    }
    
    /**
     * Get the index of a layer by name.
     * @param layerName Name of the layer
     * @return Layer index, or -1 if not found
     */
    public int getLayerIndex(String layerName) {
        for (int i = 0; i < layers.size(); i++) {
            if (layers.get(i).name.equals(layerName)) {
                return i;
            }
        }
        return -1;
    }
    
    /**
     * Get total number of trainable parameters.
     * @return Total parameter count
     */
    public int getTotalParameters() {
        return totalParameters;
    }
    
    /**
     * Layer information for visualization.
     */
    public static class LayerInfo {
        public final String name;
        public final LayerSnapshot.LayerType type;
        public final int inputChannels;
        public final int outputChannels;
        public final int height;
        public final int width;
        public final int kernelSize;
        public final int stride;
        public final int parameters;
        public final float computationalCost;
        public final float[] position; // [x, y, z] position in 3D space
        
        public LayerInfo(Builder builder) {
            this.name = builder.name;
            this.type = builder.type;
            this.inputChannels = builder.inputChannels;
            this.outputChannels = builder.outputChannels;
            this.height = builder.height;
            this.width = builder.width;
            this.kernelSize = builder.kernelSize;
            this.stride = builder.stride;
            this.parameters = builder.parameters;
            this.computationalCost = builder.computationalCost;
            this.position = builder.position != null ? builder.position : new float[]{0, 0, 0};
        }
        
        /**
         * Get a human-readable description of the layer.
         * @return Layer description
         */
        public String getDescription() {
            switch (type) {
                case INPUT:
                    return String.format("Input: %dx%d", width, height);
                case CONV2D:
                    return String.format("Conv2D: %dx%d kernel, %d filters", kernelSize, kernelSize, outputChannels);
                case BILSTM:
                    return String.format("BiLSTM: %d units", outputChannels);
                case DENSE:
                    return String.format("Dense: %d neurons", outputChannels);
                case ACTIVATION:
                    return "Activation";
                case DROPOUT:
                    return String.format("Dropout: %.1f%%", 0.0f);
                case POOLING:
                    return String.format("Pooling: %dx%d", kernelSize, kernelSize);
                case OUTPUT:
                    return String.format("Output: %d classes", outputChannels);
                default:
                    return name;
            }
        }
        
        /**
         * Builder for constructing LayerInfo.
         */
        public static class Builder {
            private String name = "Unnamed";
            private LayerSnapshot.LayerType type = LayerSnapshot.LayerType.DENSE;
            private int inputChannels = 0;
            private int outputChannels = 0;
            private int height = 0;
            private int width = 0;
            private int kernelSize = 1;
            private int stride = 1;
            private int parameters = 0;
            private float computationalCost = 0f;
            private float[] position = null;
            private float dropoutRate = 0f;
            
            public Builder setName(String name) {
                this.name = name;
                return this;
            }
            
            public Builder setType(LayerSnapshot.LayerType type) {
                this.type = type;
                return this;
            }
            
            public Builder setDimensions(int inputChannels, int outputChannels, int height, int width) {
                this.inputChannels = inputChannels;
                this.outputChannels = outputChannels;
                this.height = height;
                this.width = width;
                return this;
            }
            
            public Builder setKernelInfo(int kernelSize, int stride) {
                this.kernelSize = kernelSize;
                this.stride = stride;
                return this;
            }
            
            public Builder setParameters(int parameters) {
                this.parameters = parameters;
                return this;
            }
            
            public Builder setComputationalCost(float cost) {
                this.computationalCost = cost;
                return this;
            }
            
            public Builder setPosition(float x, float y, float z) {
                this.position = new float[]{x, y, z};
                return this;
            }
            
            public Builder setDropoutRate(float rate) {
                this.dropoutRate = rate;
                return this;
            }
            
            public LayerInfo build() {
                return new LayerInfo(this);
            }
        }
    }
    
    /**
     * Builder for constructing NetworkArchitecture.
     */
    public static class Builder {
        private String name = "Network";
        private List<LayerInfo> layers = new ArrayList<>();
        private int totalParameters = 0;
        private int inputSize = 0;
        private int outputSize = 0;
        private float totalComputation = 0f;
        
        public Builder setName(String name) {
            this.name = name;
            return this;
        }
        
        public Builder addLayer(LayerInfo layer) {
            layers.add(layer);
            totalParameters += layer.parameters;
            totalComputation += layer.computationalCost;
            return this;
        }
        
        public Builder setInputSize(int inputSize) {
            this.inputSize = inputSize;
            return this;
        }
        
        public Builder setOutputSize(int outputSize) {
            this.outputSize = outputSize;
            return this;
        }
        
        public Builder setTotalParameters(int totalParameters) {
            this.totalParameters = totalParameters;
            return this;
        }
        
        public Builder setTotalComputation(float totalComputation) {
            this.totalComputation = totalComputation;
            return this;
        }
        
        public NetworkArchitecture build() {
            return new NetworkArchitecture(this);
        }
    }
}
