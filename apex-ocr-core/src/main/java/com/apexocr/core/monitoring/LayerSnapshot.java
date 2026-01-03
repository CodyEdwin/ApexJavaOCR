package com.apexocr.core.monitoring;

/**
 * Snapshot of layer-specific data for visualization.
 * Contains activation, gradient, and weight statistics for a single layer.
 */
public class LayerSnapshot {
    
    public final String layerName;
    public final LayerType layerType;
    
    // Activation statistics
    public final float activationMean;
    public final float activationStd;
    public final float activationMin;
    public final float activationMax;
    
    // Gradient statistics
    public final float gradientMean;
    public final float gradientStd;
    public final float gradientMin;
    public final float gradientMax;
    public final float gradientL2Norm;
    
    // Weight statistics
    public final float weightMean;
    public final float weightStd;
    public final float weightMin;
    public final float weightMax;
    
    // Layer metadata
    public final int inputChannels;
    public final int outputChannels;
    public final int height;
    public final int width;
    public final int parameters;
    
    // Visualization data
    public final float activationIntensity;
    public final float gradientFlow;
    public final float weightHealth;
    
    public LayerSnapshot(Builder builder) {
        this.layerName = builder.layerName;
        this.layerType = builder.layerType;
        this.activationMean = builder.activationMean;
        this.activationStd = builder.activationStd;
        this.activationMin = builder.activationMin;
        this.activationMax = builder.activationMax;
        this.gradientMean = builder.gradientMean;
        this.gradientStd = builder.gradientStd;
        this.gradientMin = builder.gradientMin;
        this.gradientMax = builder.gradientMax;
        this.gradientL2Norm = builder.gradientL2Norm;
        this.weightMean = builder.weightMean;
        this.weightStd = builder.weightStd;
        this.weightMin = builder.weightMin;
        this.weightMax = builder.weightMax;
        this.inputChannels = builder.inputChannels;
        this.outputChannels = builder.outputChannels;
        this.height = builder.height;
        this.width = builder.width;
        this.parameters = builder.parameters;
        
        // Calculate visualization metrics
        this.activationIntensity = calculateActivationIntensity();
        this.gradientFlow = calculateGradientFlow();
        this.weightHealth = calculateWeightHealth();
    }
    
    /**
     * Create a builder for constructing layer snapshots.
     * @return New builder instance
     */
    public static Builder builder() {
        return new Builder();
    }
    
    private float calculateActivationIntensity() {
        // Normalized intensity for visualization (0-1)
        float range = activationMax - Math.abs(activationMin);
        if (range == 0) return 0f;
        return Math.min(1f, Math.abs(activationMean) / (range / 2f));
    }
    
    private float calculateGradientFlow() {
        // Gradient flow intensity (indicates how much gradient is passing through)
        if (gradientL2Norm < 1e-7f) return 0f; // Vanishing gradient
        if (gradientL2Norm > 1e3f) return 1f; // Exploding gradient
        return Math.min(1f, (float) Math.log10(gradientL2Norm + 1) / 3f);
    }
    
    private float calculateWeightHealth() {
        // Weight health based on distribution (good weights have reasonable std around 0)
        if (weightStd < 0.01f) return 0.2f; // Underfitting risk
        if (weightStd > 1.0f) return 0.8f; // Overfitting risk
        return 0.5f + (1f - Math.abs(weightStd - 0.5f)); // Healthy range around 0.1-0.5
    }
    
    /**
     * Layer type enumeration.
     */
    public enum LayerType {
        INPUT,
        CONV2D,
        BILSTM,
        DENSE,
        ACTIVATION,
        DROPOUT,
        POOLING,
        OUTPUT
    }
    
    /**
     * Builder for constructing LayerSnapshot instances.
     */
    public static class Builder {
        private String layerName = "Unnamed";
        private LayerType layerType = LayerType.DENSE;
        
        private float activationMean = 0f;
        private float activationStd = 0f;
        private float activationMin = 0f;
        private float activationMax = 0f;
        
        private float gradientMean = 0f;
        private float gradientStd = 0f;
        private float gradientMin = 0f;
        private float gradientMax = 0f;
        private float gradientL2Norm = 0f;
        
        private float weightMean = 0f;
        private float weightStd = 0f;
        private float weightMin = 0f;
        private float weightMax = 0f;
        
        private int inputChannels = 0;
        private int outputChannels = 0;
        private int height = 0;
        private int width = 0;
        private int parameters = 0;
        
        public Builder setLayerName(String layerName) {
            this.layerName = layerName;
            return this;
        }
        
        public Builder setLayerType(LayerType layerType) {
            this.layerType = layerType;
            return this;
        }
        
        public Builder setActivationStats(float mean, float std, float min, float max) {
            this.activationMean = mean;
            this.activationStd = std;
            this.activationMin = min;
            this.activationMax = max;
            return this;
        }
        
        public Builder setGradientStats(float mean, float std, float min, float max, float l2Norm) {
            this.gradientMean = mean;
            this.gradientStd = std;
            this.gradientMin = min;
            this.gradientMax = max;
            this.gradientL2Norm = l2Norm;
            return this;
        }
        
        public Builder setWeightStats(float mean, float std, float min, float max) {
            this.weightMean = mean;
            this.weightStd = std;
            this.weightMin = min;
            this.weightMax = max;
            return this;
        }
        
        public Builder setDimensions(int inputChannels, int outputChannels, int height, int width) {
            this.inputChannels = inputChannels;
            this.outputChannels = outputChannels;
            this.height = height;
            this.width = width;
            return this;
        }
        
        public Builder setParameters(int parameters) {
            this.parameters = parameters;
            return this;
        }
        
        public LayerSnapshot build() {
            return new LayerSnapshot(this);
        }
    }
}
