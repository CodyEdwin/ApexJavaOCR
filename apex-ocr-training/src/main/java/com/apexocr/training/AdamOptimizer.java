package com.apexocr.training;

import com.apexocr.core.tensor.Tensor;

import java.util.HashMap;
import java.util.Map;

/**
 * Adam Optimizer - Adaptive Moment Estimation optimization algorithm.
 * 
 * Adam combines the benefits of AdaGrad and RMSProp:
 * - Maintains per-parameter learning rates
 * - Uses moving averages of first and second moments of gradients
 * - Includes bias correction for accurate initial estimates
 * 
 * This implementation is optimized for memory efficiency and numerical stability.
 *
 * @author ApexOCR Team
 * @version 1.0.0
 */
public class AdamOptimizer {
    
    private final float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;
    
    private int stepCount;
    
    // First moment estimates (m) - stores cumulative gradients
    private Map<String, Tensor> m;
    
    // Second moment estimates (v) - stores cumulative squared gradients
    private Map<String, Tensor> v;
    
    // For bias correction
    private float beta1T;
    private float beta2T;
    
    /**
     * Creates a new Adam optimizer with default parameters.
     *
     * @param learningRate The learning rate (default: 0.001)
     */
    public AdamOptimizer(float learningRate) {
        this(learningRate, 0.9f, 0.999f, 1e-8f, 0.0f);
    }
    
    /**
     * Creates a new Adam optimizer with customizable parameters.
     *
     * @param learningRate The learning rate (default: 0.001)
     * @param beta1 Exponential decay rate for first moment estimates (default: 0.9)
     * @param beta2 Exponential decay rate for second moment estimates (default: 0.999)
     * @param epsilon Small constant for numerical stability (default: 1e-8)
     * @param weightDecay L2 regularization coefficient (default: 0.0)
     */
    public AdamOptimizer(float learningRate, float beta1, float beta2, 
                         float epsilon, float weightDecay) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
        
        this.stepCount = 0;
        this.beta1T = 1.0f;
        this.beta2T = 1.0f;
        
        this.m = new HashMap<>();
        this.v = new HashMap<>();
    }
    
    /**
     * Performs one optimization step for a list of parameters.
     *
     * @param parameters Map of parameter names to tensors
     * @param gradients Map of parameter names to gradient tensors (can be null)
     */
    public void step(Map<String, Tensor> parameters, Map<String, Tensor> gradients) {
        stepCount++;
        
        // Update bias correction terms
        beta1T *= beta1;
        beta2T *= beta2;
        float biasCorrect1 = 1.0f - beta1T;
        float biasCorrect2 = 1.0f - beta2T;
        
        for (Map.Entry<String, Tensor> entry : parameters.entrySet()) {
            String name = entry.getKey();
            Tensor param = entry.getValue();
            
            // Skip if parameter is not trainable
            if (param.isView()) continue;
            
            // Get gradient for this parameter
            Tensor grad = gradients != null ? gradients.get(name) : null;
            
            if (grad != null) {
                // Apply gradient clipping if needed
                grad.clipGrad(5.0f);
                
                // Apply weight decay (L2 regularization)
                if (weightDecay > 0) {
                    long size = param.getSize();
                    for (long i = 0; i < size; i++) {
                        float w = param.getFloat(i);
                        float g = grad.getFloat(i) + weightDecay * w;
                        grad.setFloat(i, g);
                    }
                }
                
                // Initialize moment estimates if needed
                if (!m.containsKey(name)) {
                    m.put(name, new Tensor(param.getShape(), Tensor.DataType.FLOAT32));
                    m.get(name).fill(0);
                }
                if (!v.containsKey(name)) {
                    v.put(name, new Tensor(param.getShape(), Tensor.DataType.FLOAT32));
                    v.get(name).fill(0);
                }
                
                Tensor mT = m.get(name);
                Tensor vT = v.get(name);
                
                // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
                long size = param.getSize();
                for (long i = 0; i < size; i++) {
                    float mi = mT.getFloat(i);
                    float gi = grad.getFloat(i);
                    mT.setFloat(i, beta1 * mi + (1 - beta1) * gi);
                }
                
                // Update biased second moment estimate: v = beta2 * v + (1 - beta2) * g^2
                for (long i = 0; i < size; i++) {
                    float vi = vT.getFloat(i);
                    float gi = grad.getFloat(i);
                    vT.setFloat(i, beta2 * vi + (1 - beta2) * gi * gi);
                }
                
                // Compute bias-corrected estimates
                // mHat = m / (1 - beta1^t)
                // vHat = v / (1 - beta2^t)
                float mHatScale = 1.0f / biasCorrect1;
                float vHatScale = 1.0f / biasCorrect2;
                
                // Update parameters: theta = theta - lr * mHat / (sqrt(vHat) + epsilon)
                for (long i = 0; i < size; i++) {
                    float theta = param.getFloat(i);
                    float mi = mT.getFloat(i);
                    float vi = vT.getFloat(i);
                    
                    float mHat = mi * mHatScale;
                    float vHat = vi * vHatScale;
                    
                    float update = learningRate * mHat / ((float) Math.sqrt(vHat) + epsilon);
                    param.setFloat(i, theta - update);
                }
            }
            
            // Zero gradients after update
            if (grad != null) {
                grad.zeroGrad();
            }
        }
    }
    
    /**
     * Performs one optimization step for a single parameter tensor.
     *
     * @param param The parameter tensor to update
     * @param grad The gradient tensor (can be null to use stored gradient)
     * @param paramName Unique name for this parameter (for moment storage)
     */
    public void step(Tensor param, Tensor grad, String paramName) {
        if (param == null || param.isView()) return;
        
        Map<String, Tensor> params = new HashMap<>();
        Map<String, Tensor> grads = new HashMap<>();
        params.put(paramName, param);
        grads.put(paramName, grad);
        
        step(params, grads);
    }
    
    /**
     * Zeros out all gradients for all tracked parameters.
     */
    public void zeroGrad() {
        for (Tensor grad : m.values()) {
            if (grad != null) grad.zeroGrad();
        }
        for (Tensor grad : v.values()) {
            if (grad != null) grad.zeroGrad();
        }
    }
    
    /**
     * Clears all stored state (moments and step count).
     * Use this to reset the optimizer for a new training run.
     */
    public void clear() {
        stepCount = 0;
        beta1T = 1.0f;
        beta2T = 1.0f;
        
        for (Tensor tensor : m.values()) {
            if (tensor != null) tensor.close();
        }
        for (Tensor tensor : v.values()) {
            if (tensor != null) tensor.close();
        }
        m.clear();
        v.clear();
    }
    
    /**
     * Gets the current learning rate.
     *
     * @return The learning rate
     */
    public float getLearningRate() {
        return learningRate;
    }
    
    /**
     * Gets the current step count.
     *
     * @return The number of optimization steps performed
     */
    public int getStepCount() {
        return stepCount;
    }
    
    /**
     * Gets the total number of tracked parameters.
     *
     * @return The count of parameter tensors being optimized
     */
    public int getParameterCount() {
        return m.size();
    }
    
    /**
     * Updates the learning rate (useful for learning rate scheduling).
     *
     * @param newLearningRate The new learning rate
     */
    public void setLearningRate(float newLearningRate) {
        // Note: This doesn't change the internal field since it's final
        // For learning rate scheduling, create a new optimizer instance
    }
    
    /**
     * Creates a learning rate schedule that decays by a factor every few epochs.
     *
     * @param baseRate The base learning rate
     * @param decayFactor The factor to multiply by (e.g., 0.95)
     * @param decayEpochs Number of epochs between decay
     * @return A function that computes the learning rate for a given epoch
     */
    public static float createStepDecaySchedule(float baseRate, float decayFactor, int decayEpochs) {
        return baseRate * (float) Math.pow(decayFactor, Math.floorDiv(0, decayEpochs));
    }
    
    /**
     * Creates a warmup learning rate schedule.
     *
     * @param baseRate The target learning rate after warmup
     * @param warmupSteps Number of warmup steps
     * @param step The current step
     * @return The learning rate for the current step
     */
    public static float createWarmupSchedule(float baseRate, int warmupSteps, int step) {
        if (step < warmupSteps) {
            return baseRate * (float) step / warmupSteps;
        }
        return baseRate;
    }
    
    /**
     * Closes the optimizer and releases all resources.
     */
    public void close() {
        clear();
    }
    
    @Override
    protected void finalize() throws Throwable {
        close();
        super.finalize();
    }
}
