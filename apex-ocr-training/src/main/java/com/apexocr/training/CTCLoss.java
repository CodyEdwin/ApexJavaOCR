package com.apexocr.training;

import com.apexocr.core.tensor.Tensor;
import com.apexocr.core.tensor.TensorOperations;

import java.util.ArrayList;
import java.util.List;

/**
 * CTCLoss - Connectionist Temporal Classification Loss implementation.
 * 
 * CTC is designed for sequence labeling tasks where the alignment between
 * input and output is unknown. It allows the model to output sequences
 * that are longer than the target by using a blank symbol (typically index 0).
 * 
 * This implementation uses the log-space forward-backward algorithm for
 * numerical stability and provides gradients with respect to the logits.
 *
 * Key features:
 * - Log-space computations to prevent underflow
 * - Efficient forward-backward algorithm
 * - Proper gradient computation for backpropagation
 * - Support for variable length sequences
 *
 * @author ApexOCR Team
 * @version 1.0.0
 */
public class CTCLoss {
    
    private final int blankIndex;
    private final float reduction;
    
    /**
     * Reduction methods for aggregating loss values.
     */
    public enum Reduction {
        NONE,    // Return per-sample losses
        MEAN,    // Mean over all samples
        SUM      // Sum over all samples
    }
    
    /**
     * Creates a new CTC Loss calculator.
     *
     * @param blankIndex The index of the blank symbol in the vocabulary (typically 0)
     */
    public CTCLoss(int blankIndex) {
        this.blankIndex = blankIndex;
        this.reduction = 1.0f;
    }
    
    /**
     * Creates a new CTC Loss calculator with mean reduction.
     *
     * @param blankIndex The index of the blank symbol in the vocabulary
     * @param reduction The reduction method to apply
     */
    public CTCLoss(int blankIndex, Reduction reduction) {
        this.blankIndex = blankIndex;
        this.reduction = reduction == Reduction.MEAN ? 1.0f : 
                         reduction == Reduction.SUM ? 1.0f : 0.0f;
    }
    
    /**
     * Computes the CTC loss and returns both the scalar loss and gradients.
     * 
     * This method implements the forward-backward algorithm in log-space:
     * 1. Forward pass: computes alpha_t(k) - probability of being at position t
     *    with label k ending at current position
     * 2. Backward pass: computes beta_t(k) - probability of completing the
     *    sequence from position t with label k
     * 3. Combines alpha and beta to get gradient contributions
     *
     * @param logits Input tensor of shape [batch, timeSteps, numClasses]
     *               Contains unnormalized scores (logits)
     * @param targets Target labels as int arrays (variable length)
     * @param inputLengths Array of actual sequence lengths for each batch element
     * @param targetLengths Array of target lengths for each batch element
     * @return A result object containing loss value and gradients
     */
    public CTCLossResult compute(Tensor logits, int[][] targets, 
                                  int[] inputLengths, int[] targetLengths) {
        long[] shape = logits.getShape();
        int batchSize = (int) shape[0];
        int timeSteps = (int) shape[1];
        int numClasses = (int) shape[2];
        
        // Apply softmax to get probabilities
        Tensor probs = TensorOperations.softmax(logits);
        
        // Convert to log probabilities for numerical stability
        Tensor logProbs = logSoftmax(probs);
        
        float totalLoss = 0;
        
        // Output gradient tensor
        Tensor gradOutput = new Tensor(shape, Tensor.DataType.FLOAT32);
        
        for (int b = 0; b < batchSize; b++) {
            int T = inputLengths[b];
            int L = targetLengths[b];
            int[] target = targets[b];
            
            // Extended target with blanks at both ends: _t1_t2_..._tL_
            int[] extendedTarget = new int[L + 2];
            extendedTarget[0] = blankIndex;
            System.arraycopy(target, 0, extendedTarget, 1, L);
            extendedTarget[L + 1] = blankIndex;
            
            int S = L + 2;  // Extended sequence length
            
            // Forward pass (alpha) in log-space
            float[][] alpha = new float[T + 1][S];
            for (int t = 0; t <= T; t++) {
                for (int s = 0; s < S; s++) {
                    alpha[t][s] = Float.NEGATIVE_INFINITY;
                }
            }
            alpha[0][0] = 0.0f;  // Start state
            
            for (int t = 1; t <= T; t++) {
                for (int s = 0; s < S; s++) {
                    float logProb = logProbs.getFloat(b, t - 1, extendedTarget[s]);
                    
                    // Transition from previous states
                    float maxVal = alpha[t - 1][s];  // Stay in same state (including blank)
                    if (s > 0) {
                        maxVal = Math.max(maxVal, alpha[t - 1][s - 1]);  // From previous character
                    }
                    
                    if (maxVal == Float.NEGATIVE_INFINITY) {
                        alpha[t][s] = Float.NEGATIVE_INFINITY;
                    } else {
                        // log-sum-exp trick for numerical stability
                        float sum = 0;
                        boolean hasValid = false;
                        
                        // From same state (stay)
                        if (alpha[t - 1][s] != Float.NEGATIVE_INFINITY) {
                            sum += (float) Math.exp(alpha[t - 1][s] - maxVal);
                            hasValid = true;
                        }
                        
                        // From previous character (skip blank or transition)
                        if (s > 0 && alpha[t - 1][s - 1] != Float.NEGATIVE_INFINITY) {
                            // Can't stay in same character (must emit something)
                            if (extendedTarget[s] != extendedTarget[s - 1] || extendedTarget[s] == blankIndex) {
                                sum += (float) Math.exp(alpha[t - 1][s - 1] - maxVal);
                                hasValid = true;
                            }
                        }
                        
                        if (hasValid) {
                            alpha[t][s] = maxVal + (float) Math.log(sum);
                        } else {
                            alpha[t][s] = Float.NEGATIVE_INFINITY;
                        }
                    }
                }
            }
            
            // Backward pass (beta) in log-space
            float[][] beta = new float[T + 1][S];
            for (int t = 0; t <= T; t++) {
                for (int s = 0; s < S; s++) {
                    beta[t][s] = Float.NEGATIVE_INFINITY;
                }
            }
            beta[T][S - 1] = 0.0f;  // End state
            
            for (int t = T - 1; t >= 0; t--) {
                for (int s = S - 1; s >= 0; s--) {
                    float maxVal = beta[t + 1][s];  // Stay in same state
                    
                    // From next character
                    if (s < S - 1) {
                        maxVal = Math.max(maxVal, beta[t + 1][s + 1]);
                    }
                    
                    if (maxVal == Float.NEGATIVE_INFINITY) {
                        beta[t][s] = Float.NEGATIVE_INFINITY;
                    } else {
                        float sum = 0;
                        boolean hasValid = false;
                        
                        // From same state (stay)
                        if (beta[t + 1][s] != Float.NEGATIVE_INFINITY) {
                            sum += (float) Math.exp(beta[t + 1][s] - maxVal);
                            hasValid = true;
                        }
                        
                        // From next character (transition)
                        if (s < S - 1) {
                            // Can't stay in same character
                            if (extendedTarget[s] != extendedTarget[s + 1] || extendedTarget[s + 1] == blankIndex) {
                                if (beta[t + 1][s + 1] != Float.NEGATIVE_INFINITY) {
                                    sum += (float) Math.exp(beta[t + 1][s + 1] - maxVal);
                                    hasValid = true;
                                }
                            }
                        }
                        
                        if (hasValid) {
                            beta[t][s] = maxVal + (float) Math.log(sum);
                        } else {
                            beta[t][s] = Float.NEGATIVE_INFINITY;
                        }
                    }
                }
            }
            
            // Compute log probability of the path
            float logProbPath = alpha[T][S - 1];
            
            // For numerical stability
            if (logProbPath == Float.NEGATIVE_INFINITY) {
                logProbPath = -1000f;  // Very small probability
            }
            
            // Accumulate loss
            totalLoss -= logProbPath;
            
            // Compute gradients
            // The gradient for each (t, k) is: P(t, k) - expected probability
            // where P(t, k) = alpha[t, state_k] * beta[t, state_k] / P(path)
            
            for (int t = 0; t < T; t++) {
                for (int k = 0; k < numClasses; k++) {
                    float grad = 0;
                    
                    // Sum over all positions where this class appears in extended target
                    for (int s = 0; s < S; s++) {
                        if (extendedTarget[s] == k) {
                            if (alpha[t + 1][s] != Float.NEGATIVE_INFINITY && 
                                beta[t][s] != Float.NEGATIVE_INFINITY) {
                                // log(alpha * beta) = log(alpha) + log(beta)
                                float logJoint = alpha[t + 1][s] + beta[t][s];
                                float probContribution = (float) Math.exp(logJoint - logProbPath);
                                
                                // Probability of emitting this class at this time
                                float emitProb = (float) Math.exp(logProbs.getFloat(b, t, k));
                                
                                grad += probContribution - emitProb;
                            }
                        }
                    }
                    
                    // Store gradient (negative because we minimized loss)
                    float currentGrad = gradOutput.getFloat(b, t, k);
                    gradOutput.setFloat(currentGrad - grad, b, t, k);
                }
            }
            
            // Apply probability gradient for softmax
            // dL/dlogit = probability - target_distribution
            for (int t = 0; t < T; t++) {
                for (int k = 0; k < numClasses; k++) {
                    float prob = probs.getFloat(b, t, k);
                    float grad = gradOutput.getFloat(b, t, k);
                    gradOutput.setFloat(grad * prob, b, t, k);
                }
            }
        }
        
        // Apply reduction
        float finalLoss;
        if (reduction != 0) {
            finalLoss = totalLoss / batchSize;
        } else {
            finalLoss = totalLoss;
        }
        
        probs.close();
        
        return new CTCLossResult(finalLoss, gradOutput);
    }
    
    /**
     * Simple log softmax for numerical stability.
     */
    private Tensor logSoftmax(Tensor input) {
        long[] shape = input.getShape();
        int batchSize = (int) shape[0];
        int timeSteps = (int) shape[1];
        int numClasses = (int) shape[2];
        
        Tensor result = new Tensor(shape, Tensor.DataType.FLOAT32);
        
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < timeSteps; t++) {
                // Find max for numerical stability
                float maxVal = Float.NEGATIVE_INFINITY;
                for (int k = 0; k < numClasses; k++) {
                    maxVal = Math.max(maxVal, input.getFloat(b, t, k));
                }
                
                // Compute log-sum-exp
                float sum = 0;
                for (int k = 0; k < numClasses; k++) {
                    sum += (float) Math.exp(input.getFloat(b, t, k) - maxVal);
                }
                float logSum = maxVal + (float) Math.log(sum);
                
                // Compute log softmax
                for (int k = 0; k < numClasses; k++) {
                    result.setFloat(input.getFloat(b, t, k) - logSum, b, t, k);
                }
            }
        }
        
        return result;
    }
    
    /**
     * Result container for CTC loss computation.
     */
    public static class CTCLossResult {
        private final float loss;
        private final Tensor gradients;
        
        public CTCLossResult(float loss, Tensor gradients) {
            this.loss = loss;
            this.gradients = gradients;
        }
        
        public float getLoss() {
            return loss;
        }
        
        public Tensor getGradients() {
            return gradients;
        }
    }
    
    /**
     * Computes a simplified CTC-like loss for quick training.
     * This is a faster approximation that works well for initial training.
     *
     * @param output Network output tensor [batch, timeSteps, numClasses]
     * @param targetLabels Target text labels
     * @param vocabulary The vocabulary string
     * @return Loss value
     */
    public static float computeSimpleLoss(Tensor output, List<String> targetLabels, String vocabulary) {
        long[] shape = output.getShape();
        int batchSize = (int) shape[0];
        int timeSteps = (int) shape[1];
        int numClasses = (int) shape[2];
        
        float totalLoss = 0;
        
        for (int b = 0; b < batchSize; b++) {
            String target = targetLabels.get(b);
            int[] targetIndices = new int[target.length()];
            for (int i = 0; i < target.length(); i++) {
                targetIndices[i] = vocabulary.indexOf(target.charAt(i)) + 1;  // +1 for blank
            }
            
            // Simple alignment-based loss
            int targetPos = 0;
            for (int t = 0; t < timeSteps && targetPos < targetIndices.length; t++) {
                int targetIdx = targetIndices[targetPos];
                float prob = output.getFloat(b, t, targetIdx);
                prob = Math.max(prob, 1e-7f);  // Avoid log(0)
                totalLoss += (float) -Math.log(prob);
                
                // Move to next character if probability is high enough
                if (prob > 0.5f) {
                    targetPos++;
                }
            }
            
            // Penalize remaining steps
            for (int t = targetPos; t < timeSteps; t++) {
                float blankProb = output.getFloat(b, t, 0);
                blankProb = Math.max(blankProb, 1e-7f);
                totalLoss += (float) -Math.log(blankProb);
            }
        }
        
        return totalLoss / batchSize;
    }
    
    /**
     * Greedy decoding for CTC output.
     *
     * @param output Network output tensor [batch, timeSteps, numClasses]
     * @param blankIndex Index of the blank symbol
     * @return List of decoded strings
     */
    public static List<String> decode(Tensor output, int blankIndex) {
        long[] shape = output.getShape();
        int batchSize = (int) shape[0];
        int timeSteps = (int) shape[1];
        int numClasses = (int) shape[2];
        
        List<String> results = new ArrayList<>();
        
        for (int b = 0; b < batchSize; b++) {
            StringBuilder sb = new StringBuilder();
            int prevClass = -1;
            
            for (int t = 0; t < timeSteps; t++) {
                int bestClass = 0;
                float bestProb = output.getFloat(b, t, 0);
                
                for (int c = 1; c < numClasses; c++) {
                    float prob = output.getFloat(b, t, c);
                    if (prob > bestProb) {
                        bestProb = prob;
                        bestClass = c;
                    }
                }
                
                // Skip blanks and repeated characters
                if (bestClass != blankIndex && bestClass != prevClass) {
                    sb.append((char) ('A' + bestClass - 1));  // Adjust for vocabulary
                }
                prevClass = bestClass;
            }
            
            results.add(sb.toString());
        }
        
        return results;
    }
}
