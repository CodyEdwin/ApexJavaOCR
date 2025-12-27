package com.apexocr.core.ctc;

import com.apexocr.core.tensor.Tensor;

import java.util.*;

/**
 * CTCDecoder - Connectionist Temporal Classification decoder for OCR sequence recognition.
 * Implements beam search decoding to convert neural network output probabilities
 * into the most likely character sequence.
 *
 * CTC is essential for OCR because it handles variable-length output sequences
 * without requiring explicit alignment between input and output sequences.
 *
 * This implementation provides high-performance beam search decoding optimized
 * for the JVM, enabling real-time OCR inference.
 *
 * @author ApexOCR Team
 * @version 1.0.0
 */
public class CTCDecoder {
    private final int beamWidth;
    private final float[] blankProbability;
    private final Map<Integer, String> labelToChar;
    private final Map<String, Integer> charToLabel;
    private final boolean mergeRepeated;
    private final float cutoffProbability;

    /**
     * Creates a new CTC decoder with the specified configuration.
     *
     * @param beamWidth Number of beams to maintain during search
     * @param labels The set of character labels (including blank token)
     * @param mergeRepeated Whether to merge repeated characters in output
     * @param cutoffProbability Minimum probability threshold for pruning
     */
    public CTCDecoder(int beamWidth, String[] labels, boolean mergeRepeated, float cutoffProbability) {
        this.beamWidth = beamWidth;
        this.mergeRepeated = mergeRepeated;
        this.cutoffProbability = cutoffProbability;

        this.labelToChar = new HashMap<>();
        this.charToLabel = new HashMap<>();

        for (int i = 0; i < labels.length; i++) {
            String label = labels[i];
            labelToChar.put(i, label);
            charToLabel.put(label, i);
        }

        // Pre-compute blank probability array for quick lookup
        this.blankProbability = new float[labels.length];
        // Assume blank is the last label or explicitly marked
        // For this implementation, we assume index 0 is the blank token
        this.blankProbability[0] = 1.0f;
    }

    /**
     * Creates a CTC decoder with default settings.
     *
     * @param labels The set of character labels
     * @return A new decoder instance
     */
    public static CTCDecoder createDefault(String[] labels) {
        return new CTCDecoder(10, labels, true, 1e-3f);
    }

    /**
     * Decodes the network output using beam search.
     *
     * @param input Network output tensor of shape [timeSteps, numClasses]
     *            containing log probabilities for each character at each time step
     * @return Decoded text string
     */
    public String decode(Tensor input) {
        long[] shape = input.getShape();
        int timeSteps = (int) shape[0];
        int numClasses = (int) shape[1];

        // Convert to log probabilities for numerical stability
        float[][] logProbs = new float[timeSteps][numClasses];
        for (int t = 0; t < timeSteps; t++) {
            for (int c = 0; c < numClasses; c++) {
                float prob = input.getFloat(t, c);
                // Convert to log space with clipping
                logProbs[t][c] = (float) Math.log(Math.max(prob, 1e-10f));
            }
        }

        // Perform beam search
        BeamSearchResult result = beamSearch(logProbs, timeSteps, numClasses);

        // Convert label sequence to string
        return labelsToString(result.labels);
    }

    /**
     * Performs CTC beam search decoding.
     */
    private BeamSearchResult beamSearch(float[][] logProbs, int timeSteps, int numClasses) {
        // Initialize beam with blank sequence
        Map<String, BeamState> beams = new HashMap<>();
        BeamState initialState = new BeamState();
        initialState.logProb = 0f;
        initialState.labels = new ArrayList<>();
        initialState.lastLabel = 0; // Blank
        beams.put("", initialState);

        // Process each time step
        for (int t = 0; t < timeSteps; t++) {
            Map<String, BeamState> newBeams = new HashMap<>();

            for (Map.Entry<String, BeamState> entry : beams.entrySet()) {
                BeamState state = entry.getValue();
                String prefix = entry.getKey();

                for (int c = 0; c < numClasses; c++) {
                    float logProb = logProbs[t][c];

                    // Skip low probability beams
                    if (logProb < Math.log(cutoffProbability)) {
                        continue;
                    }

                    // Blank token: extend existing beam
                    if (c == 0) {
                        BeamState newState = extendBeam(state, prefix, -1);
                        mergeBeam(newBeams, prefix, newState);
                    }
                    // Non-blank token: can extend or create new beam
                    else {
                        // Check if this extends the previous non-blank character
                        if (state.lastLabel == c && (mergeRepeated || prefix.isEmpty())) {
                            // Same as last non-blank: extend existing beam
                            BeamState newState = extendBeam(state, prefix, c);
                            mergeBeam(newBeams, prefix, newState);
                        } else {
                            // Different character: create new beam
                            String newPrefix = prefix + labelToChar.get(c);
                            BeamState newState = extendBeam(state, newPrefix, c);
                            mergeBeam(newBeams, newPrefix, newState);
                        }
                    }
                }
            }

            // Keep only top beams
            beams = pruneBeams(newBeams, beamWidth);
        }

        // Return best beam
        return getBestBeam(beams);
    }

    /**
     * Extends a beam with a new character.
     */
    private BeamState extendBeam(BeamState state, String prefix, int newLabel) {
        BeamState newState = new BeamState();
        newState.logProb = state.logProb;
        newState.labels = new ArrayList<>(state.labels);
        newState.lastLabel = newLabel;
        return newState;
    }

    /**
     * Merges a beam into the beam map, keeping the best probability for duplicate prefixes.
     */
    private void mergeBeam(Map<String, BeamState> beams, String prefix, BeamState state) {
        BeamState existing = beams.get(prefix);
        if (existing == null || state.logProb > existing.logProb) {
            beams.put(prefix, state);
        }
    }

    /**
     * Prunes beams to keep only the top k by probability.
     */
    private Map<String, BeamState> pruneBeams(Map<String, BeamState> beams, int maxBeams) {
        if (beams.size() <= maxBeams) {
            return beams;
        }

        // Sort beams by probability
        List<Map.Entry<String, BeamState>> sorted = new ArrayList<>(beams.entrySet());
        sorted.sort((a, b) -> Float.compare(b.getValue().logProb, a.getValue().logProb));

        // Keep top k
        Map<String, BeamState> pruned = new LinkedHashMap<>();
        for (int i = 0; i < Math.min(maxBeams, sorted.size()); i++) {
            pruned.put(sorted.get(i).getKey(), sorted.get(i).getValue());
        }

        return pruned;
    }

    /**
     * Gets the best beam from the final set.
     */
    private BeamSearchResult getBestBeam(Map<String, BeamState> beams) {
        BeamSearchResult result = new BeamSearchResult();

        Map.Entry<String, BeamState> best = null;
        float bestProb = Float.NEGATIVE_INFINITY;

        for (Map.Entry<String, BeamState> entry : beams.entrySet()) {
            if (entry.getValue().logProb > bestProb) {
                bestProb = entry.getValue().logProb;
                best = entry;
            }
        }

        if (best != null) {
            result.text = best.getKey();
            result.labels = best.getValue().labels;
            result.logProb = best.getValue().logProb;
        }

        return result;
    }

    /**
     * Converts a list of labels to a string.
     */
    private String labelsToString(List<Integer> labels) {
        StringBuilder sb = new StringBuilder();
        Integer prevLabel = null;

        for (Integer label : labels) {
            if (prevLabel != null && label.equals(prevLabel) && mergeRepeated) {
                continue; // Skip repeated characters
            }
            if (label == 0) {
                prevLabel = label;
                continue; // Skip blank tokens
            }
            String ch = labelToChar.get(label);
            if (ch != null) {
                sb.append(ch);
            }
            prevLabel = label;
        }

        return sb.toString();
    }

    /**
     * Performs greedy decoding (faster but less accurate than beam search).
     *
     * @param input Network output tensor
     * @return Decoded text string
     */
    public String decodeGreedy(Tensor input) {
        long[] shape = input.getShape();
        int timeSteps = (int) shape[0];
        int numClasses = (int) shape[1];

        StringBuilder result = new StringBuilder();
        Integer lastLabel = null;

        for (int t = 0; t < timeSteps; t++) {
            // Find the class with highest probability
            int bestLabel = 0;
            float bestProb = Float.NEGATIVE_INFINITY;

            for (int c = 0; c < numClasses; c++) {
                if (input.getFloat(t, c) > bestProb) {
                    bestProb = input.getFloat(t, c);
                    bestLabel = c;
                }
            }

            // Skip blanks and repeated characters
            if (bestLabel == 0) {
                lastLabel = null;
            } else if (lastLabel == null || bestLabel != lastLabel || !mergeRepeated) {
                String ch = labelToChar.get(bestLabel);
                if (ch != null) {
                    result.append(ch);
                }
                lastLabel = bestLabel;
            }
        }

        return result.toString();
    }

    /**
     * Computes the log probability of a given label sequence.
     *
     * @param input Network output tensor
     * @param labels The label sequence
     * @return Log probability of the sequence
     */
    public float getSequenceLogProb(Tensor input, List<Integer> labels) {
        long[] shape = input.getShape();
        int timeSteps = (int) shape[0];

        float logProb = 0f;
        int labelIdx = 0;
        int prevLabel = -1;

        for (int t = 0; t < timeSteps && labelIdx < labels.size(); t++) {
            int currentLabel = labels.get(labelIdx);

            // For CTC, blank can repeat
            if (currentLabel == 0) {
                logProb += input.getFloat(t, 0);
                prevLabel = 0;
            } else {
                // Only move to next label when we see a non-blank different from prev
                if (currentLabel != prevLabel) {
                    logProb += input.getFloat(t, currentLabel);
                    labelIdx++;
                    prevLabel = currentLabel;
                }
            }
        }

        return logProb;
    }

    /**
     * Gets the vocabulary size.
     *
     * @return Number of classes including blank
     */
    public int getVocabularySize() {
        return labelToChar.size();
    }

    /**
     * Gets the blank token index.
     *
     * @return The blank token index (assumed to be 0)
     */
    public int getBlankIndex() {
        return 0;
    }

    /**
     * Beam state container.
     */
    private static class BeamState {
        float logProb;
        List<Integer> labels;
        int lastLabel;
    }

    /**
     * Beam search result container.
     */
    private static class BeamSearchResult {
        String text;
        List<Integer> labels;
        float logProb;
    }

    /**
     * Decodes multiple inputs and returns results with confidence scores.
     *
     * @param inputs Array of network output tensors
     * @return Array of decode results
     */
    public DecodeResult[] decodeBatch(Tensor[] inputs) {
        DecodeResult[] results = new DecodeResult[inputs.length];

        for (int i = 0; i < inputs.length; i++) {
            String text = decode(inputs[i]);
            float confidence = computeConfidence(inputs[i], text);
            results[i] = new DecodeResult(text, confidence);
        }

        return results;
    }

    /**
     * Computes confidence score for a decoded result.
     *
     * @param input Network output
     * @param text Decoded text
     * @return Confidence score between 0 and 1
     */
    private float computeConfidence(Tensor input, String text) {
        long[] shape = input.getShape();
        int timeSteps = (int) shape[0];
        int numClasses = (int) shape[1];

        float totalProb = 0f;

        for (int t = 0; t < timeSteps; t++) {
            float maxProb = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < numClasses; c++) {
                maxProb = Math.max(maxProb, input.getFloat(t, c));
            }
            totalProb += maxProb;
        }

        return totalProb / timeSteps;
    }

    /**
     * Result container for batch decoding.
     */
    public static class DecodeResult {
        public final String text;
        public final float confidence;

        public DecodeResult(String text, float confidence) {
            this.text = text;
            this.confidence = confidence;
        }
    }
}
