package com.apexocr.engine;

import java.util.ArrayList;
import java.util.List;

/**
 * OcrResult - Container for OCR processing results.
 * Holds the recognized text, confidence score, and metadata
 * from the OCR processing pipeline.
 *
 * This class provides a comprehensive result structure that includes
 * not only the text but also quality metrics and processing information
 * useful for downstream applications.
 *
 * @author ApexOCR Team
 * @version 1.0.0
 */
public class OcrResult {
    private final String text;
    private final float confidence;
    private final long processingTimeMs;
    private final int imageWidth;
    private final int imageHeight;
    private final List<TextRegion> regions;
    private final List<Word> words;
    private final List<CharacterInfo> characterInfos;

    /**
     * Creates a new OCR result with basic information.
     *
     * @param text The recognized text
     * @param confidence Confidence score (0-1)
     * @param processingTimeMs Processing time in milliseconds
     * @param imageWidth Width of processed image
     * @param imageHeight Height of processed image
     */
    public OcrResult(String text, float confidence, long processingTimeMs,
                     int imageWidth, int imageHeight) {
        this.text = text != null ? text : "";
        this.confidence = Math.max(0, Math.min(1, confidence));
        this.processingTimeMs = processingTimeMs;
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        this.regions = new ArrayList<>();
        this.words = new ArrayList<>();
        this.characterInfos = new ArrayList<>();
    }

    /**
     * Creates a basic result with zero processing time.
     *
     * @param text The recognized text
     * @param confidence Confidence score
     */
    public OcrResult(String text, float confidence) {
        this(text, confidence, 0, 0, 0);
    }

    /**
     * Gets the recognized text.
     *
     * @return The text string
     */
    public String getText() {
        return text;
    }

    /**
     * Gets the confidence score.
     *
     * @return Confidence score between 0 and 1
     */
    public float getConfidence() {
        return confidence;
    }

    /**
     * Gets the confidence as a percentage.
     *
     * @return Confidence percentage (0-100)
     */
    public float getConfidencePercent() {
        return confidence * 100f;
    }

    /**
     * Gets the processing time.
     *
     * @return Processing time in milliseconds
     */
    public long getProcessingTimeMs() {
        return processingTimeMs;
    }

    /**
     * Gets the processing time in seconds.
     *
     * @return Processing time in seconds
     */
    public float getProcessingTimeSeconds() {
        return processingTimeMs / 1000f;
    }

    /**
     * Gets the image dimensions.
     *
     * @return Image width
     */
    public int getImageWidth() {
        return imageWidth;
    }

    /**
     * Gets the image dimensions.
     *
     * @return Image height
     */
    public int getImageHeight() {
        return imageHeight;
    }

    /**
     * Gets the list of detected text regions.
     *
     * @return List of text regions
     */
    public List<TextRegion> getRegions() {
        return new ArrayList<>(regions);
    }

    /**
     * Gets the list of detected words.
     *
     * @return List of words
     */
    public List<Word> getWords() {
        return new ArrayList<>(words);
    }

    /**
     * Gets character-level information.
     *
     * @return List of character information
     */
    public List<CharacterInfo> getCharacterInfos() {
        return new ArrayList<>(characterInfos);
    }

    /**
     * Sets the list of text regions.
     *
     * @param regions List of text regions
     */
    public void setRegions(List<TextRegion> regions) {
        this.regions.clear();
        this.regions.addAll(regions);
    }

    /**
     * Sets the list of words.
     *
     * @param words List of words
     */
    public void setWords(List<Word> words) {
        this.words.clear();
        this.words.addAll(words);
    }

    /**
     * Sets character information.
     *
     * @param characterInfos List of character information
     */
    public void setCharacterInfos(List<CharacterInfo> characterInfos) {
        this.characterInfos.clear();
        this.characterInfos.addAll(characterInfos);
    }

    /**
     * Gets the number of words in the result.
     *
     * @return Word count
     */
    public int getWordCount() {
        if (text == null || text.isEmpty()) {
            return 0;
        }
        return text.trim().split("\\s+").length;
    }

    /**
     * Gets the number of characters in the result.
     *
     * @return Character count
     */
    public int getCharacterCount() {
        return text.length();
    }

    /**
     * Checks if the result has high confidence.
     *
     * @param threshold Confidence threshold
     * @return True if confidence is above threshold
     */
    public boolean hasHighConfidence(float threshold) {
        return confidence >= threshold;
    }

    /**
     * Gets a summary of the result.
     *
     * @return Summary string
     */
    public String getSummary() {
        return String.format(
            "Text: '%s' | Confidence: %.2f%% | Words: %d | Chars: %d | Time: %dms",
            truncate(text, 50),
            getConfidencePercent(),
            getWordCount(),
            getCharacterCount(),
            processingTimeMs
        );
    }

    /**
     * Truncates a string to the specified length.
     */
    private String truncate(String str, int maxLen) {
        if (str == null) return "";
        if (str.length() <= maxLen) return str;
        return str.substring(0, maxLen - 3) + "...";
    }

    @Override
    public String toString() {
        return text;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof OcrResult)) return false;
        OcrResult other = (OcrResult) obj;
        return text.equals(other.text) &&
               Math.abs(confidence - other.confidence) < 0.001f;
    }

    @Override
    public int hashCode() {
        return text.hashCode() + Float.hashCode(confidence);
    }

    /**
     * TextRegion - Represents a detected region of text in the image.
     */
    public static class TextRegion {
        private final int x, y, width, height;
        private final float confidence;

        public TextRegion(int x, int y, int width, int height, float confidence) {
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
            this.confidence = confidence;
        }

        public int getX() { return x; }
        public int getY() { return y; }
        public int getWidth() { return width; }
        public int getHeight() { return height; }
        public float getConfidence() { return confidence; }

        public int getCenterX() { return x + width / 2; }
        public int getCenterY() { return y + height / 2; }

        public double getAspectRatio() {
            return height > 0 ? (double) width / height : 0;
        }

        @Override
        public String toString() {
            return String.format("Region(%d,%d,%d,%d) conf=%.2f",
                x, y, width, height, confidence);
        }
    }

    /**
     * Word - Represents a detected word with its position and confidence.
     */
    public static class Word {
        private final String text;
        private final int x, y, width, height;
        private final float confidence;

        public Word(String text, int x, int y, int width, int height, float confidence) {
            this.text = text;
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
            this.confidence = confidence;
        }

        public String getText() { return text; }
        public int getX() { return x; }
        public int getY() { return y; }
        public int getWidth() { return width; }
        public int getHeight() { return height; }
        public float getConfidence() { return confidence; }

        @Override
        public String toString() {
            return String.format("Word('%s' at %d,%d conf=%.2f)", text, x, y, confidence);
        }
    }

    /**
     * CharacterInfo - Contains information about a recognized character.
     */
    public static class CharacterInfo {
        private final char character;
        private final int x, y, width, height;
        private final float confidence;

        public CharacterInfo(char character, int x, int y, int width, int height, float confidence) {
            this.character = character;
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
            this.confidence = confidence;
        }

        public char getCharacter() { return character; }
        public int getX() { return x; }
        public int getY() { return y; }
        public int getWidth() { return width; }
        public int getHeight() { return height; }
        public float getConfidence() { return confidence; }

        @Override
        public String toString() {
            return String.format("Char('%c' at %d,%d conf=%.2f)", character, x, y, confidence);
        }
    }

    /**
     * Builder class for creating OcrResult instances.
     */
    public static class Builder {
        private String text = "";
        private float confidence = 1.0f;
        private long processingTimeMs = 0;
        private int imageWidth = 0;
        private int imageHeight = 0;
        private final List<TextRegion> regions = new ArrayList<>();
        private final List<Word> words = new ArrayList<>();
        private final List<CharacterInfo> characterInfos = new ArrayList<>();

        public Builder setText(String text) {
            this.text = text != null ? text : "";
            return this;
        }

        public Builder setConfidence(float confidence) {
            this.confidence = Math.max(0, Math.min(1, confidence));
            return this;
        }

        public Builder setProcessingTime(long ms) {
            this.processingTimeMs = ms;
            return this;
        }

        public Builder setImageDimensions(int width, int height) {
            this.imageWidth = width;
            this.imageHeight = height;
            return this;
        }

        public Builder addRegion(TextRegion region) {
            this.regions.add(region);
            return this;
        }

        public Builder addWord(Word word) {
            this.words.add(word);
            return this;
        }

        public Builder addCharacterInfo(CharacterInfo info) {
            this.characterInfos.add(info);
            return this;
        }

        public OcrResult build() {
            OcrResult result = new OcrResult(text, confidence, processingTimeMs, imageWidth, imageHeight);
            result.setRegions(regions);
            result.setWords(words);
            result.setCharacterInfos(characterInfos);
            return result;
        }
    }
}
