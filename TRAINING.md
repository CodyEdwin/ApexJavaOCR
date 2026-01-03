# ApexOCR Training Guide

This guide explains how to create training data, train the OCR model, and use the trained weights.

## Prerequisites

- Java 17 or higher
- Gradle 8.x
- At least 2GB of free memory for training

## Step 1: Generate Synthetic Training Data

The project includes a `SyntheticDataGenerator` class that creates training images with known text labels. Each image is named after its text content, making it easy to extract ground truth labels.

### Command to Generate Training Images

```bash
export JAVA_HOME=/path/to/jdk-21.0.9+10
cd /path/to/apex-java-ocr
./gradlew -p apex-ocr-training run --args="./training_data 1000" --no-daemon
```

### Parameters

- `./training_data` - Output directory for generated images
- `1000` - Number of images to generate
- Optional: add min/max text length (default 3-8 characters)

Example with custom text length:
```bash
./gradlew -p apex-ocr-training run --args="./training_data 500 4 10" --no-daemon
```
This generates 500 images with 4-10 character text.

### How It Works

- Creates grayscale images (32px height)
- Renders random alphanumeric text using Arial Bold font
- Adds random noise to simulate real-world conditions
- Images are saved as PNG files named after their text content

Example: `HELLO.png` contains an image with the text "HELLO"

## Step 2: Train the Model

Once you have training data, use the `OCRTrainer` class to train the network.

### Command to Run Training

```bash
export JAVA_HOME=/path/to/jdk-21.0.9+10
cd /path/to/apex-java-ocr
./gradlew -p apex-ocr-training run --args="./training_data 50" --no-daemon
```

### Parameters

- `./training_data` - Directory containing training images
- `50` - Number of epochs (default is 100)

Example with 20 epochs:
```bash
./gradlew -p apex-ocr-training run --args="./training_data 20" --no-daemon
```

### What the Trainer Does

1. Loads all images from the training directory
2. Extracts labels from filenames (removes .png extension)
3. Runs forward pass through the network
4. Computes CTC loss between predictions and ground truth
5. Updates weights using gradient descent
6. Saves trained weights to `apex-ocr-weights.bin` after each epoch

### Training Output

The trainer will output progress like:
```
Loaded 1000 training samples
Epoch 1/50 - Loss: 12.3456
Epoch 2/50 - Loss: 8.7654
...
Training complete!
Final loss: 0.1234
Saved trained weights to apex-ocr-weights.bin
```

## Step 3: Use the Trained Model

Once training is complete, use the trained weights with the OCR engine.

### Option A: Copy Weights to CLI Module

```bash
cp apex-ocr-weights.bin apex-ocr-cli/apex-ocr-weights.bin
```

### Option B: Run with CLI Module

```bash
export JAVA_HOME=/path/to/jdk-21.0.9+10
cd /path/to/apex-java-ocr
./gradlew -p apex-ocr-cli run --args="path/to/your/image.jpg" --no-daemon
```

The CLI will automatically load `apex-ocr-cli/apex-ocr-weights.bin` if it exists.

## Training Tips

### Vocabulary

The current training code uses uppercase letters and digits:
- Characters: `A-Z`, `0-9`
- Total classes: 36 + 1 (blank token for CTC)

### Recommended Training Size

- **Minimum**: 100-500 images for quick testing
- **Good results**: 5,000-10,000 images
- **Production**: 50,000+ images with diverse fonts and backgrounds

### Improving Results

1. **More data**: Generate more images with varied text lengths
2. **Longer training**: Increase epochs to 100-200
3. **Adjust learning rate**: Modify in OCRTrainer.java (default: 0.001)
4. **Adjust batch size**: Modify in OCRTrainer.java (default: 8)

### Using Real Training Data

To use real images instead of synthetic data:

1. Create a directory with your training images
2. Rename images to match their text content (e.g., `HELLO.png`)
3. Run the trainer on that directory

Example directory structure:
```
my_training_data/
  HELLO.png
  WORLD.png
  APEX.png
  OCR.png
```

Run training:
```bash
./gradlew -p apex-ocr-training run --args="./my_training_data 50" --no-daemon
```

## File Locations

| File | Location | Description |
|------|----------|-------------|
| SyntheticDataGenerator | `apex-ocr-training/src/main/java/com/apexocr/training/SyntheticDataGenerator.java` | Creates training images |
| OCRTrainer | `apex-ocr-training/src/main/java/com/apexocr/training/OCRTrainer.java` | Trains the network |
| DocumentPreprocessor | `apex-ocr-training/src/main/java/com/apexocr/training/DocumentPreprocessor.java` | Preprocesses scanned documents |
| DocumentLabelExtractor | `apex-ocr-training/src/main/java/com/apexocr/training/DocumentLabelExtractor.java` | Extracts labels from filenames |
| GroundTruthValidator | `apex-ocr-training/src/main/java/com/apexocr/training/GroundTruthValidator.java` | Validates and corrects labels |
| Trained Weights | Project root `apex-ocr-weights.bin` | Output after training |

## Using Real Scanned Documents (Court Filings)

If you have a collection of scanned court documents, you can use them to train a highly effective OCR model. The project includes utilities specifically designed for processing scanned documents.

### Overview of Court Document Workflow

```
1. PREPROCESS      - Clean up scanned images (grayscale, binarize, crop)
2. LABEL EXTRACT   - Extract case numbers/docket numbers from filenames
3. VALIDATE        - Review and correct extracted labels
4. TRAIN           - Train the OCR model
5. USE             - Apply trained model to new documents
```

### Step 1: Preprocess Court Documents

Court documents often have noise, skew, and varying sizes. Use `DocumentPreprocessor` to normalize them.

```bash
export JAVA_HOME=/path/to/jdk-21.0.9+10
cd /path/to/apex-java-ocr

# Basic preprocessing - grayscale, normalize size, remove noise
./gradlew -p apex-ocr-training run --args="com.apexocr.training.DocumentPreprocessor ./court_scans ./processed" --no-daemon

# With custom height (48px) for higher accuracy
./gradlew -p apex-ocr-training run --args="com.apexocr.training.DocumentPreprocessor ./court_scans ./processed --height 48" --no-daemon

# Extract text regions (16px height strips)
./gradlew -p apex-ocr-training run --args="com.apexocr.training.DocumentPreprocessor ./court_scans ./regions --regions 16" --no-daemon
```

### Step 2: Extract Labels from Filenames

Court documents often have case numbers in their filenames. Use `DocumentLabelExtractor` to extract and normalize them.

```bash
export JAVA_HOME=/path/to/jdk-21.0.9+10
cd /path/to/apex-java-ocr

# Standard pattern matching (CV-2024-001234, 1:24-cv-00123)
./gradlew -p apex-ocr-training run --args="com.apexocr.training.DocumentLabelExtractor ./processed ./labeled STANDARD"

# Numeric-only patterns
./gradlew -p apex-ocr-training run --args="com.apexocr.training.DocumentLabelExtractor ./processed ./labeled SHORT"

# All patterns
./gradlew -p apex-ocr-training run --args="com.apexocr.training.DocumentLabelExtractor ./processed ./labeled ALL"
```

This will rename files to match their extracted case numbers:
- Before: `scan_001234_page1.png`
- After: `CV-2024-001234.png`

### Step 3: Validate and Correct Labels

Review the extracted labels and make corrections using `GroundTruthValidator`.

```bash
export JAVA_HOME=/path/to/jdk-21.0.9+10
cd /path/to/apex-java-ocr

./gradlew -p apex-ocr-training run --args="com.apexocr.training.GroundTruthValidator ./labeled ./validated" --no-daemon
```

During validation, you can:
- Press Enter to accept the current label
- Type `c NEWLABEL` to change the label
- Type `r` to regenerate from filename
- Type `s` to skip
- Type `q` to quit

### Step 4: Prepare Training Data

Once labels are validated, prepare them for training with the correct vocabulary.

```bash
# After validation, export to training-ready format
# The validator automatically creates training_labels.txt
```

### Step 5: Train on Court Documents

```bash
export JAVA_HOME=/path/to/jdk-21.0.9+10
cd /path/to/apex-java-ocr

# Train on validated court documents
./gradlew -p apex-ocr-training run --args="./validated 100" --no-daemon
```

### Example Workflow for Court Filings

```bash
# Assuming you have court scans in ./scans

# 1. Preprocess for OCR
./gradlew -p apex-ocr-training run --args="com.apexocr.training.DocumentPreprocessor ./scans ./processed --height 48" --no-daemon

# 2. Extract case numbers from filenames
./gradlew -p apex-ocr-training run --args="com.apexocr.training.DocumentLabelExtractor ./processed ./labeled STANDARD" --no-daemon

# 3. Validate labels (interactive)
./gradlew -p apex-ocr-training run --args="com.apexocr.training.GroundTruthValidator ./labeled ./validated" --no-daemon

# 4. Train the model
./gradlew -p apex-ocr-training run --args="./validated 100" --no-daemon

# 5. Copy weights to CLI
cp apex-ocr-weights.bin apex-ocr-cli/apex-ocr-weights.bin

# 6. Test on a new document
./gradlew -p apex-ocr-cli run --args="path/to/new_document.png" --no-daemon
```

### Tips for Court Documents

1. **Focus on specific document types**: Start with one type (e.g., minute orders, docket sheets) for best initial results

2. **Use header/footer text**: Court documents typically have consistent headers with case information

3. **Extract docket numbers**: These are usually prominently displayed and machine-readable

4. **Batch by court type**: Different courts may have different formats - train separate models if needed

5. **Quality matters**: Remove severely degraded or illegible scans from training

### Recommended Vocabulary for Court Documents

The default vocabulary includes `A-Z` and `0-9`. For court documents, you may want to add:
- Hyphens and slashes (`-`, `/`)
- Colons (`:`)
- Parentheses (`(`, `)`)

Modify `OCRTrainer.java` to extend the vocabulary:

```java
private static final String UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
private static final String DIGITS = "0123456789";
private static final String SPECIAL = "-/:(). ";
private final String vocabulary = UPPERCASE + DIGITS + SPECIAL;
```

## Troubleshooting

### Out of Memory

If training runs out of memory, reduce batch size in OCRTrainer.java:
```java
private int batchSize = 4; // Reduce from 8 to 4
```

### Low Accuracy

- Train for more epochs
- Generate more training data
- Ensure text in images is clear and readable
- Use consistent font sizes and image dimensions

### Images Not Loading

Ensure images are in PNG format and named with alphanumeric characters only.

## Next Steps

After training, you can:

1. Test the trained model on new images
2. Continue training with more epochs for better accuracy
3. Generate more diverse training data
4. Experiment with different network hyperparameters
