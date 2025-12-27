# ApexJavaOCR

A high-performance, pure Java OCR (Optical Character Recognition) engine designed to outperform traditional OCR systems like Tesseract through modern deep learning architectures optimized for the JVM.

## Overview

ApexJavaOCR implements a CRNN (Convolutional Recurrent Neural Network) architecture with CTC (Connectionist Temporal Classification) decoding, specifically designed for efficient text recognition in the Java ecosystem. The engine achieves superior performance through aggressive memory management, parallel processing, and optimized tensor operations.

### Key Features

- **Pure Java Implementation**: No native dependencies, runs anywhere Java is available
- **CRNN Architecture**: Combines CNN feature extraction with bidirectional LSTM sequence modeling
- **CTC Decoding**: Beam search decoding for accurate sequence recognition without alignment
- **High Performance**: Optimized tensor operations with off-heap memory management
- **Parallel Processing**: Multi-threaded batch processing for maximum throughput
- **Comprehensive Preprocessing**: Built-in image enhancement and binarization pipeline
- **Flexible Configuration**: Multiple engine modes for accuracy or speed optimization

## Architecture

### Core Components

```
apex-java-ocr/
├── apex-ocr-core/              # Tensor operations and neural network layers
│   ├── tensor/
│   │   ├── Tensor.java         # Multi-dimensional array implementation
│   │   ├── TensorOperations.java # High-performance tensor operations
│   │   └── MemoryManager.java  # Off-heap memory management
│   ├── neural/
│   │   ├── Layer.java          # Base layer interface
│   │   ├── Conv2D.java         # 2D convolutional layer
│   │   ├── Dense.java          # Fully connected layer
│   │   ├── MaxPool2D.java      # Max pooling layer
│   │   └── BiLSTM.java         # Bidirectional LSTM layer
│   └── ctc/
│       └── CTCDecoder.java     # CTC beam search decoder
├── apex-ocr-preprocessing/     # Image preprocessing pipeline
│   └── ImagePreprocessor.java
├── apex-ocr-engine/            # Main OCR engine
│   ├── OcrEngine.java          # Core recognition engine
│   └── OcrResult.java          # Result container
└── apex-ocr-cli/               # Command-line interface
    └── Main.java
```

### Network Architecture

The OCR engine uses a CRNN architecture optimized for text recognition:

1. **Feature Extraction (CNN)**
   - Multiple convolutional layers with ReLU activation
   - Max pooling for spatial downsampling
   - Batch normalization for stable training

2. **Sequence Modeling (BiLSTM)**
   - Bidirectional LSTM layers for context understanding
   - Captures character dependencies in both directions
   - Dropout regularization for robustness

3. **Transcription (CTC)**
   - Connectionist Temporal Classification
   - Beam search decoding for optimal sequence prediction
   - Handles variable-length input and output

## Building the Project

### Prerequisites

- Java 21 or higher
- Gradle 8.x

### Build Commands

```bash
# Build all modules
./gradlew build

# Build without tests
./gradlew assemble

# Run tests
./gradlew test

# Clean build artifacts
./gradlew clean

# Build all projects
./gradlew buildAll
```

### Running Tests

```bash
# Run all tests
./gradlew test

# Run specific test class
./gradlew test --tests "com.apexocr.OcrEngineTest"

# Run with verbose output
./gradlew test --info
```

## Usage

### Basic OCR

```java
try (OcrEngine engine = new OcrEngine()) {
    engine.initialize();
    
    OcrResult result = engine.processFile("image.png");
    System.out.println("Recognized: " + result.getText());
    System.out.println("Confidence: " + (result.getConfidence() * 100) + "%");
}
```

### With Custom Configuration

```java
OcrEngine.EngineConfig config = OcrEngine.EngineConfig.forHighAccuracy();
config.beamWidth = 20;
config.confidenceThreshold = 0.3f;

try (OcrEngine engine = new OcrEngine(config)) {
    engine.initialize();
    
    BufferedImage image = ImageIO.read(new File("document.png"));
    OcrResult result = engine.process(image);
    
    System.out.println("Text: " + result.getText());
    System.out.println("Processing time: " + result.getProcessingTimeMs() + "ms");
}
```

### Batch Processing

```java
List<String> filePaths = Arrays.asList(
    "page1.png", "page2.png", "page3.png"
);

try (OcrEngine engine = new OcrEngine()) {
    engine.initialize();
    
    List<OcrResult> results = engine.processFiles(filePaths);
    
    for (int i = 0; i < results.size(); i++) {
        System.out.println("Page " + (i + 1) + ": " + results.get(i).getText());
    }
}
```

### Custom Preprocessing

```java
ImagePreprocessor.PreprocessingConfig config =
    ImagePreprocessor.PreprocessingConfig.forChallengingImages();
config.binarizationMethod = BinarizationMethod.SAUVOLA;
config.applyContrastEnhancement = true;

ImagePreprocessor preprocessor = new ImagePreprocessor(config);

// Get preprocessed image
BufferedImage processed = preprocessor.process(originalImage);

// Or extract features directly for neural network input
float[][] features = preprocessor.extractFeatures(processed);
```

### CLI Usage

```bash
# Process a single image
apex-ocr -i image.png

# Process with JSON output
apex-ocr -i image.png -f json -o result.json

# Process entire directory
apex-ocr -i ./images/ -o all_text.txt

# High accuracy mode
apex-ocr -i image.png -c accuracy

# Fast mode
apex-ocr -i image.png -c speed

# Verbose output
apex-ocr -i image.png -v
```

## Configuration Options

### Engine Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `batchSize` | 1 | Number of images to process simultaneously |
| `numThreads` | CPU cores | Worker threads for parallel processing |
| `beamWidth` | 10 | CTC beam search width (higher = more accurate) |
| `confidenceThreshold` | 0.5 | Minimum confidence for valid recognition |
| `enablePreprocessing` | true | Apply image preprocessing pipeline |

### Preprocessing Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `targetHeight` | 32 | Neural network input height |
| `binarizationMethod` | SAUVOLA | Thresholding algorithm |
| `applyContrastEnhancement` | true | Apply histogram equalization |
| `applyNoiseReduction` | true | Apply median filter |
| `medianFilterSize` | 3 | Noise reduction kernel size |

### Configuration Presets

```java
// High accuracy (slower)
OcrEngine.EngineConfig.forHighAccuracy();

// Maximum speed (lower accuracy)
OcrEngine.EngineConfig.forSpeed();

// Default balance
OcrEngine.EngineConfig.createDefault();
```

## Performance Characteristics

### Memory Management

The engine uses off-heap memory allocation through `MemoryManager` to avoid garbage collection overhead during inference:

- Tensor data stored in native memory
- Direct memory access for maximum performance
- Configurable memory pools for large inputs

### Parallel Processing

- Multi-threaded batch processing
- Parallel tensor operations via ForkJoinPool
- Optimized for modern multi-core processors

### Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Single image processing | < 200ms | Standard document, modern CPU |
| Batch throughput | > 20 img/sec | 8-core CPU, batch size 4 |
| Memory usage | < 2GB | For typical document processing |
| Accuracy (clean text) | > 95% | CER on standard datasets |

## Vocabulary and Character Set

The default vocabulary supports:

- Lowercase English letters (a-z)
- Uppercase English letters (A-Z)
- Digits (0-9)
- Common punctuation marks
- Extended ASCII characters

Custom vocabulary can be provided during engine initialization:

```java
String[] customVocab = {"a", "b", "c", "1", "2", "3", " "};
OcrEngine.EngineConfig config = new OcrEngine.EngineConfig();
config.vocabulary = customVocab;
```

## API Reference

### OcrEngine

```java
// Initialize and process
void initialize()
OcrResult process(BufferedImage image)
OcrResult processFile(String path) throws IOException
List<OcrResult> processBatch(List<BufferedImage> images)
List<OcrResult> processFiles(List<String> paths) throws IOException

// State management
void resetState()
void eval()   // Evaluation mode
void train()  // Training mode

// Information
String getArchitectureSummary()
long getParameterCount()
int getLayerCount()
```

### OcrResult

```java
// Basic properties
String getText()
float getConfidence()        // 0-1
float getConfidencePercent() // 0-100
long getProcessingTimeMs()
int getImageWidth()
int getImageHeight()

// Derived properties
int getWordCount()
int getCharacterCount()
boolean hasHighConfidence(float threshold)

// Detailed results
List<TextRegion> getRegions()
List<Word> getWords()
List<CharacterInfo> getCharacterInfos()
```

## Testing

The project includes comprehensive unit tests covering:

- Tensor operations and memory management
- Neural network layer forward passes
- CTC decoding accuracy
- Image preprocessing pipeline
- Full OCR pipeline integration
- Performance benchmarks

Run tests with:
```bash
./gradlew test
```

## Future Enhancements

- Pre-trained model loading (ONNX, TensorFlow Lite)
- GPU acceleration via Java bindings
- Multi-language support
- Handwriting recognition
- Table and structure detection
- PDF document processing

## License

This project is proprietary software. All rights reserved.

## Contributing

For issues, enhancements, or support, please contact the development team.

## Version History

- **1.0.0** - Initial release
  - CRNN architecture implementation
  - CTC beam search decoding
  - Comprehensive preprocessing pipeline
  - CLI interface
  - Full test coverage
