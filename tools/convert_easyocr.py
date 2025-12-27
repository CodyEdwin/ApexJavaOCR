#!/usr/bin/env python3
"""
EasyOCR to ApexOCR Weight Converter
====================================
Converts EasyOCR's english_g2.zip model to ApexOCR's custom binary format.

EasyOCR uses a CRNN architecture with:
- CustomCNN feature extractor
- Bidirectional LSTM sequence encoder
- Attention-based or CTC decoder

The converter extracts weights from the PyTorch model and converts them
to ApexOCR's binary format.

Usage:
    # Convert english_g2.zip directly
    python convert_easyocr.py --input english_g2.zip --output apex-english.bin

    # Convert extracted .pth file
    python convert_easyocr.py --input english_g2.pth --output apex-english.bin

    # With vocabulary extraction
    python convert_easyocr.py --input english_g2.zip --output apex-english.bin --extract-vocab

Author: ApexOCR Team
"""

import argparse
import json
import numpy as np
import struct
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class EasyOCRConverter:
    """
    Converts EasyOCR models to ApexOCR binary format.
    
    EasyOCR Model Structure (english_g2):
    ├── model_weights.pth (or various naming conventions)
    ├── modules.json (optional)
    └── vocabulary.json (optional)
    
    Common layer names in EasyOCR models:
    - Feature extractor: CNN layers (conv1, conv2, ...)
    - Sequence encoder: rnn layers (rnn.weight_ih, rnn.weight_hh, ...)
    - Decoder: fc or attention layers
    """
    
    # Layer type codes (matching ApexOCR)
    LAYER_CONV2D = 1
    LAYER_DENSE = 2
    LAYER_BILSTM = 3
    LAYER_LSTM = 4
    
    # EasyOCR specific layer name patterns
    CNN_PATTERNS = ['conv', 'cnn', 'feature', 'body']
    RNN_PATTERNS = ['rnn', 'lstm', 'encoder', 'bi']
    FC_PATTERNS = ['fc', 'classifier', 'decoder', 'pred', 'out']
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.vocabulary = []
        
    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"  [INFO] {message}")
            
    def print(self, message: str):
        """Print message (always)."""
        print(f"  [EasyOCR] {message}")
            
    def find_model_file(self, source: str) -> Tuple[str, bool]:
        """
        Find the model .pth file from source (zip or file path).
        
        Returns:
            Tuple of (model_path, is_temp)
        """
        source_path = Path(source)
        
        # Check if it's a zip file
        if source_path.suffix.lower() == '.zip':
            self.print(f"Extracting from zip: {source}")
            
            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            
            with zipfile.ZipFile(source, 'r') as zip_ref:
                # List all files
                file_list = zip_ref.namelist()
                self.print(f"Files in zip: {file_list}")
                
                # Find .pth file
                pth_files = [f for f in file_list if f.endswith('.pth')]
                if not pth_files:
                    # Try .pt files
                    pth_files = [f for f in file_list if f.endswith('.pt')]
                    
                if pth_files:
                    # Extract the first .pth/.pt file
                    model_file = pth_files[0]
                    zip_ref.extract(model_file, temp_dir)
                    model_path = str(Path(temp_dir) / model_file)
                    self.print(f"Extracted model: {model_path}")
                    return model_path, True
                else:
                    # Look for model weights in different formats
                    for name in ['model_weights', 'model', 'net', 'crnn']:
                        for ext in ['.pth', '.pt', '.pth.tar']:
                            for f in file_list:
                                if name + ext in f:
                                    zip_ref.extract(f, temp_dir)
                                    model_path = str(Path(temp_dir) / f)
                                    self.print(f"Found model: {model_path}")
                                    return model_path, True
                    
            raise FileNotFoundError(f"No model file found in {source}")
            
        # It's already a model file
        if source_path.exists():
            return str(source_path), False
            
        raise FileNotFoundError(f"File not found: {source}")
        
    def find_vocabulary(self, source: str) -> Optional[List[str]]:
        """
        Extract vocabulary from EasyOCR model or zip file.
        
        EasyOCR stores vocabulary in different formats:
        1. Inside the state_dict as 'char2idx'/'idx2char'
        2. As a separate vocab.json file
        3. As character list in metadata
        """
        source_path = Path(source)
        
        # Check if it's a zip file
        if source_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(source, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                # Look for vocabulary files
                for name in ['vocab', 'vocabulary', 'char2idx', 'labels']:
                    for ext in ['.json', '.txt']:
                        for f in file_list:
                            if name in f.lower() and ext in f:
                                self.print(f"Found vocabulary: {f}")
                                try:
                                    content = zip_ref.read(f).decode('utf-8')
                                    if ext == '.json':
                                        vocab = json.loads(content)
                                        if isinstance(vocab, dict):
                                            # Convert dict to ordered list
                                            return list(vocab.keys())
                                        return vocab
                                    else:
                                        return content.strip().split('\n')
                                except Exception as e:
                                    self.log(f"Failed to read vocabulary: {e}")
        else:
            # Try to load from model checkpoint
            if HAS_TORCH:
                try:
                    checkpoint = torch.load(source, map_location='cpu')
                    if isinstance(checkpoint, dict):
                        # Look for vocabulary in common keys
                        for key in ['char_list', 'vocab', 'alphabet', 'characters', 'labels']:
                            if key in checkpoint:
                                vocab = checkpoint[key]
                                if isinstance(vocab, list):
                                    return vocab
                                elif isinstance(vocab, dict):
                                    return list(vocab.keys())
                        # Look for idx2char mappings
                        for key in checkpoint.keys():
                            if 'idx' in key.lower() and 'char' in key.lower():
                                try:
                                    idx2char = checkpoint[key]
                                    if isinstance(idx2char, dict):
                                        return list(idx2char.values())
                                except:
                                    pass
                except Exception as e:
                    self.log(f"Could not extract vocabulary from model: {e}")
                    
        return None
        
    def load_easyocr_model(self, source: str) -> Tuple[Dict[str, np.ndarray], Optional[List[str]]]:
        """
        Load EasyOCR model weights.
        
        Returns:
            Tuple of (state_dict, vocabulary)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required. Install with: pip install torch")
            
        model_path, is_temp = self.find_model_file(source)
        vocabulary = self.find_vocabulary(source)
        
        self.print(f"Loading model: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Clean up temp file if created
        if is_temp:
            try:
                import shutil
                shutil.rmtree(Path(model_path).parent)
            except:
                pass
            
        # Extract state dict
        if isinstance(checkpoint, dict):
            # EasyOCR stores weights in 'state_dict' or directly
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'net' in checkpoint:
                state_dict = checkpoint['net']
            else:
                # Filter out non-weight keys
                state_dict = {k: v for k, v in checkpoint.items() 
                             if isinstance(v, torch.Tensor)}
        else:
            state_dict = checkpoint
            
        # Convert to numpy
        numpy_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                numpy_dict[key] = value.detach().cpu().numpy()
                
        self.print(f"Loaded {len(numpy_dict)} weight tensors")
        
        return numpy_dict, vocabulary
        
    def infer_layer_type(self, name: str, shape: Tuple) -> int:
        """Infer layer type from name and shape."""
        name_lower = name.lower()
        
        # Check patterns
        for pattern in self.CNN_PATTERNS:
            if pattern in name_lower and len(shape) == 4:
                return self.LAYER_CONV2D
                
        for pattern in self.RNN_PATTERNS:
            if pattern in name_lower:
                return self.LAYER_BILSTM if 'bi' in name_lower else self.LAYER_LSTM
                
        for pattern in self.FC_PATTERNS:
            if pattern in name_lower:
                return self.LAYER_DENSE
                
        # Fallback: infer from shape
        if len(shape) == 4:
            return self.LAYER_CONV2D
        elif len(shape) == 2:
            return self.LAYER_DENSE
            
        return self.LAYER_DENSE
        
    def normalize_weight(self, name: str, weight: np.ndarray) -> np.ndarray:
        """
        Normalize weight tensor for ApexOCR format.
        
        Handles:
        - Conv2D: PyTorch [out_ch, in_ch, h, w] -> ApexOCR [h, w, in_ch, out_ch]
        - Dense: Standardize to [input, output]
        - LSTM: Combine gate weights
        """
        if weight.ndim < 2:
            return weight
            
        layer_type = self.infer_layer_type(name, weight.shape)
        
        # Conv2D normalization
        if layer_type == self.LAYER_CONV2D and weight.ndim == 4:
            # PyTorch: [out_channels, in_channels, kernel_h, kernel_w]
            # ApexOCR expects: [kernel_h, kernel_w, in_channels, out_channels]
            
            # Check if it's PyTorch format (out_ch is typically larger than in_ch for later layers)
            if weight.shape[0] == weight.shape[1] == weight.shape[2] == weight.shape[3]:
                return weight
                
            # Detect format and convert if needed
            if weight.shape[0] > weight.shape[1]:
                # Likely PyTorch format [out, in, h, w] -> need transpose
                return np.transpose(weight, (2, 3, 1, 0))
            else:
                # Likely Keras format [h, w, in, out] -> already correct
                return weight
                
        # Dense normalization
        elif layer_type == self.LAYER_DENSE and weight.ndim == 2:
            # Standardize to [input, output]
            # If first dim > second dim, it's likely [output, input]
            if weight.shape[0] > weight.shape[1]:
                return weight.T
            return weight
            
        return weight
        
    def get_corresponding_bias(self, name: str, state_dict: Dict) -> Optional[np.ndarray]:
        """Find the bias tensor corresponding to a weight tensor."""
        # Common bias naming patterns
        bias_names = [
            name.replace('.weight', '.bias').replace('_w', '_b'),
            name + '.bias',
            name.replace('weight', 'bias'),
        ]
        
        for bias_name in bias_names:
            if bias_name in state_dict:
                bias = state_dict[bias_name]
                if isinstance(bias, np.ndarray):
                    return bias
                    
        return None
        
    def serialize_tensor(self, tensor: np.ndarray) -> bytes:
        """Serialize a numpy tensor to bytes."""
        shape = np.array(tensor.shape, dtype=np.int64)
        data = tensor.astype(np.float32).tobytes()
        shape_bytes = shape.tobytes()
        return struct.pack('I', len(shape)) + shape_bytes + data
        
    def serialize_layer(self, name: str, layer_type: int, weights: np.ndarray, 
                       biases: Optional[np.ndarray] = None) -> bytes:
        """Serialize a single layer's weights to bytes."""
        name_bytes = name.encode('utf-8')
        header = struct.pack('B', len(name_bytes)) + name_bytes + struct.pack('B', layer_type)
        weight_data = self.serialize_tensor(weights)
        
        if biases is not None and biases.size > 0:
            bias_data = self.serialize_tensor(biases)
        else:
            bias_data = b''
            
        return header + weight_data + bias_data
        
    def convert(self, input_path: str, output_path: str, 
                extract_vocab: bool = True) -> Tuple[int, Optional[List[str]]]:
        """
        Convert EasyOCR model to ApexOCR format.
        
        Args:
            input_path: Path to .zip or .pth file
            output_path: Path to output .bin file
            extract_vocab: Whether to extract vocabulary
            
        Returns:
            Tuple of (num_layers, vocabulary)
        """
        self.print(f"Converting: {input_path}")
        self.print(f"Output: {output_path}")
        
        # Load model
        state_dict, vocabulary = self.load_easyocr_model(input_path)
        
        if extract_vocab and vocabulary:
            self.vocabulary = vocabulary
            self.print(f"Vocabulary: {len(vocabulary)} characters")
            
        # Filter and sort layers
        # EasyOCR has specific layer naming: module.conv1.weight, module.rnn.weight_ih_l0, etc.
        filtered_layers = {}
        for name, weight in state_dict.items():
            # Skip non-weight tensors (attention masks, etc.)
            if weight.ndim < 2:
                continue
                
            # Remove 'module.' prefix if present
            clean_name = name.replace('module.', '')
            
            # Simplify name for ApexOCR
            # Convert 'rnn.weight_ih_l0' to 'rnn_forward_weights'
            simplified = self.simplify_layer_name(clean_name)
            filtered_layers[simplified] = weight
            
        self.print(f"Filtered to {len(filtered_layers)} weight layers")
        
        # Serialize layers
        serialized_layers = []
        layer_info = []
        
        for name, weight in filtered_layers.items():
            # Normalize weight format
            normalized = self.normalize_weight(name, weight)
            
            # Get corresponding bias
            bias = self.get_corresponding_bias(name, state_dict)
            
            # Infer layer type
            layer_type = self.infer_layer_type(name, normalized.shape)
            
            # Serialize
            layer_data = self.serialize_layer(name, layer_type, normalized, bias)
            serialized_layers.append(layer_data)
            
            layer_info.append({
                'name': name,
                'type': layer_type,
                'shape': list(normalized.shape),
                'bias_shape': list(bias.shape) if bias is not None else []
            })
            
        # Write output file
        with open(output_path, 'wb') as f:
            # Magic number "APEX"
            f.write(struct.pack('I', 0x41504558))
            # Version
            f.write(struct.pack('I', 1))
            # Number of layers
            f.write(struct.pack('I', len(serialized_layers)))
            # Layer data
            for layer_data in serialized_layers:
                f.write(layer_data)
                
        self.print(f"Successfully wrote {output_path}")
        self.print(f"Total layers: {len(serialized_layers)}")
        
        # Save vocabulary if extracted
        if vocabulary and extract_vocab:
            vocab_path = output_path.replace('.bin', '-vocab.json')
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(vocabulary, f, ensure_ascii=False, indent=2)
            self.print(f"Vocabulary saved: {vocab_path}")
            
        # Save layer info
        info_path = output_path.replace('.bin', '-layers.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(layer_info, f, indent=2)
        self.print(f"Layer info saved: {info_path}")
            
        return len(serialized_layers), vocabulary
        
    def simplify_layer_name(self, name: str) -> str:
        """Simplify layer name for ApexOCR."""
        # Handle EasyOCR specific naming
        
        # Remove RNN specific suffixes
        name = name.replace('_ih_l0', '_forward')
        name = name.replace('_hh_l0', '_forward_hidden')
        name = name.replace('_ih_l0_reverse', '_backward')
        name = name.replace('_hh_l0_reverse', '_backward_hidden')
        
        # Simplify common patterns
        name = name.replace('weight_ih', 'w_input')
        name = name.replace('weight_hh', 'w_hidden')
        name = name.replace('bias_ih', 'b_input')
        name = name.replace('bias_hh', 'b_hidden')
        
        return name
        
    def print_summary(self, state_dict: Dict[str, np.ndarray], output_path: str):
        """Print a summary of the converted model."""
        print("\n" + "="*60)
        print("EasyOCR to ApexOCR Conversion Summary")
        print("="*60)
        
        # Count layer types
        conv_count = 0
        rnn_count = 0
        fc_count = 0
        other_count = 0
        
        for name, weight in state_dict.items():
            if weight.ndim < 2:
                continue
            layer_type = self.infer_layer_type(name, weight.shape)
            if layer_type == self.LAYER_CONV2D:
                conv_count += 1
            elif layer_type in [self.LAYER_BILSTM, self.LAYER_LSTM]:
                rnn_count += 1
            elif layer_type == self.LAYER_DENSE:
                fc_count += 1
            else:
                other_count += 1
                
        print(f"Input model: {output_path}")
        print(f"Total weight tensors: {len([w for w in state_dict.values() if w.ndim >= 2])}")
        print(f"  - Conv2D layers: {conv_count}")
        print(f"  - RNN/LSTM layers: {rnn_count}")
        print(f"  - Dense/FC layers: {fc_count}")
        print(f"  - Other: {other_count}")
        
        # Calculate total parameters
        total_params = sum(w.size for w in state_dict.values() if isinstance(w, np.ndarray))
        print(f"Total parameters: {total_params:,}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert EasyOCR models to ApexOCR binary format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert english_g2.zip
  python convert_easyocr.py --input english_g2.zip --output apex-english.bin
  
  # Convert with verbose output
  python convert_easyocr.py --input english_g2.zip --output apex-english.bin --verbose
  
  # Convert .pth file directly
  python convert_easyocr.py --input model.pth --output apex-model.bin
  
  # Convert and save vocabulary
  python convert_easyocr.py --input english_g2.zip --output apex-english.bin --extract-vocab
        """
    )
    
    parser.add_argument('--input', '-i', required=True, 
                       help='Input file (.zip or .pth)')
    parser.add_argument('--output', '-o', required=True, 
                       help='Output ApexOCR weight file (.bin)')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose output')
    parser.add_argument('--no-vocab', action='store_true',
                       help='Skip vocabulary extraction')
    
    args = parser.parse_args()
    
    # Create converter
    converter = EasyOCRConverter(verbose=args.verbose)
    
    try:
        num_layers, vocabulary = converter.convert(
            args.input,
            args.output,
            extract_vocab=not args.no_vocab
        )
        
        converter.print_summary({}, args.input)
        converter.print(f"Successfully converted {num_layers} layers to {args.output}")
        
        if vocabulary:
            converter.print(f"Vocabulary: {len(vocabulary)} characters")
            print(f"\nFirst 20 characters: {vocabulary[:20]}")
            print(f"Last 20 characters: {vocabulary[-20:]}")
            
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
