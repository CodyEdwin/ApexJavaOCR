#!/usr/bin/env python3
"""
ApexOCR Weight Converter
========================
Converts pre-trained neural network weights from PyTorch (.pth) or TensorFlow/Keras (.h5)
formats to ApexOCR's custom binary format.

Supported architectures:
- CRNN (Convolutional Recurrent Neural Network)
- Standard CNN backbones (VGG, ResNet variants)
- Dense and LSTM layers

Usage:
    # Convert PyTorch CRNN weights
    python convert_weights.py --input crnn.pth --output apex-weights.bin --format pytorch

    # Convert Keras/H5 weights
    python convert_weights.py --input model.h5 --output apex-weights.bin --format keras

    # Convert with custom architecture mapping
    python convert_weights.py --input crnn.pth --output apex-weights.bin \
        --format pytorch --mapping config/architecture_map.json

Author: ApexOCR Team
"""

import argparse
import json
import numpy as np
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False


class WeightConverter:
    """
    Converts neural network weights from various formats to ApexOCR binary format.
    
    The ApexOCR weight file format:
    - Magic number: 0x41504558 ("APEX" in little-endian, 4 bytes)
    - Version: 1 (4 bytes)
    - Number of layers with weights: N (4 bytes)
    - For each layer:
      - Layer name length (1 byte)
      - Layer name (variable)
      - Layer type (1 byte)
      - Weight data size (4 bytes)
      - Weight data (shape + raw float32 values)
      - Bias data size (4 bytes)
      - Bias data (shape + raw float32 values)
    """
    
    # Layer type codes
    LAYER_CONV2D = 1
    LAYER_DENSE = 2
    LAYER_BILSTM = 3
    LAYER_LSTM = 4
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.layer_mapping = {}
        
    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"  [INFO] {message}")
            
    def load_pytorch_weights(self, filepath: str) -> Dict[str, np.ndarray]:
        """Load weights from PyTorch .pth file."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required. Install with: pip install torch")
            
        self.log(f"Loading PyTorch weights from {filepath}")
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # Convert to numpy arrays
        weights = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                weights[key] = value.detach().cpu().numpy()
            else:
                weights[key] = np.array(value)
                
        self.log(f"Loaded {len(weights)} weight tensors")
        return weights
    
    def load_keras_weights(self, filepath: str) -> Dict[str, np.ndarray]:
        """Load weights from Keras .h5 file."""
        if not HAS_TF:
            raise ImportError("TensorFlow/Keras is required. Install with: pip install tensorflow")
            
        self.log(f"Loading Keras weights from {filepath}")
        model = keras.models.load_model(filepath, compile=False)
        
        weights = {}
        for layer in model.layers:
            if layer.weights:
                layer_name = layer.name
                for i, tensor in enumerate(layer.weights):
                    weight_name = f"{layer_name}.{i}"
                    weights[weight_name] = tensor.numpy()
                    
        self.log(f"Loaded {len(weights)} weight tensors")
        return weights
    
    def infer_layer_type(self, name: str, shape: Tuple) -> int:
        """Infer the layer type from name and shape."""
        name_lower = name.lower()
        
        # Conv2D layers
        if 'conv' in name_lower or 'kernel' in name_lower:
            if len(shape) == 4:
                return self.LAYER_CONV2D
                
        # Dense/Linear layers
        if 'dense' in name_lower or 'linear' in name_lower or 'fc' in name_lower:
            if len(shape) == 2:
                return self.LAYER_DENSE
                
        # LSTM layers
        if 'lstm' in name_lower or 'rnn' in name_lower:
            return self.LAYER_BILSTM if 'bi' in name_lower else self.LAYER_LSTM
            
        # Fallback: infer from shape
        if len(shape) == 4:
            return self.LAYER_CONV2D
        elif len(shape) == 2:
            return self.LAYER_DENSE
            
        return self.LAYER_DENSE
    
    def normalize_conv_weight(self, weight: np.ndarray) -> np.ndarray:
        """
        Normalize Conv2D weight from [out_ch, in_ch, h, w] to ApexOCR's expected format.
        
        PyTorch: [out_channels, in_channels, kernel_height, kernel_width]
        Keras:   [kernel_height, kernel_width, in_channels, out_channels]
        
        ApexOCR expects: [kernel_height, kernel_width, in_channels, out_channels]
        """
        if weight.ndim != 4:
            return weight
            
        if weight.shape[0] == weight.shape[1] == weight.shape[2] == weight.shape[3]:
            # Can't determine, return as-is
            return weight
            
        # Check if PyTorch format (out_ch, in_ch, h, w)
        if weight.shape[1] < weight.shape[2] and weight.shape[1] < weight.shape[3]:
            # PyTorch format, need to transpose
            # [out_ch, in_ch, h, w] -> [h, w, in_ch, out_ch]
            return np.transpose(weight, (2, 3, 1, 0))
            
        return weight
    
    def normalize_dense_weight(self, weight: np.ndarray) -> np.ndarray:
        """
        Normalize Dense weight to [input_size, output_size].
        
        Some frameworks use [output_size, input_size], we standardize to
        [input_size, output_size] for matrix multiplication.
        """
        if weight.ndim != 2:
            return weight
            
        # If shape looks like [output, input], transpose
        # Heuristic: first dimension is often smaller for weights
        if weight.shape[0] > weight.shape[1]:
            return weight.T
            
        return weight
    
    def normalize_lstm_weight(self, weight: np.ndarray, is_bidirectional: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize LSTM weights to separate weights and biases.
        
        LSTM gates are typically concatenated:
        PyTorch: [input_size + hidden_size, 4 * hidden_size]
        
        Returns:
            weights: Normalized weight matrix
            biases:  Extracted or zero bias vector
        """
        if weight.ndim != 2:
            return weight, np.zeros(weight.shape[1] if weight.ndim == 1 else weight.shape[-1])
            
        input_size = weight.shape[0]
        hidden_size = weight.shape[1] // 4
        
        # Extract weights for each gate
        # For ApexOCR, we keep the combined format [input+hidden, 4*hidden]
        return weight, np.zeros(4 * hidden_size)
    
    def serialize_tensor(self, tensor: np.ndarray) -> bytes:
        """Serialize a numpy tensor to bytes."""
        shape = np.array(tensor.shape, dtype=np.int64)
        data = tensor.astype(np.float32).tobytes()
        shape_bytes = shape.tobytes()
        return struct.pack('I', len(shape)) + shape_bytes + data
    
    def serialize_layer(self, name: str, layer_type: int, weights: np.ndarray, 
                       biases: Optional[np.ndarray] = None) -> bytes:
        """Serialize a single layer's weights to bytes."""
        # Layer header
        name_bytes = name.encode('utf-8')
        header = struct.pack('B', len(name_bytes)) + name_bytes + struct.pack('B', layer_type)
        
        # Serialize weights
        weight_data = self.serialize_tensor(weights)
        
        # Serialize biases
        if biases is not None and biases.size > 0:
            bias_data = self.serialize_tensor(biases)
        else:
            bias_data = b''
            
        return header + weight_data + bias_data
    
    def convert(self, input_path: str, output_path: str, input_format: str = 'pytorch',
                layer_order: Optional[List[str]] = None, custom_mapping: Optional[Dict] = None):
        """
        Convert weights to ApexOCR format.
        
        Args:
            input_path: Path to input weight file
            output_path: Path to output ApexOCR weight file
            input_format: 'pytorch' or 'keras'
            layer_order: Optional ordered list of layer names
            custom_mapping: Optional custom layer name mapping
        """
        # Load weights
        if input_format == 'pytorch':
            weights = self.load_pytorch_weights(input_path)
        elif input_format == 'keras':
            weights = self.load_keras_weights(input_path)
        else:
            raise ValueError(f"Unsupported format: {input_format}")
            
        # Apply custom mapping if provided
        if custom_mapping:
            self.log("Applying custom layer mapping")
            weights = self._apply_mapping(weights, custom_mapping)
            
        # Serialize layers
        serialized_layers = []
        layer_names = layer_order if layer_order else list(weights.keys())
        
        for name in layer_names:
            if name not in weights:
                self.log(f"Warning: Layer {name} not found in weights, skipping")
                continue
                
            weight = weights[name]
            
            # Normalize weight format
            layer_type = self.infer_layer_type(name, weight.shape)
            
            if layer_type == self.LAYER_CONV2D:
                weight = self.normalize_conv_weight(weight)
            elif layer_type == self.LAYER_DENSE:
                weight = self.normalize_dense_weight(weight)
                
            # Look for corresponding bias
            bias_name = name.replace('.weight', '.bias').replace('_w', '_b')
            if bias_name in weights:
                biases = weights[bias_name]
            else:
                # Try common bias naming patterns
                biases = None
                
            # Serialize
            layer_data = self.serialize_layer(name, layer_type, weight, biases)
            serialized_layers.append(layer_data)
            self.log(f"Serialized layer: {name} ({weight.shape})")
            
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
                
        self.log(f"Successfully wrote {output_path}")
        self.log(f"Total layers: {len(serialized_layers)}")
        
        return len(serialized_layers)
    
    def _apply_mapping(self, weights: Dict[str, np.ndarray], 
                      mapping: Dict) -> Dict[str, np.ndarray]:
        """Apply custom layer name mapping."""
        new_weights = {}
        for old_name, weight in weights.items():
            new_name = old_name
            for pattern, replacement in mapping.get('rename', {}).items():
                new_name = new_name.replace(pattern, replacement)
            new_weights[new_name] = weight
        return new_weights


def create_crnn_converter():
    """
    Create a pre-configured converter for standard CRNN architectures.
    
    This handles the common CRNN structure:
    - Conv2D layers (feature extraction)
    - BiLSTM layers (sequence modeling)
    - Dense layers (classification)
    """
    converter = WeightConverter(verbose=True)
    
    # Standard CRNN layer ordering
    crnn_layer_order = [
        'conv1.weight', 'conv1.bias',
        'conv2.weight', 'conv2.bias',
        'conv3.weight', 'conv3.bias',
        'conv4.weight', 'conv4.bias',
        'conv5.weight', 'conv5.bias',
        'rnn.0.weight_ih_l0', 'rnn.0.weight_hh_l0', 'rnn.0.bias_ih_l0', 'rnn.0.bias_hh_l0',
        'rnn.0.weight_ih_l0_reverse', 'rnn.0.weight_hh_l0_reverse', 'rnn.0.bias_ih_l0_reverse', 'rnn.0.bias_hh_l0_reverse',
        'rnn.1.weight_ih_l0', 'rnn.1.weight_hh_l0', 'rnn.1.bias_ih_l0', 'rnn.1.bias_hh_l0',
        'rnn.1.weight_ih_l0_reverse', 'rnn.1.weight_hh_l0_reverse', 'rnn.1.bias_ih_l0_reverse', 'rnn.1.bias_hh_l0_reverse',
        'fc.weight', 'fc.bias',
    ]
    
    return converter, crnn_layer_order


def main():
    parser = argparse.ArgumentParser(
        description='Convert neural network weights to ApexOCR binary format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert PyTorch CRNN weights
  python convert_weights.py --input crnn_model.pth --output weights.bin --format pytorch
  
  # Convert Keras model with verbose output
  python convert_weights.py --input model.h5 --output weights.bin --format keras --verbose
  
  # Convert with custom layer ordering
  python convert_weights.py --input crnn.pth --output weights.bin \\
      --layer-order config/layer_order.txt
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='Input weight file path')
    parser.add_argument('--output', '-o', required=True, help='Output ApexOCR weight file path')
    parser.add_argument('--format', '-f', choices=['pytorch', 'keras'], default='pytorch',
                       help='Input weight format (default: pytorch)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--layer-order', '-l', help='File containing ordered list of layer names')
    parser.add_argument('--mapping', '-m', help='JSON file with custom layer mapping')
    
    args = parser.parse_args()
    
    # Load layer order if provided
    layer_order = None
    if args.layer_order:
        with open(args.layer_order, 'r') as f:
            layer_order = [line.strip() for line in f if line.strip()]
            
    # Load custom mapping if provided
    custom_mapping = None
    if args.mapping:
        with open(args.mapping, 'r') as f:
            custom_mapping = json.load(f)
    
    # Create converter and run
    converter = WeightConverter(verbose=args.verbose)
    
    try:
        num_layers = converter.convert(
            args.input, args.output, args.format,
            layer_order, custom_mapping
        )
        print(f"Successfully converted {num_layers} layers to {args.output}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
