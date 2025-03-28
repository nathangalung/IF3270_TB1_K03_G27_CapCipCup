import numpy as np
from typing import Callable, Dict, Optional, Union

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.activations import Activation, Linear
from core.weight_initializers import initialize_weights

class Layer:
    """Base layer interface"""
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def backward(self, inputs: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def update_weights(self, learning_rate: float) -> None:
        raise NotImplementedError

class DenseLayer(Layer):
    """Standard fully-connected layer"""
    
    def __init__(self, input_size: int, output_size: int, activation: Activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = None
        self.bias = None
        self.weights_gradient = None
        self.bias_gradient = None
        self.output = None
        self.z = None
    
    def initialize_weights(self, method: str, **params):
        """Set initial weights using specified method"""
        # Get initialization parameters
        seed = params.get('seed', None)
        init_params = {**params, 'input_size': self.input_size, 'output_size': self.output_size}
        
        # Initialize weights
        self.weights = initialize_weights(
            method=method,
            shape=(self.input_size, self.output_size),
            **init_params
        )
        
        # Initialize bias
        if method in ['zero', 'xavier', 'he']:
            self.bias = np.zeros((1, self.output_size))
        elif method == 'random_uniform':
            lower = params.get('lower', -0.1)
            upper = params.get('upper', 0.1)
            rng = np.random.default_rng(seed)
            self.bias = rng.uniform(lower, upper, size=(1, self.output_size)).astype(np.float32)
        elif method == 'random_normal':
            mean = params.get('mean', 0.0)
            std = params.get('std', 0.1)
            rng = np.random.default_rng(seed)
            self.bias = rng.normal(mean, std, size=(1, self.output_size)).astype(np.float32)
        else:
            raise ValueError(f"Unknown weight initialization method: {method}")
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass computation"""
        if self.weights is None or self.bias is None:
            raise ValueError("Weights and bias must be initialized before forward pass")
        
        # Linear transformation
        self.z = np.dot(inputs, self.weights) + self.bias
        
        # Apply activation function
        self.output = self.activation.forward(self.z)
        
        return self.output
    
    def backward(self, previous_layer_activations: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Backward pass computation"""
        # Bias gradient
        self.bias_gradient = np.sum(gradient, axis=0, keepdims=True)
        
        # Weight gradient
        self.weights_gradient = np.dot(previous_layer_activations.T, gradient)
        
        # L1 regularization
        if hasattr(self, 'l1_lambda') and self.l1_lambda > 0:
            l1_grad_scale = self.l1_lambda / self.weights.size
            epsilon = 1e-8
            l1_grad = l1_grad_scale * self.weights / (np.abs(self.weights) + epsilon)
            self.weights_gradient += l1_grad
        
        # L2 regularization
        if hasattr(self, 'l2_lambda') and self.l2_lambda > 0:
            l2_grad_scale = self.l2_lambda / self.weights.size
            self.weights_gradient += l2_grad_scale * self.weights
        
        # Gradient for previous layer
        return np.dot(gradient, self.weights.T)
    
    def update_weights(self, learning_rate: float) -> None:
        """Update weights using gradients"""
        if self.weights_gradient is None or self.bias_gradient is None:
            raise ValueError("Gradients must be computed before updating weights")
        
        # Apply weight updates
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * self.bias_gradient
    
class RMSNormalizationLayer(Layer):
    """RMS Normalization Layer"""
    
    def __init__(self, input_size: int, epsilon: float = 1e-8):
        self.input_size = input_size
        self.output_size = input_size
        self.epsilon = epsilon
        self.gamma = np.ones((1, input_size))
        self.x = None
        self.rms = None
        self.x_normalized = None
        self.weights = self.gamma
        self.bias = np.zeros((1, input_size))
        self.weights_gradient = None
        self.bias_gradient = None
        self.output = None
        self.activation = Linear()
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply RMS normalization"""
        # Store input
        self.x = inputs
        
        # Calculate root mean square
        ms = np.mean(np.square(inputs), axis=1, keepdims=True)
        self.rms = np.sqrt(ms + self.epsilon)
        self.rms = np.maximum(self.rms, 1e-6)  # Avoid division by zero
        
        # Normalize input
        self.x_normalized = inputs / self.rms
        
        # Scale with gamma parameter
        self.output = self.gamma * self.x_normalized
        
        return self.output
    
    def backward(self, previous_layer_activations: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Compute gradient for RMS normalization"""
        # Handle NaN values
        gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Gradient for gamma
        self.weights_gradient = np.sum(gradient * self.x_normalized, axis=0, keepdims=True)
        self.bias_gradient = np.zeros_like(self.bias)
        
        # Gradient for normalized input
        d_normalized = gradient * self.gamma
        
        # Gradient for RMS
        d_rms = -np.sum(d_normalized * self.x_normalized, axis=1, keepdims=True) / self.rms
        d_ms = d_rms * 0.5 / self.rms
        
        # Gradient for input
        d_x = d_normalized / self.rms
        d_x += 2.0 * self.x * d_ms / self.input_size
        
        return d_x
    
    def update_weights(self, learning_rate: float) -> None:
        """Update gamma parameter"""
        if self.weights_gradient is not None:
            self.gamma -= learning_rate * self.weights_gradient
    
    def initialize_weights(self, method: str, **params):
        """Dummy method for API consistency"""
        pass