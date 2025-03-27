import numpy as np
from typing import Callable, Dict, Optional, Union, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.activations import Activation, Linear
from core.weight_initializers import initialize_weights

class Layer:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def backward(self, inputs: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def update_weights(self, learning_rate: float) -> None:
        raise NotImplementedError

class DenseLayer(Layer):
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
        seed = params.get('seed', None)
        init_params = {**params, 'input_size': self.input_size, 'output_size': self.output_size}
        
        self.weights = initialize_weights(
            method=method,
            shape=(self.input_size, self.output_size),
            **init_params
        )
        
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
        if self.weights is None or self.bias is None:
            raise ValueError("Weights and bias must be initialized before forward pass")
        self.z = np.dot(inputs, self.weights) + self.bias
        self.output = self.activation.forward(self.z)
        return self.output
    
    def backward(self, previous_layer_activations: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        self.bias_gradient = np.sum(gradient, axis=0, keepdims=True)
        self.weights_gradient = np.dot(previous_layer_activations.T, gradient)
        if hasattr(self, 'l1_lambda') and self.l1_lambda > 0:
            l1_grad_scale = self.l1_lambda / self.weights.size
            epsilon = 1e-8
            l1_grad = l1_grad_scale * self.weights / (np.abs(self.weights) + epsilon)
            l1_grad = np.clip(l1_grad, -1.0, 1.0)
            self.weights_gradient += l1_grad
        if hasattr(self, 'l2_lambda') and self.l2_lambda > 0:
            l2_grad_scale = self.l2_lambda / self.weights.size
            self.weights_gradient += l2_grad_scale * self.weights
        return np.dot(gradient, self.weights.T)
    
    def update_weights(self, learning_rate: float) -> None:
        if self.weights_gradient is None or self.bias_gradient is None:
            raise ValueError("Gradients must be computed before updating weights")
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * self.bias_gradient
        
class BatchNormalizationLayer(Layer):
    """Layer that performs batch normalization."""
    
    def __init__(self, input_size: int, momentum: float = 0.99, epsilon: float = 1e-5):
        """Initialize batch normalization layer."""
        self.input_size = input_size
        self.output_size = input_size
        self.gamma = np.ones((1, input_size))  # Scale parameter
        self.beta = np.zeros((1, input_size))  # Shift parameter
        self.epsilon = epsilon  # Small constant for numerical stability
        self.momentum = momentum  # Momentum for running statistics
        
        # Running statistics for inference
        self.running_mean = np.zeros((1, input_size))
        self.running_var = np.ones((1, input_size))
        
        # For backward pass
        self.x = None
        self.x_normalized = None
        self.batch_mean = None
        self.batch_var = None
        self.std = None
        
        # For optimization
        self.weights = self.gamma
        self.bias = self.beta
        self.weights_gradient = None
        self.bias_gradient = None
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with batch normalization."""
        # Store input for backward pass
        self.x = inputs
        
        if training:
            # Calculate batch statistics
            self.batch_mean = np.mean(inputs, axis=0, keepdims=True)
            self.batch_var = np.var(inputs, axis=0, keepdims=True)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
            
            # Normalize
            self.std = np.sqrt(self.batch_var + self.epsilon)
            self.x_normalized = (inputs - self.batch_mean) / self.std
            
            # Scale and shift
            output = self.gamma * self.x_normalized + self.beta
        else:
            # Use running statistics for inference
            x_normalized = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            output = self.gamma * x_normalized + self.beta
        
        return output
    
    def backward(self, previous_layer_activations: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Backward pass for batch normalization."""
        batch_size = self.x.shape[0]
        
        # Gradients for gamma and beta
        self.weights_gradient = np.sum(gradient * self.x_normalized, axis=0, keepdims=True)
        self.bias_gradient = np.sum(gradient, axis=0, keepdims=True)
        
        # Gradient for normalized input
        dx_normalized = gradient * self.gamma
        
        # Gradient for batch variance
        dvar = np.sum(dx_normalized * (self.x - self.batch_mean) * -0.5 * 
                      np.power(self.batch_var + self.epsilon, -1.5), axis=0, keepdims=True)
        
        # Gradient for batch mean
        dmean = np.sum(dx_normalized * -1.0 / self.std, axis=0, keepdims=True) + \
                dvar * np.mean(-2.0 * (self.x - self.batch_mean), axis=0, keepdims=True)
        
        # Gradient for input
        dx = dx_normalized / self.std + \
             dvar * 2.0 * (self.x - self.batch_mean) / batch_size + \
             dmean / batch_size
        
        return dx
    
    def update_weights(self, learning_rate: float) -> None:
        """Update gamma and beta parameters."""
        if self.weights_gradient is None or self.bias_gradient is None:
            return
            
        self.gamma -= learning_rate * self.weights_gradient
        self.beta -= learning_rate * self.bias_gradient
    
class RMSNormalizationLayer(Layer):
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
        self.z = None
        self.output = None
        self.activation = Linear()
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.z = inputs
        self.x = inputs
        ms = np.mean(np.square(inputs), axis=1, keepdims=True)
        self.rms = np.sqrt(ms + self.epsilon)
        self.rms = np.maximum(self.rms, 1e-6)
        self.x_normalized = inputs / self.rms
        self.output = self.gamma * self.x_normalized
        return self.output
    
    def backward(self, previous_layer_activations: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
        self.weights_gradient = np.sum(gradient * self.x_normalized, axis=0, keepdims=True)
        self.bias_gradient = np.zeros_like(self.bias)
        d_normalized = gradient * self.gamma
        d_normalized = np.clip(d_normalized, -1.0, 1.0)
        batch_size = gradient.shape[0]
        d_rms = -np.sum(d_normalized * self.x_normalized, axis=1, keepdims=True) / self.rms
        d_ms = d_rms * 0.5 / self.rms
        d_x = d_normalized / self.rms
        d_x += 2.0 * self.x * d_ms / self.input_size
        d_x = np.clip(d_x, -10.0, 10.0)
        return d_x
    
    def initialize_weights(self, method: str, **params):
        pass
