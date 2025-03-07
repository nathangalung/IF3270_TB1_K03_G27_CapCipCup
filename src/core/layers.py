import numpy as np
from typing import Callable, Dict, Optional, Union, Any
from .activations import Activation


class Layer:
    """Base layer interface for neural network"""
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def backward(self, inputs: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def update_weights(self, learning_rate: float) -> None:
        raise NotImplementedError


class DenseLayer(Layer):
    """Fully connected (dense) layer implementation"""
    
    def __init__(self, input_size: int, output_size: int, activation: Activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Initialize weights and biases to None
        self.weights = None
        self.bias = None
        self.weights_gradient = None
        self.bias_gradient = None
        
        # Cache for backward pass
        self.output = None
        self.z = None
    
    def initialize_weights(self, method: str, **params):
        seed = params.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
        
        # if method == 'zero':
            
            
        # elif method == 'random_uniform':
            
            
        # elif method == 'random_normal':
            
            
        # elif method == 'xavier':
            
            
        # elif method == 'he':
            
            
        else:
            raise ValueError(f"Unknown weight initialization method: {method}")
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise ValueError("Weights and bias must be initialized before forward pass")
            
        # Linear transformation: Z = X·W + b
        self.z = np.dot(inputs, self.weights) + self.bias
        
        # Apply activation function
        self.output = self.activation.forward(self.z)
        
        return self.output
    
    def backward(self, inputs: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        # Gradient through activation function
        dZ = self.activation.backward(self.z) * gradient
        
        # Gradient of weights: dW = X^T · dZ
        batch_size = inputs.shape[0]
        self.weights_gradient = np.dot(inputs.T, dZ) / batch_size
        
        # Gradient of bias: db = mean(dZ)
        self.bias_gradient = np.mean(dZ, axis=0, keepdims=True)
        
        # Gradient to pass to previous layer: dX = dZ · W^T
        dX = np.dot(dZ, self.weights.T)
        
        return dX
    
    def update_weights(self, learning_rate: float) -> None:
        if self.weights_gradient is None or self.bias_gradient is None:
            raise ValueError("Gradients must be computed before updating weights")
        
        # Update weights: W = W - lr * dW
        self.weights -= learning_rate * self.weights_gradient
        
        # Update bias: b = b - lr * db
        self.bias -= learning_rate * self.bias_gradient