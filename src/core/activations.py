import numpy as np
from typing import Union


class Activation:
    """Base class for activation functions"""
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Linear(Activation):
    """Linear activation function: f(x) = x"""
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return inputs
    
    def backward(self, inputs: np.ndarray) -> np.ndarray:
        # Derivative: 1
        return np.ones_like(inputs)


class ReLU(Activation):
    """Rectified Linear Unit activation function: f(x) = max(0, x)"""
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = np.maximum(0, inputs)
        return self.output
    
    def backward(self, inputs: np.ndarray) -> np.ndarray:
        # Derivative: 1 if x > 0 else 0
        return np.where(inputs > 0, 1, 0)


class Sigmoid(Activation):
    """Sigmoid activation function: f(x) = 1 / (1 + e^(-x))"""
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        inputs = np.clip(inputs, -500, 500)
        self.output = 1.0 / (1.0 + np.exp(-inputs))
        return self.output
    
    def backward(self, inputs: np.ndarray) -> np.ndarray:
        # Derivative: sigmoid(x) * (1 - sigmoid(x))
        sigmoid_output = self.forward(inputs)
        return sigmoid_output * (1 - sigmoid_output)


class Tanh(Activation):
    """Hyperbolic Tangent activation function: f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))"""
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = np.tanh(inputs)
        return self.output
    
    def backward(self, inputs: np.ndarray) -> np.ndarray:
        # Derivative: 1 - tanh^2(x)
        tanh_output = np.tanh(inputs)
        return 1 - tanh_output ** 2


class Softmax(Activation):
    """Softmax activation function: f(x_i) = e^(x_i) / sum(e^(x_j))"""
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass with improved numerical stability.
        """
        # Shift inputs for numerical stability (subtract max)
        shifted_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        
        # Calculate exponentials with shifted values
        exp_values = np.exp(shifted_inputs)
        
        # Normalize by sum
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        # Handle any remaining NaN values (rare but possible)
        self.output = np.nan_to_num(self.output, nan=1e-8, posinf=1.0, neginf=0.0)
        
        return self.output
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Backward pass for softmax.
        For categorical cross-entropy loss, this is handled specially.
        """
        return gradient

class Softplus(Activation):
    """Softplus activation function: f(x) = ln(1 + e^x)"""
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.log1p(np.exp(-np.abs(inputs))) + np.maximum(inputs, 0)
    
    def backward(self, inputs: np.ndarray) -> np.ndarray:
        # Derivative: sigmoid
        return 1 / (1 + np.exp(-inputs))


class ELU(Activation):
    """Exponential Linear Unit activation function: f(x) = x if x > 0 else α * (e^x - 1)"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Save original inputs for backward pass
        self.inputs = inputs
        
        # Apply ELU formula with safe exp for negative inputs
        result = inputs.copy()
        mask = inputs <= 0
        # Use np.exp with clipping to avoid overflow
        safe_neg_inputs = np.clip(inputs[mask], -30.0, 0)
        result[mask] = self.alpha * (np.exp(safe_neg_inputs) - 1)
        
        return result
    
    def backward(self, inputs: np.ndarray) -> np.ndarray:
        # Use the stored inputs from forward pass
        result = np.ones_like(self.inputs)
        mask = self.inputs <= 0
        
        # Use np.exp with clipping to avoid overflow
        safe_neg_inputs = np.clip(self.inputs[mask], -30.0, 0)
        result[mask] = self.alpha * np.exp(safe_neg_inputs)
        
        return result