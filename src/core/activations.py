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
        self.output = np.where(
            inputs >= 0,
            1 / (1 + np.exp(-inputs)),
            np.exp(inputs) / (1 + np.exp(inputs))
        )
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
        shifted_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values = np.exp(shifted_inputs)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output
    
    def backward(self, inputs: np.ndarray) -> np.ndarray:
        # Derivative: Already calculated in the loss function CCE
        return np.ones_like(inputs)


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
        self.inputs = inputs
        return np.where(inputs > 0, inputs, self.alpha * (np.exp(inputs) - 1))
    
    def backward(self, inputs: np.ndarray) -> np.ndarray:
        # Derivative: 1 if x > 0 else α * e^x
        return np.where(inputs > 0, 1, self.alpha * np.exp(inputs))