import numpy as np
from typing import Optional

class Activation:
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Sigmoid(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def backward(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        sigmoid_x = self.forward(x)
        return grad * sigmoid_x * (1 - sigmoid_x)

class DenseLayer:
    def __init__(self, input_size: int, output_size: int, activation: Activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        
        self.input_cache = None
        self.output_cache = None
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.input_cache = inputs
        z = np.dot(inputs, self.weights) + self.bias
        self.output_cache = self.activation.forward(z)
        return self.output_cache
    
    def backward(self, grad: np.ndarray, learning_rate: float) -> np.ndarray:
        dZ = self.activation.backward(self.output_cache, grad)
        dW = np.dot(self.input_cache.T, dZ) / self.input_cache.shape[0]
        db = np.sum(dZ, axis=0, keepdims=True) / self.input_cache.shape[0]
        
        grad_input = np.dot(dZ, self.weights.T)
        
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db
        
        return grad_input

class RMSNormalizationLayer:
    def __init__(self, input_size: int, epsilon: float = 1e-8):
        self.input_size = input_size
        self.epsilon = epsilon
        self.gamma = np.ones((1, input_size))
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        mean_square = np.mean(np.square(inputs), axis=1, keepdims=True)
        self.rms = np.sqrt(mean_square + self.epsilon)
        return self.gamma * (inputs / self.rms)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        d_gamma = np.sum(grad * (self.input_cache / self.rms), axis=0, keepdims=True)
        d_input = grad / self.rms
        return d_input
