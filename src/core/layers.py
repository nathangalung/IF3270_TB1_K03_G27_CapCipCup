import numpy as np
from typing import Callable, Dict, Optional, Union, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.activations import Activation, Linear
from core.weight_initializers import initialize_weights, random_uniform_init, random_normal_init

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
        init_params = {**params, 'input_size': self.input_size, 'output_size': self.output_size}
        self.weights = initialize_weights(method=method, shape=(self.input_size, self.output_size), **init_params)
        if method in ['zero', 'xavier', 'he']:
            self.bias = np.zeros((1, self.output_size))
        elif method == 'random_uniform':
            lower_bound = params.get('lower_bound', -0.1)
            upper_bound = params.get('upper_bound', 0.1)
            seed = params.get('seed', None)
            self.bias = random_uniform_init(shape=(1, self.output_size), lower_bound=lower_bound, upper_bound=upper_bound, seed=seed)
        elif method == 'random_normal':
            mean = params.get('mean', 0.0)
            var = params.get('var', 0.1)
            seed = params.get('seed', None)
            self.bias = random_normal_init(shape=(1, self.output_size), mean=mean, var=var, seed=seed)
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
