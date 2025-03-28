import numpy as np

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
    """Softmax activation function."""
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        shifted_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values = np.exp(shifted_inputs)
        sum_exp = np.sum(exp_values, axis=1, keepdims=True)
        self.output = exp_values / sum_exp
        self.output = np.nan_to_num(self.output, nan=1e-8, posinf=1.0, neginf=0.0)
        
        return self.output
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        # Backward pass for softmax.
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
        self.inputs = inputs
        result = inputs.copy()
        mask = inputs <= 0
        safe_neg_inputs = np.clip(inputs[mask], -30.0, 0)
        result[mask] = self.alpha * (np.exp(safe_neg_inputs) - 1)
        
        return result
    
    def backward(self, inputs: np.ndarray) -> np.ndarray:
        # Derivative: 1 if x > 0 else α * e^x
        result = np.ones_like(self.inputs)
        mask = self.inputs <= 0
        safe_neg_inputs = np.clip(self.inputs[mask], -30.0, 0)
        result[mask] = self.alpha * np.exp(safe_neg_inputs)
        
        return result