import numpy as np
from typing import Tuple, Dict, Any

def zero_init(shape: Tuple[int, ...], **kwargs):
    """Initialize weights with zeros"""
    return np.zeros(shape, dtype=np.float32)

def random_uniform_init(shape: Tuple[int, ...], lower=-0.1, upper=0.1, seed=None, **kwargs):
    """Initialize weights using uniform distribution"""
    rng = np.random.default_rng(seed)
    return rng.uniform(lower, upper, size=shape).astype(np.float32)

def random_normal_init(shape: Tuple[int, ...], mean=0.0, std=0.1, seed=None, **kwargs):
    """Initialize weights using normal distribution"""
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, size=shape).astype(np.float32)

def xavier_init(shape: Tuple[int, ...], input_size, output_size, seed=None, **kwargs):
    """Xavier/Glorot initialization"""
    rng = np.random.default_rng(seed)
    limit = np.sqrt(6 / (input_size + output_size))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)

def he_init(shape: Tuple[int, ...], input_size, seed=None, **kwargs):
    """He initialization"""
    rng = np.random.default_rng(seed)
    std = np.sqrt(2 / input_size)
    return rng.normal(0, std, size=shape).astype(np.float32)

def initialize_weights(method, shape, **params):
    """Initialize weights using specified method"""
    methods = {
        "zero": zero_init,
        "random_uniform": random_uniform_init,
        "random_normal": random_normal_init,
        "xavier": xavier_init,
        "he": he_init,
    }
    
    # Check if method exists
    if method not in methods:
        raise ValueError(f"Unknown initialization method: {method}")
    
    # Check required parameters
    if method == "xavier":
        if "input_size" not in params or "output_size" not in params:
            raise ValueError("Xavier initialization requires input_size and output_size")
    elif method == "he":
        if "input_size" not in params:
            raise ValueError("He initialization requires input_size")
    
    # Get only relevant parameters for the method
    filtered_params = _filter_params(method, params)
    
    return methods[method](shape, **filtered_params)

def _filter_params(method, params):
    """Filter parameters for specific initialization method"""
    # Parameters needed for each method
    param_map = {
        "zero": [],
        "random_uniform": ["lower", "upper", "seed"],
        "random_normal": ["mean", "std", "seed"],
        "xavier": ["input_size", "output_size", "seed"],
        "he": ["input_size", "seed"]
    }
    
    # Get list of parameters for this method
    needed_params = param_map.get(method, [])
    
    # Return all params if method not found
    if not needed_params:
        return params
    
    # Filter parameters
    result = {}
    for key, value in params.items():
        if key in needed_params:
            result[key] = value
    
    return result