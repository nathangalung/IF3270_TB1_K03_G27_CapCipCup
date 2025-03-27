import numpy as np
from typing import Tuple, Optional, Dict, Any

def zero_init(shape: Tuple[int, ...], **kwargs) -> np.ndarray:
    """Inisialisasi bobot dengan nol."""
    return np.zeros(shape, dtype=np.float32)

def random_uniform_init(shape: Tuple[int, ...], lower: float = -0.1, upper: float = 0.1, 
                        seed: Optional[int] = None, **kwargs) -> np.ndarray:
    """Inisialisasi bobot dengan distribusi uniform."""
    rng = np.random.default_rng(seed)
    return rng.uniform(lower, upper, size=shape).astype(np.float32)

def random_normal_init(shape: Tuple[int, ...], mean: float = 0.0, std: float = 0.1, 
                      seed: Optional[int] = None, **kwargs) -> np.ndarray:
    """Inisialisasi bobot dengan distribusi normal."""
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, size=shape).astype(np.float32)

def xavier_init(shape: Tuple[int, ...], input_size: int, output_size: int, 
               seed: Optional[int] = None, **kwargs) -> np.ndarray:
    """Inisialisasi bobot dengan metode Xavier/Glorot."""
    rng = np.random.default_rng(seed)
    limit = np.sqrt(6 / (input_size + output_size))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)

def he_init(shape: Tuple[int, ...], input_size: int, 
           seed: Optional[int] = None, **kwargs) -> np.ndarray:
    """Inisialisasi bobot dengan metode He."""
    rng = np.random.default_rng(seed)
    std = np.sqrt(2 / input_size)
    return rng.normal(0, std, size=shape).astype(np.float32)

def initialize_weights(method: str, shape: Tuple[int, ...], **params) -> np.ndarray:
    """
    Fungsi utama untuk menginisialisasi bobot dengan metode yang dipilih.
    
    Parameters:
    -----------
    method: str
        Metode inisialisasi ('zero', 'random_uniform', 'random_normal', 'xavier', 'he')
    shape: Tuple[int, ...]
        Bentuk tensor yang akan diinisialisasi
    params: dict
        Parameter tambahan untuk metode inisialisasi
    
    Returns:
    --------
    np.ndarray
        Bobot yang telah diinisialisasi
    """
    methods = {
        "zero": zero_init,
        "random_uniform": random_uniform_init,
        "random_normal": random_normal_init,
        "xavier": xavier_init,
        "he": he_init,
    }
    
    if method not in methods:
        raise ValueError(f"Metode inisialisasi tidak dikenali: {method}")
    
    if method in {"xavier", "he"}:
        if "input_size" not in params:
            raise ValueError(f"Metode {method} memerlukan parameter 'input_size'")
        if method == "xavier" and "output_size" not in params:
            raise ValueError("Metode xavier memerlukan parameter 'output_size'")
    
    filtered_params = _filter_params_for_method(method, params)
    
    return methods[method](shape, **filtered_params)

def _filter_params_for_method(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter parameter berdasarkan metode inisialisasi yang digunakan.
    
    Parameters:
    -----------
    method: str
        Metode inisialisasi
    params: Dict[str, Any]
        Parameter asli
    
    Returns:
    --------
    Dict[str, Any]
        Parameter yang telah difilter
    """
    required_params = {
        "zero": [],
        "random_uniform": ["lower", "upper", "seed"],
        "random_normal": ["mean", "std", "seed"],
        "xavier": ["input_size", "output_size", "seed"],
        "he": ["input_size", "seed"]
    }
    
    param_list = required_params.get(method, [])
    
    if not param_list:
        return params
    
    filtered = {}
    for key, value in params.items():
        if key in param_list:
            filtered[key] = value
    
    return filtered