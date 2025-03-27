import numpy as np
from typing import Union


class Loss:
    """Base class for loss functions"""
    
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the loss value
        
        Parameters:
        -----------
        y_true: np.ndarray
            True labels
        y_pred: np.ndarray
            Predicted labels
            
        Returns:
        --------
        float: Loss value
        """
        raise NotImplementedError
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the loss
        
        Parameters:
        -----------
        y_true: np.ndarray
            True labels
        y_pred: np.ndarray
            Predicted labels
            
        Returns:
        --------
        np.ndarray: Gradient of the loss with respect to y_pred
        """
        raise NotImplementedError


class MeanSquaredError(Loss):
    """Mean Squared Error: MSE = (1/n) * Σ(y_true - y_pred)²"""
    
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate MSE loss
        
        Parameters:
        -----------
        y_true: np.ndarray
            True labels (batch_size, output_size)
        y_pred: np.ndarray
            Predicted labels (batch_size, output_size)
            
        Returns:
        --------
        float: MSE loss value
        """
        return np.mean(np.square(y_true - y_pred))
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of MSE loss
        
        Parameters:
        -----------
        y_true: np.ndarray
            True labels (batch_size, output_size)
        y_pred: np.ndarray
            Predicted labels (batch_size, output_size)
            
        Returns:
        --------
        np.ndarray: Gradient = -2(y_true - y_pred)/n
        """
        batch_size = y_true.shape[0]
        return -2 * (y_true - y_pred) / batch_size


class BinaryCrossEntropy(Loss):
    """Binary Cross-Entropy: BCE = -(1/n) * Σ[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]"""
    
    def __init__(self, epsilon: float = 1e-15):
        """
        Initialize BCE loss
        
        Parameters:
        -----------
        epsilon: float
            Small constant to avoid log(0)
        """
        self.epsilon = epsilon
    
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate BCE loss
        
        Parameters:
        -----------
        y_true: np.ndarray
            True labels (batch_size, 1) or (batch_size,)
        y_pred: np.ndarray
            Predicted probabilities (batch_size, 1) or (batch_size,)
            
        Returns:
        --------
        float: BCE loss value
        """
        # Clip predicted values to avoid numerical instability
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Calculate binary cross-entropy
        bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        return bce
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of BCE loss
        
        Parameters:
        -----------
        y_true: np.ndarray
            True labels (batch_size, 1) or (batch_size,)
        y_pred: np.ndarray
            Predicted probabilities (batch_size, 1) or (batch_size,)
            
        Returns:
        --------
        np.ndarray: Gradient = (y_pred - y_true) / (y_pred * (1 - y_pred))
        """
        # Clip predicted values to avoid numerical instability
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Calculate gradient
        batch_size = y_true.shape[0]
        return (y_pred - y_true) / (batch_size * y_pred * (1 - y_pred))


class CategoricalCrossEntropy(Loss):
    """Categorical Cross-Entropy: CCE = -(1/n) * Σ[Σ(y_true * log(y_pred))]"""
    
    def __init__(self, epsilon: float = 1e-15):
        """
        Initialize CCE loss
        
        Parameters:
        -----------
        epsilon: float
            Small constant to avoid log(0)
        """
        self.epsilon = epsilon
    
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate CCE loss
        
        Parameters:
        -----------
        y_true: np.ndarray
            True labels (one-hot encoded)
        y_pred: np.ndarray
            Predicted probabilities
            
        Returns:
        --------
        float: CCE loss value
        """
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # For numerical stability when using softmax outputs
        if len(y_true.shape) == 1:
            # If labels are provided as indices, convert to one-hot
            n_samples = len(y_true)
            loss_value = -np.sum(np.log(y_pred_clipped[np.arange(n_samples), y_true])) / n_samples
        else:
            # If labels are one-hot encoded
            n_samples = y_true.shape[0]
            loss_value = -np.sum(y_true * np.log(y_pred_clipped)) / n_samples
            
        return loss_value
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of CCE loss
        
        Parameters:
        -----------
        y_true: np.ndarray
            True labels (one-hot encoded)
        y_pred: np.ndarray
            Predicted probabilities
            
        Returns:
        --------
        np.ndarray: Gradient of CCE loss
        """
        # Convert integer labels to one-hot if needed
        if len(y_true.shape) == 1:
            n_samples = len(y_true)
            n_classes = y_pred.shape[1]
            y_true_one_hot = np.zeros((n_samples, n_classes))
            y_true_one_hot[np.arange(n_samples), y_true] = 1
        else:
            y_true_one_hot = y_true
            n_samples = y_true.shape[0]
        
        # For softmax + categorical cross-entropy, gradient is (y_pred - y_true)
        # This is more numerically stable than separate calculations
        gradient = y_pred - y_true_one_hot
        
        # Normalize by batch size
        gradient = gradient / n_samples
        
        return gradient