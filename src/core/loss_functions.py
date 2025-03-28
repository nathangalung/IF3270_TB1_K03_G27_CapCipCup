import numpy as np

class Loss:
    """Base class for loss functions"""
    
    def loss(self, y_true, y_pred):
        """Calculate loss value"""
        raise NotImplementedError
    
    def gradient(self, y_true, y_pred):
        """Calculate gradient of loss with respect to predictions"""
        raise NotImplementedError


class MeanSquaredError(Loss):
    """Mean Squared Error loss function"""
    
    def loss(self, y_true, y_pred):
        """Compute MSE: (1/n) * sum((y_true - y_pred)^2)"""
        return np.mean(np.square(y_true - y_pred))
    
    def gradient(self, y_true, y_pred):
        """Gradient of MSE: -2(y_true - y_pred)/n"""
        batch_size = y_true.shape[0]
        return -2 * (y_true - y_pred) / batch_size


class BinaryCrossEntropy(Loss):
    """Binary Cross-Entropy loss function"""
    
    def __init__(self, epsilon=1e-15):
        """Initialize with small epsilon to prevent log(0)"""
        self.epsilon = epsilon
    
    def loss(self, y_true, y_pred):
        """Compute BCE: -mean(y_true*log(y_pred) + (1-y_true)*log(1-y_pred))"""
        # Clip predictions for numerical stability
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Calculate loss
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def gradient(self, y_true, y_pred):
        """Gradient of BCE loss"""
        # Clip predictions for numerical stability
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Calculate gradient
        batch_size = y_true.shape[0]
        return (y_pred - y_true) / (batch_size * y_pred * (1 - y_pred))


class CategoricalCrossEntropy(Loss):
    """Categorical Cross-Entropy loss function"""
    
    def __init__(self, epsilon=1e-15):
        """Initialize with small epsilon to prevent log(0)"""
        self.epsilon = epsilon
    
    def loss(self, y_true, y_pred):
        """Compute CCE for one-hot encoded or class index labels"""
        # Clip predictions for numerical stability
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        n_samples = y_true.shape[0]
        
        # Handle both index labels and one-hot encoded labels
        if len(y_true.shape) == 1:
            # For class indices
            return -np.sum(np.log(y_pred[np.arange(n_samples), y_true])) / n_samples
        else:
            # For one-hot encoded
            return -np.sum(y_true * np.log(y_pred)) / n_samples
    
    def gradient(self, y_true, y_pred):
        """Gradient of CCE loss: (y_pred - y_true_one_hot)/n"""
        # Convert to one-hot if needed
        if len(y_true.shape) == 1:
            n_samples = len(y_true)
            n_classes = y_pred.shape[1]
            y_true_one_hot = np.zeros((n_samples, n_classes))
            y_true_one_hot[np.arange(n_samples), y_true] = 1
        else:
            y_true_one_hot = y_true
            n_samples = y_true.shape[0]
        
        # When using softmax outputs, gradient simplifies to (y_pred - y_true)
        return (y_pred - y_true_one_hot) / n_samples