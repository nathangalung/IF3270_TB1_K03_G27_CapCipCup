import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Callable, Tuple, Optional, Any
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.neural_network import NeuralNetwork


class Trainer:
    """Helper class for training neural networks"""
    
    def __init__(self, 
                 model: NeuralNetwork,
                 learning_rate: float,
                 loss_function: Callable,
                 batch_size: int,
                 epochs: int,
                 verbose: int):
        """
        Initialize trainer
        
        Parameters:
        -----------
        model: NeuralNetwork
            The neural network to train
        learning_rate: float
            Learning rate for gradient descent
        loss_function: Callable
            Loss function to use
        batch_size: int
            Batch size for training
        epochs: int
            Number of epochs to train for
        verbose: int
            Verbosity level (0: silent, 1: show progress)
        """
        self.model = model
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.history = None
    
    def train(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            callbacks: List[Callable] = None) -> Dict[str, List[float]]:
        """
        Train the model
        
        Parameters:
        -----------
        X_train: np.ndarray
            Training data
        y_train: np.ndarray
            Training labels
        X_val: np.ndarray, optional
            Validation data
        y_val: np.ndarray, optional
            Validation labels
        callbacks: List[Callable], optional
            Callbacks to run during training
            
        Returns:
        --------
        Dict[str, List[float]]: Training history
        """
        if callbacks is None:
            callbacks = []
        
        # Change this line to use the model's train method instead of recursively calling self.train
        self.history = self.model.train(
            X_train=X_train,
            y_train=y_train,
            learning_rate=self.learning_rate,
            loss_function=self.loss_function,
            X_val=X_val,
            y_val=y_val,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose
        )
        
        for callback in callbacks:
            callback(self.model, self.history)
        
        return self.history


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                    random_state: Optional[int] = None, 
                    stratify: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets with optional stratification
    
    Parameters:
    -----------
    X: np.ndarray
        Input data
    y: np.ndarray
        Target data
    test_size: float
        Proportion of data to use for testing
    random_state: int, optional
        Random seed for reproducibility
    stratify: np.ndarray, optional
        Labels to stratify the split by (ensures class distribution is preserved)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    
    if stratify is not None:
        unique_classes = np.unique(stratify)
        test_indices = []
        train_indices = []
        
        for cls in unique_classes:
            cls_indices = np.where(stratify == cls)[0]
            n_cls_samples = len(cls_indices)
            
            # Shuffle the indices for this class
            np.random.shuffle(cls_indices)
            
            # Calculate how many samples should go to test
            n_test_samples = int(n_cls_samples * test_size)
            
            # Split indices for this class into test and train
            test_indices.extend(cls_indices[:n_test_samples])
            train_indices.extend(cls_indices[n_test_samples:])
        
        # Double check test size is close to what was requested
        actual_test_size = len(test_indices) / n_samples
        if abs(actual_test_size - test_size) > 0.05:  # 5% tolerance
            print(f"Warning: Actual test size ({actual_test_size:.2f}) differs from requested ({test_size:.2f})")
    else:
        # Random split without stratification
        indices = np.random.permutation(n_samples)
        test_count = int(n_samples * test_size)
        
        test_indices = indices[:test_count]
        train_indices = indices[test_count:]
    
    # Create train and test sets
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test


class EarlyStopping:
    """
    Early stopping callback to prevent overfitting
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Initialize early stopping
        
        Parameters:
        -----------
        patience: int
            Number of epochs with no improvement to wait before stopping
        min_delta: float
            Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_loss = float('inf')
        self.stopped_epoch = 0
    
    def __call__(self, model: NeuralNetwork, history: Dict[str, List[float]]):
        """
        Check if training should stop
        
        Parameters:
        -----------
        model: NeuralNetwork
            The neural network being trained
        history: Dict[str, List[float]]
            Training history
            
        Returns:
        --------
        bool: True if training should stop, False otherwise
        """
        # Get the loss value to monitor (prefer val_loss if available, otherwise use train_loss)
        if 'val_loss' in history and len(history['val_loss']) > 0:
            current_loss = history['val_loss'][-1]
        elif 'train_loss' in history and len(history['train_loss']) > 0:
            current_loss = history['train_loss'][-1]
        else:
            # No loss to monitor, can't make a decision
            return False
        
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Use the number of epochs from the available key
                if 'train_loss' in history:
                    self.stopped_epoch = len(history['train_loss'])
                elif 'val_loss' in history:
                    self.stopped_epoch = len(history['val_loss'])
                else:
                    self.stopped_epoch = 0
                
                print(f"\nEarly stopping at epoch {self.stopped_epoch}")
                return True
        
        return False
    
class LearningRateScheduler:
    def __init__(self, initial_lr=0.01, decay_rate=0.95, decay_steps=1):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.current_lr = initial_lr
        self.epochs = 0
        
    def get_lr(self):
        self.current_lr = self.initial_lr * (self.decay_rate ** (self.epochs // self.decay_steps))
        return self.current_lr
        
    def step(self):
        self.epochs += 1
        return self.get_lr()