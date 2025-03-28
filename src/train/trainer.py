import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.neural_network import NeuralNetwork


class Trainer:
    """Helper class for training neural networks"""
    
    def __init__(self, model, learning_rate, loss_function, batch_size=32, epochs=10, verbose=1):
        self.model = model
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.history = None
    
    def train(self, X_train, y_train, X_val=None, y_val=None, callbacks=None):
        """Train the model"""
        if callbacks is None:
            callbacks = []
        
        # Use model's train method
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
        
        # Run callbacks
        for callback in callbacks:
            callback(self.model, self.history)
        
        return self.history


def train_test_split(X, y, test_size=0.2, random_state=None, stratify=None):
    """Split data into train and test sets"""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    
    if stratify is not None:
        # Stratified split to maintain class proportions
        unique_classes = np.unique(stratify)
        test_indices = []
        train_indices = []
        
        for cls in unique_classes:
            cls_indices = np.where(stratify == cls)[0]
            n_cls_samples = len(cls_indices)
            
            # Shuffle indices for this class
            np.random.shuffle(cls_indices)
            
            # Split by test size ratio
            n_test_samples = int(n_cls_samples * test_size)
            test_indices.extend(cls_indices[:n_test_samples])
            train_indices.extend(cls_indices[n_test_samples:])
        
        # Check test size
        actual_test_size = len(test_indices) / n_samples
        if abs(actual_test_size - test_size) > 0.05:
            print(f"Warning: Actual test size ({actual_test_size:.2f}) differs from requested ({test_size:.2f})")
    else:
        # Random split
        indices = np.random.permutation(n_samples)
        test_count = int(n_samples * test_size)
        test_indices = indices[:test_count]
        train_indices = indices[test_count:]
    
    # Create train and test sets
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test


class EarlyStopping:
    """Early stopping callback to prevent overfitting"""
    
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_loss = float('inf')
        self.stopped_epoch = 0
    
    def __call__(self, model, history):
        """Check if training should stop"""
        # Get loss value (prefer validation loss)
        if 'val_loss' in history and history['val_loss']:
            current_loss = history['val_loss'][-1]
        elif 'train_loss' in history and history['train_loss']:
            current_loss = history['train_loss'][-1]
        else:
            return False
        
        # Check for improvement
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Get epoch count
                if 'train_loss' in history:
                    self.stopped_epoch = len(history['train_loss'])
                elif 'val_loss' in history:
                    self.stopped_epoch = len(history['val_loss'])
                else:
                    self.stopped_epoch = 0
                
                print(f"\nEarly stopping at epoch {self.stopped_epoch}")
                model.stop_training = True
                return True
        
        return False
    
    
class LearningRateScheduler:
    """Learning rate scheduler with exponential decay"""
    
    def __init__(self, initial_lr=0.01, decay_rate=0.95, decay_steps=1):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.current_lr = initial_lr
        self.epochs = 0
        
    def get_lr(self):
        """Get current learning rate"""
        self.current_lr = self.initial_lr * (self.decay_rate ** (self.epochs // self.decay_steps))
        return self.current_lr
        
    def step(self):
        """Increment epoch counter and return new learning rate"""
        self.epochs += 1
        return self.get_lr()
    
    def __call__(self, model, history):
        """Callback interface for use with Trainer"""
        self.epochs = len(history.get('train_loss', []))
        new_lr = self.get_lr()
        model.learning_rate = new_lr
        return new_lr