import numpy as np
import pickle
from typing import List, Callable, Dict, Tuple, Union, Optional, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.layers import Layer, DenseLayer, RMSNormalizationLayer
from core.activations import Activation, Linear, ReLU, Sigmoid, Tanh, Softmax, Softplus, ELU


class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], activation_functions: List[Callable], 
                weight_initialization: str,
                use_batch_norm: bool = False,
                dropout_rates: Optional[List[float]] = None,
                seed: Optional[int] = None,
                **init_params):
        if len(layer_sizes) < 2:
            raise ValueError("At least input and output layers are required")
        if len(activation_functions) != len(layer_sizes) - 1:
            raise ValueError("Number of activation functions must match number of layers - 1")
        
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self.use_batch_norm = use_batch_norm
        self.dropout_rates = dropout_rates if dropout_rates else [0.0] * (len(layer_sizes) - 1)
        
        self.seed = seed
        self.beta1 = 0.9  # Momentum parameter
        self.beta2 = 0.999  # RMSprop parameter
        self.epsilon = 1e-8  # Small constant for numerical stability
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.t = 0  # Timestep
        
        # Make sure dropout_rates has the correct length
        if len(self.dropout_rates) != len(layer_sizes) - 1:
            raise ValueError("Number of dropout rates must match number of layers - 1")
        
        # Create the layers
        self.layers = self._create_layers(layer_sizes, activation_functions)
        
        # Initialize weights using the specified method and parameters
        self._initialize_weights(weight_initialization, seed=seed, **init_params)
    
    def _create_layers(self, layer_sizes: List[int], 
                  activation_functions: List[Callable]) -> List[Layer]:
        layers = []
        
        for i in range(len(layer_sizes) - 1):
            if isinstance(activation_functions[i], str):
                act_name = activation_functions[i].lower()
                if act_name == 'linear':
                    activation = Linear()
                elif act_name == 'relu':
                    activation = ReLU()
                elif act_name == 'sigmoid':
                    activation = Sigmoid()
                elif act_name == 'tanh':
                    activation = Tanh()
                elif act_name == 'softmax':
                    activation = Softmax()
                elif act_name == 'softplus':
                    activation = Softplus()
                elif act_name == 'elu':
                    activation = ELU()
                else:
                    raise ValueError(f"Unknown activation function: {activation_functions[i]}")
            else:
                activation = self.activation_functions[i]
            
            # Add dense layer
            dense_layer = DenseLayer(
                input_size=self.layer_sizes[i],
                output_size=self.layer_sizes[i+1],
                activation=activation
            )
            layers.append(dense_layer)
        
        return layers
    
    def _initialize_weights(self, method: str, **params):
        for layer in self.layers:
            layer.initialize_weights(method, **params)
    
    def forward_propagation(self, X, training=True):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            training: Whether the model is in training mode (for BN and Dropout)
        
        Returns:
            Output of the final layer
        """
        # Input validation
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        A = X
        for layer in self.layers:
            if hasattr(layer, 'training_mode'): 
                # For layers that support training/inference modes
                A = layer.forward(A, training=training)
            else:
                A = layer.forward(A)
            
            # Check for NaN values in activations
            if np.any(np.isnan(A)) or np.any(np.isinf(A)):
                # Replace NaN and Inf with zeros
                A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        
        return A

    def backward_propagation(self, X, y, gradient):
        """
        Backward propagation through all layers.
        
        Args:
            X: Input data
            y: Target values
            gradient: Initial gradient from loss function
        """
        # Input validation
        if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
            gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check for NaN values in gradient and replace with zeros
        dA = gradient
        for i in range(len(self.layers) - 1, -1, -1):
            if i == 0:
                dA = self.layers[i].backward(X, dA)
            else:
                dA = self.layers[i].backward(self.layers[i-1].output, dA)
    
    def update_weights(self, learning_rate):
        """
        Update weights using Adam optimizer.
        
        Parameters:
            learning_rate: Learning rate for weight updates
        """
        # Increment timestep
        self.t += 1
        
        # Initialize moment estimates if not already done
        if not self.m:
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'weights') and layer.weights is not None:
                    self.m[f'w{i}'] = np.zeros_like(layer.weights)
                    self.m[f'b{i}'] = np.zeros_like(layer.bias)
                    self.v[f'w{i}'] = np.zeros_like(layer.weights)
                    self.v[f'b{i}'] = np.zeros_like(layer.bias)
        
        # Update weights and biases using Adam
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights') and layer.weights_gradient is not None:
                # Get gradients
                dw = layer.weights_gradient
                db = layer.bias_gradient
                
                # Update first moment estimate (momentum)
                self.m[f'w{i}'] = self.beta1 * self.m[f'w{i}'] + (1 - self.beta1) * dw
                self.m[f'b{i}'] = self.beta1 * self.m[f'b{i}'] + (1 - self.beta1) * db
                
                # Update second moment estimate (RMSprop)
                self.v[f'w{i}'] = self.beta2 * self.v[f'w{i}'] + (1 - self.beta2) * (dw**2)
                self.v[f'b{i}'] = self.beta2 * self.v[f'b{i}'] + (1 - self.beta2) * (db**2)
                
                # Bias correction
                m_hat_w = self.m[f'w{i}'] / (1 - self.beta1**self.t)
                m_hat_b = self.m[f'b{i}'] / (1 - self.beta1**self.t)
                v_hat_w = self.v[f'w{i}'] / (1 - self.beta2**self.t)
                v_hat_b = self.v[f'b{i}'] / (1 - self.beta2**self.t)
                
                # Update weights and biases
                layer.weights -= learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                layer.bias -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
    def apply_gradient_clipping(self, clip_value=5.0):
        """Apply gradient clipping to prevent exploding gradients."""
        # Calculate total gradient norm
        total_norm = 0
        for layer in self.layers:
            if hasattr(layer, 'weights_gradient') and layer.weights_gradient is not None:
                total_norm += np.sum(np.square(layer.weights_gradient))
            if hasattr(layer, 'bias_gradient') and layer.bias_gradient is not None:
                total_norm += np.sum(np.square(layer.bias_gradient))
        
        total_norm = np.sqrt(total_norm)
        
        # Apply clipping if the norm exceeds the threshold
        if total_norm > clip_value:
            scale = clip_value / (total_norm + 1e-7)
            for layer in self.layers:
                if hasattr(layer, 'weights_gradient') and layer.weights_gradient is not None:
                    layer.weights_gradient *= scale
                if hasattr(layer, 'bias_gradient') and layer.bias_gradient is not None:
                    layer.bias_gradient *= scale
    
    def train_batch(self, X_batch, y_batch, learning_rate, loss_function):
        """
        Train model on a single batch.
        
        Parameters:
        -----------
        X_batch: np.ndarray
            Batch of input data
        y_batch: np.ndarray
            Batch of target data
        learning_rate: float
            Learning rate for weight updates
        loss_function: callable
            Loss function to use
            
        Returns:
        --------
        float: Batch loss value
        """
        try:
            # Forward pass with training=True
            y_pred = self.forward_propagation(X_batch, training=True)
            
            # Check for NaN values in prediction
            if np.any(np.isnan(y_pred)):
                print("Warning: NaN values detected in forward pass")
                # Replace NaN with small values
                y_pred = np.nan_to_num(y_pred, nan=1e-8)
            
            # Calculate loss
            loss_value = loss_function.loss(y_batch, y_pred)
            
            # Add regularization terms to loss if applicable
            reg_loss = 0
            total_weights = 0
            
            for layer in self.layers:
                if hasattr(layer, 'weights'):
                    layer_weights = layer.weights.size
                    total_weights += layer_weights
                    
                    if hasattr(layer, 'l1_lambda') and layer.l1_lambda > 0:
                        reg_loss += layer.l1_lambda * np.sum(np.abs(layer.weights))
                    if hasattr(layer, 'l2_lambda') and layer.l2_lambda > 0:
                        reg_loss += 0.5 * layer.l2_lambda * np.sum(np.square(layer.weights))
            
            # Normalize by total number of weights
            if total_weights > 0:
                reg_loss = reg_loss / total_weights
                
            loss_value += reg_loss
            
            # Calculate initial gradient
            loss_gradient = loss_function.gradient(y_batch, y_pred)
            
            # Check for NaN in loss gradient
            if np.any(np.isnan(loss_gradient)):
                print("Warning: NaN values detected in loss gradient")
                # Replace NaN with zeros
                loss_gradient = np.nan_to_num(loss_gradient, nan=0.0)
            
            # Backward pass
            self.backward_propagation(X_batch, y_batch, loss_gradient)
            
            # Apply gradient clipping with a stricter threshold for L1 regularization
            self.apply_gradient_clipping(clip_value=1.0)
            
            # Update weights
            self.update_weights(learning_rate)
            
            # Final NaN check on loss
            if np.isnan(loss_value):
                return 0.0
                
            return loss_value
        except Exception as e:
            print(f"Error in train_batch: {e}")
            return 0.0
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              learning_rate: float, 
              loss_function: Callable,
              X_val: Optional[np.ndarray], 
              y_val: Optional[np.ndarray],
              batch_size: int = 1,
              epochs: int = 5, 
              verbose: int = 1) -> Dict[str, List[float]]:
        
        training_history = {
            'train_loss': [],
            'val_loss': []
        }
        
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                batch_loss = self.train_batch(X_batch, y_batch, learning_rate, loss_function)
                epoch_loss += batch_loss
                
                if verbose == 1:
                    progress = (batch + 1) / n_batches
                    progress_bar = '#' * int(progress * 20) + '-' * (20 - int(progress * 20))
                    print(f"\rEpoch {epoch+1}/{epochs} [{progress_bar}] {progress*100:.1f}% - batch_loss: {batch_loss:.4f}", end="")
            

            epoch_loss /= n_batches
            training_history['train_loss'].append(epoch_loss)
            

            val_loss = None
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward_propagation(X_val)
                val_loss = loss_function.loss(y_val, y_val_pred)
                training_history['val_loss'].append(val_loss)
            

            if verbose == 1:
                if val_loss is not None:
                    print(f"\rEpoch {epoch+1}/{epochs} - train_loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")
                else:
                    print(f"\rEpoch {epoch+1}/{epochs} - train_loss: {epoch_loss:.4f}")
        
        return training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on data.
        
        Parameters:
        -----------
        X: np.ndarray
            Input data
            
        Returns:
        --------
        np.ndarray: Predictions
        """
        return self.forward_propagation(X, training=False)
    
    def display_model(self) -> None:
        print("Neural Network Structure:")
        print("-------------------------")
        print(f"Total layers: {len(self.layers) + 1} (including input layer)")
        print(f"Layer sizes: {self.layer_sizes}")
        print()
        
        print("Layer Details:")
        print("-------------")
        for i, layer in enumerate(self.layers):
            print(f"Layer {i+1}: {self.layer_sizes[i]} → {self.layer_sizes[i+1]}")
            print(f"  Activation: {layer.activation.__class__.__name__}")
            print(f"  Weights shape: {layer.weights.shape}")
            print(f"  Bias shape: {layer.bias.shape}")
            print(f"  Parameter count: {layer.weights.size + layer.bias.size}")
            
            if hasattr(layer, 'weights_gradient') and layer.weights_gradient is not None:
                print(f"  Weight gradients: Min={layer.weights_gradient.min():.6f}, Max={layer.weights_gradient.max():.6f}")
            
            if hasattr(layer, 'bias_gradient') and layer.bias_gradient is not None:
                print(f"  Bias gradients: Min={layer.bias_gradient.min():.6f}, Max={layer.bias_gradient.max():.6f}")
            print()
        
        total_params = sum(layer.weights.size + layer.bias.size for layer in self.layers)
        print(f"Total parameters: {total_params}")
    
    def save(self, filename: str) -> None:
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        model_data = {
            'layer_sizes': self.layer_sizes,
            'layers': self.layers,
            'activation_functions': [layer.activation for layer in self.layers]
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
    
    @classmethod
    def load(cls, filename: str) -> 'NeuralNetwork':
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        nn = cls(model_data['layer_sizes'], [None] * (len(model_data['layer_sizes']) - 1))
        
        nn.layers = model_data['layers']
        
        print(f"Model loaded from {filename}")
        return nn