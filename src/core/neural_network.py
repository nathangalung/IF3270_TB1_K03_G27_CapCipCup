import numpy as np
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.layers import Layer, DenseLayer, RMSNormalizationLayer
from core.activations import Activation, Linear, ReLU, Sigmoid, Tanh, Softmax, Softplus, ELU


class NeuralNetwork:
    def __init__(self, layer_sizes, activation_functions, weight_initialization, 
                use_batch_norm=False, dropout_rates=None, seed=None, **init_params):
        # Validate inputs
        if len(layer_sizes) < 2:
            raise ValueError("At least input and output layers are required")
        if len(activation_functions) != len(layer_sizes) - 1:
            raise ValueError("Number of activation functions must match number of layers - 1")
        
        # Store model configuration
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self.use_batch_norm = use_batch_norm
        self.dropout_rates = dropout_rates if dropout_rates else [0.0] * (len(layer_sizes) - 1)
        self.seed = seed
        
        # Adam optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m = {}
        self.v = {}
        self.t = 0
        
        # Check dropout rates
        if len(self.dropout_rates) != len(layer_sizes) - 1:
            raise ValueError("Number of dropout rates must match number of layers - 1")
        
        # Build network
        self.layers = self._create_layers(layer_sizes, activation_functions)
        self._initialize_weights(weight_initialization, seed=seed, **init_params)
    
    def _create_layers(self, layer_sizes, activation_functions):
        """Create network layers"""
        layers = []
        
        for i in range(len(layer_sizes) - 1):
            # Handle string activation names
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
                activation = activation_functions[i]
            
            # Create layer
            dense_layer = DenseLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i+1],
                activation=activation
            )
            layers.append(dense_layer)
        
        return layers
    
    def _initialize_weights(self, method, **params):
        """Initialize weights for all layers"""
        for layer in self.layers:
            layer.initialize_weights(method, **params)
    
    def forward_propagation(self, X, training=True):
        """Perform forward pass through network"""
        # Handle invalid input values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Forward through layers
        A = X
        for layer in self.layers:
            if hasattr(layer, 'training_mode'):
                A = layer.forward(A, training=training)
            else:
                A = layer.forward(A)
            
            # Handle numerical instability
            if np.any(np.isnan(A)) or np.any(np.isinf(A)):
                A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        
        return A

    def backward_propagation(self, X, y, gradient):
        """Perform backward pass through network"""
        # Handle invalid gradient values
        if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
            gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Backward through layers
        dA = gradient
        for i in range(len(self.layers) - 1, -1, -1):
            if i == 0:
                dA = self.layers[i].backward(X, dA)
            else:
                dA = self.layers[i].backward(self.layers[i-1].output, dA)
    
    def update_weights(self, learning_rate):
        """Update weights using Adam optimizer"""
        # Increment timestep
        self.t += 1
        
        # Initialize moment estimates if needed
        if not self.m:
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'weights') and layer.weights is not None:
                    self.m[f'w{i}'] = np.zeros_like(layer.weights)
                    self.m[f'b{i}'] = np.zeros_like(layer.bias)
                    self.v[f'w{i}'] = np.zeros_like(layer.weights)
                    self.v[f'b{i}'] = np.zeros_like(layer.bias)
        
        # Update weights with Adam
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights') and layer.weights_gradient is not None:
                # Get gradients
                dw = layer.weights_gradient
                db = layer.bias_gradient
                
                # Update momentum
                self.m[f'w{i}'] = self.beta1 * self.m[f'w{i}'] + (1 - self.beta1) * dw
                self.m[f'b{i}'] = self.beta1 * self.m[f'b{i}'] + (1 - self.beta1) * db
                
                # Update RMSprop
                self.v[f'w{i}'] = self.beta2 * self.v[f'w{i}'] + (1 - self.beta2) * (dw**2)
                self.v[f'b{i}'] = self.beta2 * self.v[f'b{i}'] + (1 - self.beta2) * (db**2)
                
                # Bias correction
                m_hat_w = self.m[f'w{i}'] / (1 - self.beta1**self.t)
                m_hat_b = self.m[f'b{i}'] / (1 - self.beta1**self.t)
                v_hat_w = self.v[f'w{i}'] / (1 - self.beta2**self.t)
                v_hat_b = self.v[f'b{i}'] / (1 - self.beta2**self.t)
                
                # Update parameters
                layer.weights -= learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                layer.bias -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
    
    def apply_gradient_clipping(self, clip_value=5.0):
        """Clip gradients to prevent exploding gradients"""
        # Calculate gradient norm
        total_norm = 0
        for layer in self.layers:
            if hasattr(layer, 'weights_gradient') and layer.weights_gradient is not None:
                total_norm += np.sum(np.square(layer.weights_gradient))
            if hasattr(layer, 'bias_gradient') and layer.bias_gradient is not None:
                total_norm += np.sum(np.square(layer.bias_gradient))
        
        total_norm = np.sqrt(total_norm)
        
        # Apply clipping if needed
        if total_norm > clip_value:
            scale = clip_value / (total_norm + 1e-7)
            for layer in self.layers:
                if hasattr(layer, 'weights_gradient') and layer.weights_gradient is not None:
                    layer.weights_gradient *= scale
                if hasattr(layer, 'bias_gradient') and layer.bias_gradient is not None:
                    layer.bias_gradient *= scale
    
    def train_batch(self, X_batch, y_batch, learning_rate, loss_function):
        """Train on a single batch"""
        try:
            # Forward pass
            y_pred = self.forward_propagation(X_batch, training=True)
            
            # Handle NaN predictions
            if np.any(np.isnan(y_pred)):
                y_pred = np.nan_to_num(y_pred, nan=1e-8)
            
            # Calculate loss
            loss_value = loss_function.loss(y_batch, y_pred)
            
            # Add regularization
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
            
            # Normalize regularization loss
            if total_weights > 0:
                reg_loss = reg_loss / total_weights
                
            loss_value += reg_loss
            
            # Calculate gradient
            loss_gradient = loss_function.gradient(y_batch, y_pred)
            
            # Handle NaN gradient
            if np.any(np.isnan(loss_gradient)):
                loss_gradient = np.nan_to_num(loss_gradient, nan=0.0)
            
            # Backward pass
            self.backward_propagation(X_batch, y_batch, loss_gradient)
            
            # Clip gradients
            self.apply_gradient_clipping(clip_value=1.0)
            
            # Update weights
            self.update_weights(learning_rate)
            
            return loss_value if not np.isnan(loss_value) else 0.0
        except Exception as e:
            print(f"Error in train_batch: {e}")
            return 0.0
    
    def train(self, X_train, y_train, learning_rate, loss_function,
              X_val=None, y_val=None, batch_size=32, epochs=10, verbose=1):
        """Train the model"""
        # Initialize history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Calculate batches
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Batch training
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                batch_loss = self.train_batch(X_batch, y_batch, learning_rate, loss_function)
                epoch_loss += batch_loss
                
                # Show progress
                if verbose == 1:
                    progress = (batch + 1) / n_batches
                    progress_bar = '#' * int(progress * 20) + '-' * (20 - int(progress * 20))
                    print(f"\rEpoch {epoch+1}/{epochs} [{progress_bar}] {progress*100:.1f}% - batch_loss: {batch_loss:.4f}", end="")
            
            # Average loss
            epoch_loss /= n_batches
            history['train_loss'].append(epoch_loss)
            
            # Validation
            val_loss = None
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward_propagation(X_val, training=False)
                val_loss = loss_function.loss(y_val, y_val_pred)
                history['val_loss'].append(val_loss)
            
            # Print epoch results
            if verbose == 1:
                if val_loss is not None:
                    print(f"\rEpoch {epoch+1}/{epochs} - train_loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")
                else:
                    print(f"\rEpoch {epoch+1}/{epochs} - train_loss: {epoch_loss:.4f}")
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.forward_propagation(X, training=False)
    
    def display_model(self):
        """Print model summary"""
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
            print(f"  Parameters: {layer.weights.size + layer.bias.size}")
            print()
        
        total_params = sum(layer.weights.size + layer.bias.size for layer in self.layers)
        print(f"Total parameters: {total_params}")
    
    def save(self, filename):
        """Save model to file"""
        # Create directory if needed
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Prepare model data
        model_data = {
            'layer_sizes': self.layer_sizes,
            'layers': self.layers,
            'activation_functions': [layer.activation for layer in self.layers]
        }
        
        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
    
    @classmethod
    def load(cls, filename):
        """Load model from file"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create model instance
        nn = cls(
            layer_sizes=model_data['layer_sizes'],
            activation_functions=[None] * (len(model_data['layer_sizes']) - 1),
            weight_initialization='zero'
        )
        
        # Restore layers
        nn.layers = model_data['layers']
        
        print(f"Model loaded from {filename}")
        return nn
    
    def summary(self):
        """Print a summary of the neural network architecture"""
        total_params = 0
        trainable_params = 0
        
        print("\nNeural Network Summary:")
        print("=" * 80)
        print(f"{'Layer (type)':<30}{'Output Shape':<20}{'Param #':<15}{'Trainable':<10}")
        print("=" * 80)
        
        # Input layer
        if hasattr(self, 'input_size'):
            print(f"{'Input':<30}{(None, self.input_size)!s:<20}{'0':<15}{'N/A':<10}")
        
        # Hidden layers and output layer
        for i, layer in enumerate(self.layers):
            layer_name = f"{type(layer).__name__} ({i})"
            
            # Get output shape
            output_shape = "Unknown"
            if hasattr(layer, 'output_size'):
                output_shape = (None, layer.output_size)
            
            # Calculate number of parameters
            params = 0
            trainable = "Yes"
            
            if hasattr(layer, 'weights') and layer.weights is not None:
                weights_params = layer.weights.size
                bias_params = layer.bias.size if hasattr(layer, 'bias') and layer.bias is not None else 0
                params = weights_params + bias_params
                trainable_params += params
            else:
                trainable = "No"
            
            total_params += params
            
            # Print layer info
            print(f"{layer_name:<30}{output_shape!s:<20}{params:<15}{trainable:<10}")
        
        print("=" * 80)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")
        print("=" * 80)
        
        return None