import numpy as np
import pickle
import os
from typing import List, Callable, Dict, Tuple, Union, Optional, Any

from .layers import Layer, DenseLayer
from .activations import Activation, Linear, ReLU, Sigmoid, Tanh, Softmax, Softplus, ELU


class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], activation_functions: List[Callable], 
                 weight_init_method: str = "random_normal", **init_params):
        if len(layer_sizes) < 2:
            raise ValueError("At least input and output layers are required")
        if len(activation_functions) != len(layer_sizes) - 1:
            raise ValueError("Number of activation functions must match number of layers - 1")
        
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self.layers = self._create_layers(layer_sizes, activation_functions)
        self._initialize_weights(weight_init_method, **init_params)
    
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
                activation = activation_functions[i]
            
            layer = DenseLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i+1],
                activation=activation
            )
            layers.append(layer)
        
        return layers
    
    def _initialize_weights(self, method: str, **params):
        for layer in self.layers:
            layer.initialize_weights(method, **params)
    
    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        current_input = X
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input
    
    def backward_propagation(self, X: np.ndarray, y: np.ndarray, 
                            loss_gradient: np.ndarray) -> None:
        gradient = loss_gradient
        
        for i in reversed(range(len(self.layers))):
            if i == 0:
                prev_output = X
            else:
                prev_output = self.layers[i-1].output
            
            gradient = self.layers[i].backward(prev_output, gradient)
    
    def update_weights(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.update_weights(learning_rate)
    
    def train_batch(self, X_batch: np.ndarray, y_batch: np.ndarray, 
                   loss_function: Callable, learning_rate: float) -> float:
        y_pred = self.forward_propagation(X_batch)
        
        loss_value = loss_function.loss(y_batch, y_pred)
        
        loss_gradient = loss_function.gradient(y_batch, y_pred)
        
        self.backward_propagation(X_batch, y_batch, loss_gradient)
        
        self.update_weights(learning_rate)
        
        return loss_value
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              loss_function: Callable,
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              batch_size: int = 32, 
              learning_rate: float = 0.01, 
              epochs: int = 100, 
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
                
                batch_loss = self.train_batch(X_batch, y_batch, loss_function, learning_rate)
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
        return self.forward_propagation(X)
    
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
    
    def plot_weight_distribution(self, layers: List[int] = None) -> None:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib and seaborn are required for plotting weight distributions.")
            return
        
        if layers is None:
            layers = list(range(len(self.layers)))
        
        fig, axs = plt.subplots(len(layers), 2, figsize=(12, 4 * len(layers)))
        
        if len(layers) == 1:
            axs = np.array([axs])
        
        for i, layer_idx in enumerate(layers):
            if layer_idx >= len(self.layers):
                print(f"Warning: Layer {layer_idx} does not exist. Skipping.")
                continue
            
            layer = self.layers[layer_idx]
            
            sns.histplot(layer.weights.flatten(), kde=True, ax=axs[i, 0])
            axs[i, 0].set_title(f"Layer {layer_idx+1} Weight Distribution")
            axs[i, 0].set_xlabel("Weight Value")
            axs[i, 0].set_ylabel("Frequency")
            
            sns.histplot(layer.bias.flatten(), kde=True, ax=axs[i, 1])
            axs[i, 1].set_title(f"Layer {layer_idx+1} Bias Distribution")
            axs[i, 1].set_xlabel("Bias Value")
            axs[i, 1].set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.show()
    
    def plot_gradient_distribution(self, layers: List[int] = None) -> None:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib and seaborn are required for plotting gradient distributions.")
            return
        
        if layers is None:
            layers = list(range(len(self.layers)))
        
        if not hasattr(self.layers[0], 'weights_gradient') or self.layers[0].weights_gradient is None:
            print("Gradients have not been calculated yet. Run backward propagation first.")
            return
        
        fig, axs = plt.subplots(len(layers), 2, figsize=(12, 4 * len(layers)))
        
        if len(layers) == 1:
            axs = np.array([axs])
        
        for i, layer_idx in enumerate(layers):
            if layer_idx >= len(self.layers):
                print(f"Warning: Layer {layer_idx} does not exist. Skipping.")
                continue
            
            layer = self.layers[layer_idx]
            
            sns.histplot(layer.weights_gradient.flatten(), kde=True, ax=axs[i, 0])
            axs[i, 0].set_title(f"Layer {layer_idx+1} Weight Gradient Distribution")
            axs[i, 0].set_xlabel("Weight Gradient Value")
            axs[i, 0].set_ylabel("Frequency")
            
            sns.histplot(layer.bias_gradient.flatten(), kde=True, ax=axs[i, 1])
            axs[i, 1].set_title(f"Layer {layer_idx+1} Bias Gradient Distribution")
            axs[i, 1].set_xlabel("Bias Gradient Value")
            axs[i, 1].set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.show()
    
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