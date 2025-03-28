import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

def visualize_model_performance(model, history, title="Model Performance"):
    """Plot training history and weight distributions"""
    # Create figure with loss plot and weight visualization
    fig = plt.figure(figsize=(12, 5))
    
    # Left subplot: loss curves
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(history["train_loss"], label="Training Loss", color='steelblue')
    
    if "val_loss" in history:
        ax1.plot(history["val_loss"], linestyle="--", label="Validation Loss", color='indianred')
    
    ax1.set_title(f"{title}: Loss History", fontsize=12)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Right subplot: weight distributions
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Find layers with weights
    dense_layers = [i for i, layer in enumerate(model.layers) if hasattr(layer, 'weights')]
    
    if dense_layers:
        for layer_idx in dense_layers:
            weights = model.layers[layer_idx].weights.flatten()
            sns.kdeplot(weights, label=f"Layer {layer_idx}", ax=ax2)
            
        ax2.set_title("Weight Distributions", fontsize=12)
        ax2.set_xlabel("Weight Value", fontsize=10)
        ax2.set_ylabel("Density", fontsize=10)
        ax2.legend(fontsize=9)
    
    plt.tight_layout()
    return fig

def visualize_model_weights_and_gradients(model, max_layers=8):
    """Show weight and gradient distributions for layers"""
    # Find layers with weights
    layer_indices = [i for i, layer in enumerate(model.layers) 
                    if hasattr(layer, 'weights')]
    
    # Limit number of displayed layers
    if len(layer_indices) > max_layers:
        step = len(layer_indices) // max_layers
        if step < 1:
            step = 1
        layer_indices = layer_indices[::step][:max_layers]
    
    n_layers = len(layer_indices)
    if n_layers == 0:
        print("No layers with weights found.")
        return None
    
    # Create figure with weights and gradients side by side
    fig, axs = plt.subplots(n_layers, 2, figsize=(10, 3*n_layers))
    
    # Handle single layer case
    if n_layers == 1:
        axs = axs.reshape(1, 2)
    
    for i, layer_idx in enumerate(layer_indices):
        layer = model.layers[layer_idx]
        
        # Plot weights
        weights = layer.weights.flatten()
        sns.histplot(weights, kde=True, ax=axs[i, 0], color='steelblue', bins=30)
        axs[i, 0].set_title(f'Layer {layer_idx}: Weights', fontsize=10)
        
        # Add stats
        w_mean = np.mean(weights)
        w_std = np.std(weights)
        axs[i, 0].text(0.95, 0.95, f"μ: {w_mean:.5f}\nσ: {w_std:.5f}", 
                      transform=axs[i, 0].transAxes, 
                      ha='right', va='top', fontsize=8,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Plot gradients if available
        if hasattr(layer, 'weights_gradient') and layer.weights_gradient is not None:
            gradients = layer.weights_gradient.flatten()
            sns.histplot(gradients, kde=True, ax=axs[i, 1], color='indianred', bins=30)
            axs[i, 1].set_title(f'Layer {layer_idx}: Gradients', fontsize=10)
            
            # Add gradient stats
            g_mean = np.mean(gradients)
            g_std = np.std(gradients)
            axs[i, 1].text(0.95, 0.95, f"μ: {g_mean:.6f}\nσ: {g_std:.6f}", 
                          transform=axs[i, 1].transAxes, 
                          ha='right', va='top', fontsize=8,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            axs[i, 1].text(0.5, 0.5, "No gradient data available", 
                          ha='center', va='center', fontsize=10)
            axs[i, 1].set_title(f'Layer {layer_idx}: No Gradients', fontsize=10)
    
    plt.tight_layout()
    return fig

def compare_models_performance(models_dict, figsize=(12, 8)):
    """Compare training curves and weights across models"""
    # Extract models and histories
    models = {}
    histories = {}
    
    for name, result in models_dict.items():
        if isinstance(result, dict) and "model" in result:
            models[name] = result["model"]
            histories[name] = result["history"]
        elif isinstance(result, tuple) and len(result) >= 2:
            models[name] = result[0]
            histories[name] = result[1]
        else:
            raise ValueError(f"Could not extract model/history for {name}")
    
    # Create loss comparison figure
    loss_fig, loss_ax = plt.subplots(figsize=(figsize[0], figsize[1]//2))
    
    for name, history in histories.items():
        loss_ax.plot(history["train_loss"], label=f"{name} (Train)")
        if "val_loss" in history:
            loss_ax.plot(history["val_loss"], linestyle="--", label=f"{name} (Val)")
    
    loss_ax.set_title("Training Loss Comparison", fontsize=14)
    loss_ax.set_xlabel("Epoch", fontsize=12)
    loss_ax.set_ylabel("Loss", fontsize=12)
    loss_ax.grid(True, alpha=0.3)
    loss_ax.legend(fontsize=10)
    
    # Find common layers to compare weights
    common_layers = min([len(model.layers) for model in models.values()])
    dense_layers = []
    
    # Find layers with weights across all models
    for i in range(common_layers):
        if all(hasattr(model.layers[i], 'weights') for model in models.values()):
            dense_layers.append(i)
    
    if not dense_layers:
        return loss_fig, None
    
    # Select representative layers
    if len(dense_layers) > 3:
        dense_layers = [
            dense_layers[0],
            dense_layers[len(dense_layers)//2],
            dense_layers[-1]
        ]
    
    # Create weight comparison figure
    n_layers = len(dense_layers)
    weights_fig, w_axs = plt.subplots(n_layers, 1, figsize=(figsize[0], figsize[1]//2))
    
    if n_layers == 1:
        w_axs = [w_axs]
    
    # Plot weights for each model by layer
    for i, layer_idx in enumerate(dense_layers):
        for name, model in models.items():
            weights = model.layers[layer_idx].weights.flatten()
            sns.kdeplot(weights, ax=w_axs[i], label=name)
        
        w_axs[i].set_title(f"Layer {layer_idx} Weights", fontsize=12)
        w_axs[i].set_xlabel("Weight Value", fontsize=10)
        w_axs[i].set_ylabel("Density", fontsize=10)
        w_axs[i].legend(fontsize=9)
    
    plt.tight_layout()
    return loss_fig, weights_fig

def compare_per_class_accuracy(custom_model=None, sklearn_model=None, X_test=None, y_test=None, 
                              custom_history=None, sklearn_history=None, custom_preds=None,
                              model_names=None):
    if model_names is None:
        model_names = ["Custom Model", "Sklearn Model"]
    
    # Create loss comparison figure
    if custom_history is not None and sklearn_history is not None:
        loss_fig, loss_ax = plt.subplots(figsize=(12, 6))
        
        # Custom model loss
        loss_ax.plot(custom_history["train_loss"], label=f"{model_names[0]} (Train)", color='steelblue')
        if "val_loss" in custom_history:
            loss_ax.plot(custom_history["val_loss"], linestyle="--", 
                        label=f"{model_names[0]} (Val)", color='lightblue')
        
        # Sklearn model loss
        if 'loss' in sklearn_history:
            loss_ax.plot(sklearn_history["loss"], label=f"{model_names[1]}", color='indianred')
        
        loss_ax.set_title("Training Loss Comparison", fontsize=14)
        loss_ax.set_xlabel("Epoch", fontsize=12)
        loss_ax.set_ylabel("Loss", fontsize=12)
        loss_ax.grid(True, alpha=0.3)
        loss_ax.legend()
        plt.tight_layout()
    else:
        loss_fig = None
    
    # Get custom predictions
    if custom_preds is None and custom_model is not None and X_test is not None:
        custom_preds = custom_model.predict(X_test)
    
    # Get sklearn predictions
    if sklearn_model is not None and X_test is not None:
        if hasattr(sklearn_model, 'predict_proba'):
            sklearn_preds = sklearn_model.predict_proba(X_test)
        else:
            sklearn_preds = np.zeros((X_test.shape[0], y_test.shape[1]))
            preds = sklearn_model.predict(X_test)
            for i, p in enumerate(preds):
                sklearn_preds[i, p] = 1
    else:
        sklearn_preds = None
    
    # Compare per-class accuracy
    if custom_preds is not None and sklearn_preds is not None and y_test is not None:
        # Convert to class indices
        y_true = np.argmax(y_test, axis=1)
        pred1 = np.argmax(custom_preds, axis=1)
        pred2 = np.argmax(sklearn_preds, axis=1)
        
        # Calculate per-class accuracies
        num_classes = y_test.shape[1]
        acc1 = np.zeros(num_classes)
        acc2 = np.zeros(num_classes)

        for i in range(num_classes):
            class_samples = np.where(y_true == i)[0]
            if len(class_samples) > 0:
                acc1[i] = np.mean(pred1[class_samples] == i)
                acc2[i] = np.mean(pred2[class_samples] == i)
        
        # Create bar chart
        acc_fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(num_classes)
        width = 0.35
        
        ax.bar(x - width/2, acc1, width, label=model_names[0], color='steelblue', alpha=0.7)
        ax.bar(x + width/2, acc2, width, label=model_names[1], color='indianred', alpha=0.7)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Per-Class Accuracy Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(num_classes)])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Calculate overall accuracy
        custom_acc = np.mean(pred1 == y_true)
        sklearn_acc = np.mean(pred2 == y_true)
        
        # Create summary
        summary = {
            "custom_accuracy": custom_acc,
            "sklearn_accuracy": sklearn_acc,
            "custom_per_class": acc1,
            "sklearn_per_class": acc2,
            "custom_confusion": confusion_matrix(y_true, pred1),
            "sklearn_confusion": confusion_matrix(y_true, pred2)
        }
    else:
        acc_fig = None
        summary = {}
    
    # Return figures and data
    return {"loss": loss_fig, "accuracy": acc_fig}, summary