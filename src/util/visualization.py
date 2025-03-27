import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
import pandas as pd
from matplotlib.figure import Figure

def visualize_model_performance(model, history, title="Model Performance"):
    """Visualize single model's training history with loss curves.
    
    Parameters:
    -----------
    model : NeuralNetwork
        The trained model
    history : dict
        Training history dictionary with 'train_loss' and 'val_loss' keys
    title : str
        Plot title
    
    Returns:
    --------
    fig : matplotlib figure
    """
    # Create figure with 1 row, 2 columns (loss plot + weight visualization)
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(history["train_loss"], label="Training Loss", color='steelblue')
    
    if "val_loss" in history:
        ax1.plot(history["val_loss"], linestyle="--", label="Validation Loss", color='indianred')
    
    ax1.set_title(f"{title}: Loss History", fontsize=12)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    ax2 = fig.add_subplot(1, 2, 2)
    
    layers_to_sample = []
    dense_layers = [i for i, layer in enumerate(model.layers) if hasattr(layer, 'weights')]
    
    if dense_layers:
        layers_to_sample = dense_layers
            
        for layer_idx in layers_to_sample:
            weights = model.layers[layer_idx].weights.flatten()
            sns.kdeplot(weights, label=f"Layer {layer_idx}", ax=ax2)
            
        ax2.set_title("Weight Distributions", fontsize=12)
        ax2.set_xlabel("Weight Value", fontsize=10)
        ax2.set_ylabel("Density", fontsize=10)
        ax2.legend(fontsize=9)
    
    plt.tight_layout()
    return fig

def visualize_model_weights_and_gradients(model, max_layers=8):
    """Visualize weight and gradient distributions for model layers.
    
    Parameters:
    -----------
    model : NeuralNetwork
        Trained model with gradient information
    max_layers : int
        Maximum number of layers to visualize
        
    Returns:
    --------
    fig : matplotlib figure
    """
    # Find all dense layers with weights and gradients
    layer_indices = [i for i, layer in enumerate(model.layers) 
                    if hasattr(layer, 'weights')]
    
    # Limit the number of layers displayed by selecting representative ones
    if len(layer_indices) > max_layers:
        step = len(layer_indices) // max_layers
        if step < 1:
            step = 1
        layer_indices = layer_indices[::step][:max_layers]
    
    n_layers = len(layer_indices)
    if n_layers == 0:
        print("No layers with weights found.")
        return None
    
    # Create figure with a grid layout - 2 rows per layer (weights and gradients)
    fig, axs = plt.subplots(n_layers, 2, figsize=(10, 3*n_layers))
    
    # Handle case of single layer
    if n_layers == 1:
        axs = axs.reshape(1, 2)
    
    for i, layer_idx in enumerate(layer_indices):
        layer = model.layers[layer_idx]
        
        # Plot weights
        weights = layer.weights.flatten()
        sns.histplot(weights, kde=True, ax=axs[i, 0], color='steelblue', bins=30)
        axs[i, 0].set_title(f'Layer {layer_idx}: Weights', fontsize=10)
        
        # Add weight statistics
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
            
            # Add gradient statistics
            g_mean = np.mean(gradients)
            g_std = np.std(gradients)
            axs[i, 1].text(0.95, 0.95, f"μ: {g_mean:.10f}\nσ: {g_std:.10f}", 
                          transform=axs[i, 1].transAxes, 
                          ha='right', va='top', fontsize=8,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            axs[i, 1].text(0.5, 0.5, "No gradient data available", 
                          ha='center', va='center', fontsize=10)
            axs[i, 1].set_title(f'Layer {layer_idx}: No Gradients', fontsize=10)
    
    plt.tight_layout()
    return fig

def visualize_prediction_comparison(models_dict, X_test, y_test, num_samples=5):
    """Compare predictions from multiple models on the same test samples.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of {model_name: model} or {model_name: results_dict}
    X_test : ndarray
        Test features
    y_test : ndarray
        True test labels (one-hot encoded)
    num_samples : int
        Number of samples to visualize
        
    Returns:
    --------
    fig : matplotlib figure
    """
    # Select random samples to visualize
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    X_samples = X_test[indices]
    y_true = y_test[indices]
    
    # Get model names and models
    model_names = list(models_dict.keys())
    models = []
    
    # Extract models from results_dict if necessary
    for name, model_or_results in models_dict.items():
        if hasattr(model_or_results, 'predict'):
            models.append(model_or_results)  # It's a model
        elif isinstance(model_or_results, dict) and "model" in model_or_results:
            models.append(model_or_results["model"])  # It's a results dict
        elif isinstance(model_or_results, tuple) and len(model_or_results) >= 1:
            models.append(model_or_results[0])  # It's a tuple with model as first element
        else:
            raise ValueError(f"Could not extract model for {name}")
    
    # Create grid for visualization: (samples × models)
    fig, axs = plt.subplots(num_samples, len(models) + 1, figsize=(2*len(models) + 2, 1.8*num_samples))
    
    # Handle case of single sample or model
    if num_samples == 1:
        axs = axs.reshape(1, -1)
    
    # For each sample
    for i, (x, y) in enumerate(zip(X_samples, y_true)):
        # Show the input image in the first column
        if x.shape[0] == 784:  # MNIST image
            img_shape = (28, 28)
        else:
            img_shape = (int(np.sqrt(x.shape[0])), int(np.sqrt(x.shape[0])))
        
        axs[i, 0].imshow(x.reshape(img_shape), cmap='gray')
        axs[i, 0].set_title(f"True: {np.argmax(y)}", fontsize=9)
        axs[i, 0].axis('off')
        
        # For each model
        for j, model in enumerate(models):
            # Get prediction for this sample
            pred = model.predict(x.reshape(1, -1))[0]
            pred_class = np.argmax(pred)
            
            # Create bar chart of class probabilities
            bars = axs[i, j+1].bar(range(len(pred)), pred, color='steelblue', alpha=0.7)
            
            # Highlight predicted and true classes
            pred_idx = np.argmax(pred)
            true_idx = np.argmax(y)
            
            if pred_idx < len(bars):
                bars[pred_idx].set_color('green')
            if true_idx < len(bars) and true_idx != pred_idx:
                bars[true_idx].set_color('red')
            
            # Set title and labels
            axs[i, j+1].set_title(f"{model_names[j]}: {pred_class}", fontsize=9)
            axs[i, j+1].set_xticks(range(len(pred)))
            axs[i, j+1].set_ylim(0, 1)
            
            # Reduce tick label size
            axs[i, j+1].tick_params(axis='both', which='major', labelsize=7)
            
            if i == num_samples - 1:
                axs[i, j+1].set_xlabel('Class', fontsize=8)
            
            if j > 0:
                axs[i, j+1].set_yticklabels([])
    
    plt.tight_layout()
    return fig

def compare_models_performance(models_dict, figsize=(12, 9)):
    """Compare loss, weights, and gradients across multiple models.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of {model_name: (model, history, ...)} or 
                      {model_name: {'model': model, 'history': history}}
        
    Returns:
    --------
    tuple of (loss_fig, weights_fig)
    """
    # Extract models and histories
    models = {}
    histories = {}
    
    for name, result in models_dict.items():
        # Extract model
        if isinstance(result, dict) and "model" in result:
            models[name] = result["model"]
            histories[name] = result["history"]
        elif isinstance(result, tuple) and len(result) >= 2:
            models[name] = result[0]  # Assuming model is at index 0
            histories[name] = result[1]  # Assuming history is at index 1
        else:
            raise ValueError(f"Could not extract model/history for {name}")
    
    # 1. Create loss comparison figure
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
    
    # Find a representative layer index present in all models
    common_layer_indices = set(range(min([len(model.layers) for model in models.values()])))
    dense_layer_indices = []
    
    # Find the first dense layer index present in all models
    for idx in common_layer_indices:
        if all(hasattr(model.layers[idx], 'weights') for model in models.values()):
            dense_layer_indices.append(idx)
    
    if not dense_layer_indices:
        print("No common dense layers found across models")
        return loss_fig, None
    
    # Select up to 3 representative layers
    if len(dense_layer_indices) > 3:
        dense_layer_indices = [
            dense_layer_indices[0],
            dense_layer_indices[len(dense_layer_indices)//2],
            dense_layer_indices[-1]
        ]
    
    # Create weights comparison figure
    n_layers = len(dense_layer_indices)
    weights_fig, w_axs = plt.subplots(n_layers, 1, figsize=(figsize[0], figsize[1]//2))
    
    if n_layers == 1:
        w_axs = [w_axs]
    
    # Plot weight distributions for each selected layer
    for i, layer_idx in enumerate(dense_layer_indices):
        for name, model in models.items():
            weights = model.layers[layer_idx].weights.flatten()
            sns.kdeplot(weights, ax=w_axs[i], label=name)
        
        w_axs[i].set_title(f"Layer {layer_idx} Weight Distribution", fontsize=12)
        w_axs[i].set_xlabel("Weight Value", fontsize=10)
        w_axs[i].set_ylabel("Density", fontsize=10)
        w_axs[i].legend(fontsize=9)
    
    plt.tight_layout()
    return loss_fig, weights_fig

def visualize_all_model_results(models_dict, X_test=None, y_test=None, num_samples=3):
    """Comprehensive visualization of multiple models.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of {model_name: (model, history, predictions, accuracy, results)}
    X_test : ndarray, optional
        Test data for prediction visualization
    y_test : ndarray, optional
        Test labels for prediction visualization
    num_samples : int
        Number of test samples to visualize
        
    Returns:
    --------
    dict : Dictionary of figures
    """
    figures = {}
    
    # 1. Compare losses
    loss_fig, weights_fig = compare_models_performance(models_dict)
    figures['loss_comparison'] = loss_fig
    figures['weight_comparison'] = weights_fig
    
    # 2. Compare predictions if test data is provided
    if X_test is not None and y_test is not None:
        models_only = {}
        for name, result in models_dict.items():
            if isinstance(result, dict) and "model" in result:
                models_only[name] = result["model"]
            elif isinstance(result, tuple) and len(result) >= 1:
                models_only[name] = result[0]  # Assuming model is at index 0
            
        pred_fig = visualize_prediction_comparison(models_only, X_test, y_test, num_samples)
        figures['prediction_comparison'] = pred_fig
    
    has_accuracy = False
    accuracies = {}
    
    for name, result in models_dict.items():
        if isinstance(result, dict) and "accuracy" in result:
            accuracies[name] = result["accuracy"]
            has_accuracy = True
        elif isinstance(result, tuple) and len(result) >= 4:
            accuracies[name] = result[3]  # Assuming accuracy is at index 3
            has_accuracy = True
    
    if has_accuracy:
        acc_fig, acc_ax = plt.subplots(figsize=(10, 5))
        model_names = list(accuracies.keys())
        acc_values = list(accuracies.values())
        
        bars = acc_ax.bar(model_names, acc_values, color='steelblue', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            acc_ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                      f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        acc_ax.set_title("Model Accuracy Comparison", fontsize=14)
        acc_ax.set_xlabel("Model", fontsize=12)
        acc_ax.set_ylabel("Accuracy", fontsize=12)
        acc_ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.tight_layout()
        
        figures['accuracy_comparison'] = acc_fig
    
    return figures

def visualize_model_comparison(custom_model, sklearn_model, X_test, y_test, custom_history=None, sklearn_history=None, custom_preds=None):
    """Compare a custom neural network with sklearn's MLPClassifier.
    
    Parameters:
    -----------
    custom_model : NeuralNetwork
        Custom neural network model
    sklearn_model : MLPClassifier
        Scikit-learn MLPClassifier model
    X_test : ndarray
        Test data features
    y_test : ndarray
        Test data labels (one-hot encoded)
    custom_history : dict, optional
        Training history for custom model
    sklearn_history : dict, optional
        Training history for sklearn model
    custom_preds : ndarray, optional
        Precomputed predictions from custom model
        
    Returns:
    --------
    dict : Dictionary of visualization figures
    """
    figures = {}
    
    # Get predictions if not provided
    if custom_preds is None:
        custom_preds = custom_model.predict(X_test)
    
    # Calculate model metrics
    custom_accuracy = np.mean(np.argmax(custom_preds, axis=1) == np.argmax(y_test, axis=1))
    
    sklearn_preds = sklearn_model.predict(X_test)
    sklearn_preds_proba = sklearn_model.predict_proba(X_test)
    sklearn_accuracy = sklearn_model.score(X_test, np.argmax(y_test, axis=1))
    sklearn_params = sum(coef.size for coef in sklearn_model.coefs_) + sum(intercept.size for intercept in sklearn_model.intercepts_)
    
    # Convert sklearn predictions to one-hot format for consistent comparison
    sklearn_preds_one_hot = np.zeros_like(y_test)
    for i, pred_idx in enumerate(sklearn_preds):
        sklearn_preds_one_hot[i, pred_idx] = 1
    
    # Calculate custom model parameters
    custom_params = sum(layer.weights.size + layer.bias.size 
                      for layer in custom_model.layers if hasattr(layer, 'weights'))
    
    acc_fig, acc_ax = plt.subplots(figsize=(10, 5))
    accuracies = [custom_accuracy, sklearn_accuracy]
    model_names = ["Custom NN", "Scikit MLP"]
    bars = acc_ax.bar(model_names, accuracies, color=['steelblue', 'indianred'], alpha=0.7)

    # Add accuracy values on bars
    for bar in bars:
        height = bar.get_height()
        acc_ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=12)

    acc_ax.set_title("Model Accuracy Comparison", fontsize=16)
    acc_ax.set_ylabel("Accuracy", fontsize=14)
    acc_ax.grid(True, alpha=0.3, axis='y')
    acc_ax.set_ylim(0, 1.1)
    
    figures['accuracy_comparison'] = acc_fig
    
    if custom_history is not None and hasattr(sklearn_model, 'loss_curve_'):
        loss_fig, loss_ax = plt.subplots(figsize=(10, 6))
        loss_ax.plot(custom_history["train_loss"], label="Custom NN Training Loss", color='steelblue')
        loss_ax.plot(sklearn_model.loss_curve_, label="Scikit MLP Loss", color='indianred')
        loss_ax.set_title("Training Loss Comparison", fontsize=16)
        loss_ax.set_xlabel("Iterations", fontsize=14)
        loss_ax.set_ylabel("Loss", fontsize=14)
        loss_ax.grid(True, alpha=0.3)
        loss_ax.legend(fontsize=12)
        
        figures['loss_comparison'] = loss_fig
    
    num_samples = 5
    indices = np.random.choice(len(X_test), num_samples, replace=False)

    pred_fig, axs = plt.subplots(num_samples, 3, figsize=(15, 3*num_samples))
    pred_fig.suptitle("Prediction Comparison: Custom NN vs Scikit-Learn MLP", fontsize=16)

    for i, idx in enumerate(indices):
        # Original image
        img_shape = (28, 28) if X_test.shape[1] == 784 else (int(np.sqrt(X_test.shape[1])), int(np.sqrt(X_test.shape[1])))
        img = X_test[idx].reshape(img_shape)
        axs[i, 0].imshow(img, cmap='gray')
        axs[i, 0].set_title(f"True: {np.argmax(y_test[idx])}", fontsize=9)
        axs[i, 0].axis('off')
        
        # Custom NN prediction
        custom_pred = np.argmax(custom_preds[idx])
        custom_correct = custom_pred == np.argmax(y_test[idx])
        color = 'green' if custom_correct else 'red'
        axs[i, 1].bar(range(len(custom_preds[idx])), custom_preds[idx], color='steelblue', alpha=0.7)
        axs[i, 1].set_title(f"Custom NN: {custom_pred}", color=color)
        axs[i, 1].set_xticks(range(len(custom_preds[idx])))
        
        # Scikit-learn MLP prediction
        sklearn_pred = np.argmax(sklearn_preds_proba[idx])
        sklearn_correct = sklearn_pred == np.argmax(y_test[idx])
        color = 'green' if sklearn_correct else 'red'
        axs[i, 2].bar(range(len(sklearn_preds_proba[idx])), sklearn_preds_proba[idx], color='indianred', alpha=0.7)
        axs[i, 2].set_title(f"Scikit MLP: {sklearn_pred}", color=color)
        axs[i, 2].set_xticks(range(len(sklearn_preds_proba[idx])))

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    figures['prediction_comparison'] = pred_fig
    
    class_fig = compare_per_class_accuracy(
        custom_preds, 
        sklearn_preds_one_hot, 
        y_test, 
        ["Custom NN", "Scikit MLP"]
    )
    figures['class_accuracy'] = class_fig
    
    summary = f"""
    Model Comparison Summary:
    {'-' * 60}
    {'Model':<15} {'Accuracy':<10} {'Parameters':<10}
    {'-' * 60}
    {'Custom NN':<15} {custom_accuracy:<10.4f} {custom_params:<10}
    {'Scikit MLP':<15} {sklearn_accuracy:<10.4f} {sklearn_params:<10}
    {'-' * 60}
    
    Performance difference: {(sklearn_accuracy - custom_accuracy)*100:.2f}%
    """
    
    return figures, summary

def compare_per_class_accuracy(custom_preds, sklearn_preds, y_test, model_names=None):
    """Compare per-class accuracy between models.
    
    Parameters:
    -----------
    custom_preds : ndarray
        Predictions from custom model (one-hot encoded)
    sklearn_preds : ndarray
        Predictions from sklearn model (one-hot encoded)
    y_test : ndarray
        Ground truth labels (one-hot encoded)
    model_names : list, optional
        Names of the models for the legend
        
    Returns:
    --------
    fig : matplotlib figure
    """
    if model_names is None:
        model_names = ["Model 1", "Model 2"]
    
    # Calculate per-class accuracies
    custom_class_acc = np.zeros(y_test.shape[1])
    sklearn_class_acc = np.zeros(y_test.shape[1])

    y_true = np.argmax(y_test, axis=1)
    custom_pred = np.argmax(custom_preds, axis=1)
    sklearn_pred = np.argmax(sklearn_preds, axis=1)

    for i in range(y_test.shape[1]):
        class_indices = np.where(y_true == i)[0]
        if len(class_indices) > 0:
            custom_class_acc[i] = np.mean(custom_pred[class_indices] == i)
            sklearn_class_acc[i] = np.mean(sklearn_pred[class_indices] == i)
    
    # Plot per-class accuracies
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(y_test.shape[1])
    width = 0.35
    
    ax.bar(x - width/2, custom_class_acc, width, label=model_names[0], color='steelblue', alpha=0.7)
    ax.bar(x + width/2, sklearn_class_acc, width, label=model_names[1], color='indianred', alpha=0.7)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-Class Accuracy Comparison', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(y_test.shape[1])])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig