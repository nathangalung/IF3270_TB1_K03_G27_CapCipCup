import numpy as np

def accuracy_score(y_true, y_pred):
    """Calculate prediction accuracy"""
    # Convert probabilities to class predictions
    if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
        y_pred = np.argmax(y_pred, axis=-1)
        if y_true.ndim > 1 and y_true.shape[-1] > 1:
            y_true = np.argmax(y_true, axis=-1)
    else:
        y_pred = (y_pred > 0.5).astype(int)
    
    return np.mean(y_pred == y_true)


def precision_score(y_true, y_pred, average='binary', per_class=True):
    """Calculate precision score"""
    # Convert to class indices
    if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
        y_pred = np.argmax(y_pred, axis=-1)
        if y_true.ndim > 1 and y_true.shape[-1] > 1:
            y_true = np.argmax(y_true, axis=-1)
    else:
        y_pred = (y_pred > 0.5).astype(int)

    # Binary classification case
    if average == 'binary' or (np.max(y_pred) <= 1 and np.max(y_true) <= 1):
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Multi-class case
    num_classes = max(np.max(y_pred) + 1, np.max(y_true) + 1)
    precisions = np.zeros(num_classes)
    
    for cls in range(num_classes):
        # Number of true positives (correctly predicted as this class)
        tp = np.sum((y_pred == cls) & (y_true == cls))
        
        # Number of predictions for this class
        pred_as_cls = np.sum(y_pred == cls)
        
        # Calculate precision for this class
        if pred_as_cls > 0:
            precisions[cls] = tp / pred_as_cls
        # Class with no predictions gets precision of 0

    # Return per-class precision if requested
    if per_class:
        return precisions
    
    # Different averaging methods
    if average is None or average == 'none':
        return precisions
    elif average == 'macro':
        # Simple mean of class-wise precision
        return np.mean(precisions)
    elif average == 'micro':
        # Calculate global TP and FP
        tp_sum = sum(np.sum((y_pred == cls) & (y_true == cls)) for cls in range(num_classes))
        fp_sum = sum(np.sum((y_pred == cls) & (y_true != cls)) for cls in range(num_classes))
        return tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    else:  # weighted
        # Weight by class frequency
        class_counts = np.bincount(y_true, minlength=num_classes)
        sample_count = len(y_true)
        if sample_count > 0:
            weights = class_counts / sample_count
            return np.sum(precisions * weights)
        return 0.0


def recall_score(y_true, y_pred, average='binary', per_class=True):
    """Calculate recall score"""
    # Convert to class indices
    if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
        y_pred = np.argmax(y_pred, axis=-1)
        if y_true.ndim > 1 and y_true.shape[-1] > 1:
            y_true = np.argmax(y_true, axis=-1)
    else:
        y_pred = (y_pred > 0.5).astype(int)

    # Binary classification case
    if average == 'binary' or (np.max(y_pred) <= 1 and np.max(y_true) <= 1):
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Multi-class case
    num_classes = max(np.max(y_pred) + 1, np.max(y_true) + 1)
    recalls = np.zeros(num_classes)
    
    for cls in range(num_classes):
        # Number of true positives (correctly predicted as this class)
        tp = np.sum((y_pred == cls) & (y_true == cls))
        
        # Number of true instances of this class
        actual_cls = np.sum(y_true == cls)
        
        # Calculate recall for this class
        if actual_cls > 0:
            recalls[cls] = tp / actual_cls
        # Class with no instances gets recall of 0

    # Return per-class recall if requested
    if per_class:
        return recalls
        
    # Different averaging methods
    if average is None or average == 'none':
        return recalls
    elif average == 'macro':
        # Simple mean of class-wise recall
        return np.mean(recalls)
    elif average == 'micro':
        # Calculate global TP and FN
        tp_sum = sum(np.sum((y_pred == cls) & (y_true == cls)) for cls in range(num_classes))
        fn_sum = sum(np.sum((y_pred != cls) & (y_true == cls)) for cls in range(num_classes))
        return tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    else:  # weighted
        # Weight by class frequency
        class_counts = np.bincount(y_true, minlength=num_classes)
        sample_count = len(y_true)
        if sample_count > 0:
            weights = class_counts / sample_count
            return np.sum(recalls * weights)
        return 0.0


def f1_score(y_true, y_pred, average='binary', per_class=True):
    """Calculate F1 score"""
    # Get per-class precision and recall
    precisions = precision_score(y_true, y_pred, average=None)
    recalls = recall_score(y_true, y_pred, average=None)
    
    # Calculate per-class F1
    f1_scores = np.zeros_like(precisions)
    for i in range(len(precisions)):
        if precisions[i] + recalls[i] > 0:
            f1_scores[i] = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
    
    # Return per-class F1 if requested
    if per_class:
        return f1_scores
        
    # Average F1 scores based on method
    if average is None or average == 'none':
        return f1_scores
    elif average == 'macro':
        return np.mean(f1_scores)
    elif average == 'micro':
        # Recalculate using micro-averaged precision and recall
        prec = precision_score(y_true, y_pred, average='micro')
        rec = recall_score(y_true, y_pred, average='micro')
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    else:  # weighted
        # Weight by class frequency
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true_indices = np.argmax(y_true, axis=1)
        else:
            y_true_indices = y_true
            
        num_classes = len(f1_scores)
        class_counts = np.bincount(y_true_indices, minlength=num_classes)
        weights = class_counts / len(y_true_indices)
        return np.sum(f1_scores * weights)


def confusion_matrix(y_true, y_pred):
    """Generate confusion matrix where rows are true labels and columns are predicted labels"""
    # Convert to class indices
    if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
        y_pred = np.argmax(y_pred, axis=-1)
        if y_true.ndim > 1 and y_true.shape[-1] > 1:
            y_true = np.argmax(y_true, axis=-1)
    else:
        y_pred = (y_pred > 0.5).astype(int)

    # Create matrix
    num_classes = max(np.max(y_pred) + 1, np.max(y_true) + 1)
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Fill matrix: rows are true labels, columns are predicted labels
    for i in range(len(y_true)):
        matrix[y_true[i], y_pred[i]] += 1

    return matrix


def classification_report(y_true, y_pred, digits=3):
    """Generate a text report showing the main classification metrics"""
    # Convert to class indices
    if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
        y_pred_indices = np.argmax(y_pred, axis=-1)
        if y_true.ndim > 1 and y_true.shape[-1] > 1:
            y_true_indices = np.argmax(y_true, axis=-1)
        else:
            y_true_indices = y_true
    else:
        y_pred_indices = (y_pred > 0.5).astype(int)
        y_true_indices = y_true
    
    # Calculate accuracy first - this should match your reported accuracy
    acc = np.mean(y_pred_indices == y_true_indices)
    
    # Get metrics
    precisions = precision_score(y_true_indices, y_pred_indices, average=None)
    recalls = recall_score(y_true_indices, y_pred_indices, average=None)
    f1_scores = f1_score(y_true_indices, y_pred_indices, average=None)
    
    # Get support (number of samples per class)
    num_classes = len(precisions)
    support = np.bincount(y_true_indices, minlength=num_classes)
    
    # Create header
    header = f"{'':>8} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
    report = [header, "-" * len(header)]
    
    # Add per-class metrics
    for i in range(num_classes):
        report.append(f"{i:>8} {precisions[i]:{digits+6}.{digits}f} {recalls[i]:{digits+6}.{digits}f} "
                    f"{f1_scores[i]:{digits+6}.{digits}f} {support[i]:>10}")
    
    # Add averages
    report.append("-" * len(header))
    
    # Macro average (equal weight to all classes)
    macro_prec = np.mean(precisions)
    macro_rec = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)
    report.append(f"{'macro avg':>8} {macro_prec:{digits+6}.{digits}f} {macro_rec:{digits+6}.{digits}f} "
                f"{macro_f1:{digits+6}.{digits}f} {sum(support):>10}")
    
    # Weighted average (weight by class frequency)
    weights = support / sum(support)
    weighted_prec = np.sum(precisions * weights)
    weighted_rec = np.sum(recalls * weights)
    weighted_f1 = np.sum(f1_scores * weights)
    report.append(f"{'weighted avg':>8} {weighted_prec:{digits+6}.{digits}f} {weighted_rec:{digits+6}.{digits}f} "
                f"{weighted_f1:{digits+6}.{digits}f} {sum(support):>10}")
    
    # Accuracy
    report.append(f"\nAccuracy: {acc:.{digits}f}")
    
    return "\n".join(report)