import numpy as np

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Menghitung akurasi berdasarkan perbandingan label yang benar."""
    if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
        y_pred = np.argmax(y_pred, axis=-1)
        if y_true.ndim > 1 and y_true.shape[-1] > 1:
            y_true = np.argmax(y_true, axis=-1)
    else:
        y_pred = (y_pred > 0.5).astype(int)
    
    return np.mean(y_pred == y_true)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> float:
    """Menghitung precision dengan berbagai metode averaging."""
    if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(y_true, axis=-1) if y_true.ndim > 1 else y_true
    else:
        y_pred = (y_pred > 0.5).astype(int)

    if average == 'binary' or (np.max(y_pred) <= 1 and np.max(y_true) <= 1):
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    num_classes = max(np.max(y_pred) + 1, np.max(y_true) + 1)
    scores = [(np.sum((y_pred == cls) & (y_true == cls)) / max(np.sum(y_pred == cls), 1)) for cls in range(num_classes)]

    return np.mean(scores) if average in ['macro', 'micro'] else np.sum(scores * np.bincount(y_true) / len(y_true))


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> float:
    """Menghitung recall berdasarkan label prediksi dan label sebenarnya."""
    if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(y_true, axis=-1) if y_true.ndim > 1 else y_true
    else:
        y_pred = (y_pred > 0.5).astype(int)

    if average == 'binary' or (np.max(y_pred) <= 1 and np.max(y_true) <= 1):
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    num_classes = max(np.max(y_pred) + 1, np.max(y_true) + 1)
    scores = [(np.sum((y_pred == cls) & (y_true == cls)) / max(np.sum(y_true == cls), 1)) for cls in range(num_classes)]

    return np.mean(scores) if average in ['macro', 'micro'] else np.sum(scores * np.bincount(y_true) / len(y_true))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> float:
    """Menghitung F1-score dengan rumus harmonic mean dari precision dan recall."""
    precision = precision_score(y_true, y_pred, average)
    recall = recall_score(y_true, y_pred, average)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Menghasilkan confusion matrix untuk klasifikasi multi-kelas."""
    if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(y_true, axis=-1) if y_true.ndim > 1 else y_true
    else:
        y_pred = (y_pred > 0.5).astype(int)

    num_classes = max(np.max(y_pred) + 1, np.max(y_true) + 1)
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1

    return matrix
