from typing import Dict, Any, Callable
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from nltk.translate.bleu_score import corpus_bleu

class MetricsRegistry:
    def __init__(self):
        self.metrics: Dict[str, Callable] = {}
        self._register_default_metrics()
    
    def _register_default_metrics(self):
        # Classification metrics
        self.register_metric('accuracy', accuracy_score)
        self.register_metric('precision_recall_f1', 
            lambda y_true, y_pred: precision_recall_fscore_support(y_true, y_pred, average='weighted'))
        
        # Regression metrics
        self.register_metric('mae', lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)))
        self.register_metric('rmse', lambda y_true, y_pred: np.sqrt(np.mean((y_true - y_pred)**2)))
        
        # NLP metrics
        self.register_metric('bleu', lambda references, candidates: corpus_bleu(references, candidates))
    
    def register_metric(self, name: str, metric_fn: Callable):
        """Register a custom metric function"""
        self.metrics[name] = metric_fn
    
    def compute_metric(self, name: str, *args, **kwargs) -> float:
        """Compute a specific metric"""
        if name not in self.metrics:
            raise ValueError(f"Metric '{name}' not found in registry")
        return self.metrics[name](*args, **kwargs) 

def get_metric_fn(metric_name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Returns the corresponding metric function for a given metric name.
    
    Args:
        metric_name: Name of the metric to retrieve
        
    Returns:
        Metric computation function
    """
    metrics = {
        'accuracy': lambda y_true, y_pred: accuracy_score(y_true, np.argmax(y_pred, axis=1)),
        'precision': lambda y_true, y_pred: precision_score(y_true, np.argmax(y_pred, axis=1), average='macro'),
        'recall': lambda y_true, y_pred: recall_score(y_true, np.argmax(y_pred, axis=1), average='macro'),
        'f1': lambda y_true, y_pred: f1_score(y_true, np.argmax(y_pred, axis=1), average='macro'),
        'auc': lambda y_true, y_pred: roc_auc_score(y_true, y_pred, multi_class='ovr')
    }
    
    if metric_name not in metrics:
        raise ValueError(f"Unsupported metric: {metric_name}")
        
    return metrics[metric_name] 