from typing import Dict, List, Any, Optional
import time
import torch
import psutil
import numpy as np
from tqdm import tqdm
from .metrics import get_metric_fn

class ModelEvaluator:
    def __init__(self, device: Optional[str] = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self._supported_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']

    def evaluate_model(self, 
                      model: torch.nn.Module, 
                      dataloader: torch.utils.data.DataLoader,
                      metrics: List[str]) -> Dict[str, Any]:
        """
        Evaluates a PyTorch model using specified metrics.
        
        Args:
            model: PyTorch model to evaluate
            dataloader: DataLoader containing validation/test data
            metrics: List of metric names to compute
            
        Returns:
            Dictionary containing computed metrics
        """
        # Validate metrics
        if not metrics:
            raise ValueError("At least one metric must be specified")
        
        invalid_metrics = set(metrics) - set(self._supported_metrics)
        if invalid_metrics:
            raise ValueError(f"Unsupported metrics: {invalid_metrics}")

        predictions = []
        targets = []
        
        try:
            model.eval()
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Evaluating"):
                    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                        raise ValueError("Dataloader must yield (inputs, targets) pairs")
                        
                    inputs, target = batch
                    if not isinstance(inputs, torch.Tensor) or not isinstance(target, torch.Tensor):
                        raise TypeError("Both inputs and targets must be torch.Tensor objects")
                        
                    inputs = inputs.to(self.device)
                    target = target.to(self.device)
                    
                    try:
                        output = model(inputs)
                        predictions.append(output.detach().cpu().numpy())
                        targets.append(target.detach().cpu().numpy())
                    except Exception as e:
                        raise RuntimeError(f"Error during model inference: {str(e)}")
            
            if not predictions:
                raise ValueError("No predictions were generated. Dataset might be empty.")
                
            try:
                predictions = np.concatenate(predictions)
                targets = np.concatenate(targets)
            except ValueError as e:
                raise ValueError(f"Failed to concatenate predictions and targets: {str(e)}")
                
            results = {}
            for metric in metrics:
                metric_fn = get_metric_fn(metric)
                try:
                    results[metric] = metric_fn(targets, predictions)
                except Exception as e:
                    results[metric] = f"Error computing {metric}: {str(e)}"
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {str(e)}") 