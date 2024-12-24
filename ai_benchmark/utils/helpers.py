import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import psutil
import GPUtil

class ConfigLoader:
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate configuration structure"""
        required_sections = ['models', 'datasets', 'evaluation', 'visualization']
        required_settings = {
            'models': ['default_device', 'save_checkpoints', 'checkpoint_dir'],
            'datasets': ['batch_size', 'num_workers'],
            'evaluation': ['default_metrics', 'resource_monitoring'],
            'visualization': ['dashboard', 'plots']
        }
        
        # Check required sections
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
            
            # Check required settings in each section
            for setting in required_settings[section]:
                if setting not in config[section]:
                    raise ValueError(f"Missing required setting '{setting}' in section '{section}'")
        
        return True

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load and validate configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        ConfigLoader.validate_config(config)
        return config

class ResourceMonitor:
    @staticmethod
    def get_system_stats() -> Dict[str, float]:
        """Get current system resource usage"""
        stats = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3)
        }
        
        # Add GPU stats if available
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                stats[f'gpu_{i}_util'] = gpu.load * 100
                stats[f'gpu_{i}_memory'] = gpu.memoryUtil * 100
        except Exception as e:
            logging.warning(f"Failed to get GPU stats: {e}")
            
        return stats

class ModelUtils:
    @staticmethod
    def get_model_size(model: torch.nn.Module) -> Dict[str, Any]:
        """Calculate model size and parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        }
    
    @staticmethod
    def move_to_device(model: torch.nn.Module, device: Optional[str] = None) -> torch.nn.Module:
        """Move model to specified device or automatically select best available"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return model.to(device)

class ResultsManager:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_results(self, results: Dict[str, Any], name: str):
        """Save benchmark results to file"""
        output_path = self.output_dir / f"{name}.yaml"
        with open(output_path, 'w') as f:
            yaml.dump(results, f)
    
    def load_results(self, name: str) -> Dict[str, Any]:
        """Load benchmark results from file"""
        input_path = self.output_dir / f"{name}.yaml"
        with open(input_path, 'r') as f:
            return yaml.safe_load(f)

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ) 