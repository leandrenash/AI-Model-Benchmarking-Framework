import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import pandas as pd
from plotly.graph_objects import Figure

class BenchmarkPlotter:
    @staticmethod
    def create_metric_comparison(results: Dict[str, Any], metric_name: str) -> Figure:
        """
        Create comparison plot for a specific metric
        
        Args:
            results: Dictionary containing benchmark results
            metric_name: Name of the metric to plot
            
        Returns:
            plotly.graph_objects.Figure: The generated plot
            
        Raises:
            ValueError: If no results are provided or no data found for metric
        """
        if not results:
            raise ValueError("No results provided for plotting")
        
        data = []
        for model_name, model_results in results.items():
            if metric_name in model_results.get('metrics', {}):
                data.append({
                    'model': model_name,
                    'value': model_results['metrics'][metric_name]
                })
        
        if not data:
            raise ValueError(f"No data found for metric '{metric_name}'")
        
        df = pd.DataFrame(data)
        fig = px.bar(
            df,
            x='model',
            y='value',
            title=f'{metric_name.capitalize()} Comparison'
        )
        return fig
    
    @staticmethod
    def create_resource_usage_plot(results: Dict[str, Any]):
        """Create resource usage comparison plot"""
        data = []
        for model_name, model_results in results.items():
            data.append({
                'model': model_name,
                'memory_mb': model_results['resource_usage']['memory_mb'],
                'time_seconds': model_results['timing']['total_time']
            })
        
        df = pd.DataFrame(data)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Memory Usage (MB)',
            x=df['model'],
            y=df['memory_mb']
        ))
        fig.add_trace(go.Bar(
            name='Processing Time (s)',
            x=df['model'],
            y=df['time_seconds']
        ))
        
        fig.update_layout(
            title='Resource Usage Comparison',
            barmode='group'
        )
        return fig
    
    @staticmethod
    def create_performance_radar(results: Dict[str, Any], metrics: List[str]):
        """Create radar plot comparing model performance across metrics"""
        fig = go.Figure()
        
        for model_name, model_results in results.items():
            values = [model_results['metrics'].get(metric, 0) for metric in metrics]
            values.append(values[0])  # Close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                name=model_name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title='Model Performance Comparison'
        )
        return fig 