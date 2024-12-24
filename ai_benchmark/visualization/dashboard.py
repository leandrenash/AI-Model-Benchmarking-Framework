import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, Input, Output
import pandas as pd

class BenchmarkDashboard:
    def __init__(self):
        self.app = Dash(__name__)
        self.df = None  # Store DataFrame for callbacks
        self.setup_layout()
        
    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1("AI Model Benchmarking Dashboard"),
            
            # Model selection
            dcc.Dropdown(
                id='model-selector',
                options=[],  # Will be updated when data is loaded
                multi=True,
                placeholder='Select models to compare'
            ),
            
            # Metrics visualization
            html.Div([
                dcc.Graph(id='metrics-comparison'),
                dcc.Graph(id='resource-usage'),
                dcc.Graph(id='timing-analysis')
            ]),
            
            # Detailed results table
            html.Div(id='results-table')
        ])
        
        # Register callbacks
        self.register_callbacks()
    
    def register_callbacks(self):
        @self.app.callback(
            Output('metrics-comparison', 'figure'),
            [Input('model-selector', 'value')]
        )
        def update_metrics_plot(selected_models):
            if not selected_models or self.df is None:
                return go.Figure()  # Return empty figure if no data
                
            filtered_df = self.df[self.df['model'].isin(selected_models)]
            fig = px.bar(
                filtered_df,
                x='model',
                y='metric_value',
                color='metric_name',
                title='Model Performance Comparison'
            )
            return fig
    
    def update_data(self, benchmark_results: dict):
        """Update dashboard with new benchmark results"""
        self.df = self._process_results(benchmark_results)
        # Update dropdown options
        self.app.layout['model-selector'].options = [
            {'label': model, 'value': model}
            for model in self.df['model'].unique()
        ]
    
    def _process_results(self, results: dict) -> pd.DataFrame:
        """Convert benchmark results to DataFrame for visualization"""
        records = []
        for model_name, model_results in results.items():
            for metric_name, value in model_results['metrics'].items():
                records.append({
                    'model': model_name,
                    'metric_name': metric_name,
                    'metric_value': value
                })
        return pd.DataFrame(records)
    
    def run(self, port: int = 8050, debug: bool = True):
        """Run the dashboard server"""
        self.app.run_server(debug=debug, port=port) 