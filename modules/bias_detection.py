import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, accuracy_score

class BiasDetector:
    """
    A class for detecting bias in machine learning models.
    Implements various fairness metrics and bias detection algorithms.
    """
    
    def __init__(self):
        self.bias_threshold = 0.8  # Standard threshold for disparate impact
        
    def calculate_disparate_impact(self, y_pred, protected_attribute):
        """
        Calculate disparate impact ratio.
        
        Args:
            y_pred: Model predictions
            protected_attribute: Binary protected attribute (0/1)
            
        Returns:
            float: Disparate impact ratio
        """
        try:
            # Convert to numpy arrays
            y_pred = np.array(y_pred)
            protected_attribute = np.array(protected_attribute)
            
            # Calculate positive prediction rates for each group
            privileged_group = protected_attribute == 1
            unprivileged_group = protected_attribute == 0
            
            if np.sum(privileged_group) == 0 or np.sum(unprivileged_group) == 0:
                return np.nan
            
            privileged_positive_rate = np.mean(y_pred[privileged_group])
            unprivileged_positive_rate = np.mean(y_pred[unprivileged_group])
            
            if privileged_positive_rate == 0:
                return np.inf if unprivileged_positive_rate > 0 else 1.0
            
            return unprivileged_positive_rate / privileged_positive_rate
            
        except Exception as e:
            raise Exception(f"Error calculating disparate impact: {str(e)}")
    
    def calculate_statistical_parity(self, y_pred, protected_attribute):
        """
        Calculate statistical parity difference.
        
        Args:
            y_pred: Model predictions
            protected_attribute: Binary protected attribute (0/1)
            
        Returns:
            float: Statistical parity difference
        """
        try:
            y_pred = np.array(y_pred)
            protected_attribute = np.array(protected_attribute)
            
            privileged_group = protected_attribute == 1
            unprivileged_group = protected_attribute == 0
            
            if np.sum(privileged_group) == 0 or np.sum(unprivileged_group) == 0:
                return np.nan
            
            privileged_positive_rate = np.mean(y_pred[privileged_group])
            unprivileged_positive_rate = np.mean(y_pred[unprivileged_group])
            
            return unprivileged_positive_rate - privileged_positive_rate
            
        except Exception as e:
            raise Exception(f"Error calculating statistical parity: {str(e)}")
    
    def calculate_equalized_odds(self, y_true, y_pred, protected_attribute):
        """
        Calculate equalized odds difference.
        
        Args:
            y_true: True labels
            y_pred: Model predictions
            protected_attribute: Binary protected attribute (0/1)
            
        Returns:
            float: Equalized odds difference
        """
        try:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            protected_attribute = np.array(protected_attribute)
            
            # Calculate TPR for each group
            privileged_group = protected_attribute == 1
            unprivileged_group = protected_attribute == 0
            
            # True positive rates
            privileged_tpr = self._calculate_tpr(y_true[privileged_group], y_pred[privileged_group])
            unprivileged_tpr = self._calculate_tpr(y_true[unprivileged_group], y_pred[unprivileged_group])
            
            if np.isnan(privileged_tpr) or np.isnan(unprivileged_tpr):
                return np.nan
            
            return abs(unprivileged_tpr - privileged_tpr)
            
        except Exception as e:
            raise Exception(f"Error calculating equalized odds: {str(e)}")
    
    def _calculate_tpr(self, y_true, y_pred):
        """Calculate True Positive Rate."""
        if len(y_true) == 0:
            return np.nan
        
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        actual_positives = np.sum(y_true == 1)
        
        if actual_positives == 0:
            return np.nan
        
        return true_positives / actual_positives
    
    def calculate_demographic_parity(self, y_pred, protected_attribute):
        """
        Calculate demographic parity violation.
        
        Args:
            y_pred: Model predictions
            protected_attribute: Protected attribute values
            
        Returns:
            dict: Demographic parity metrics
        """
        try:
            unique_groups = np.unique(protected_attribute)
            group_rates = {}
            
            for group in unique_groups:
                group_mask = protected_attribute == group
                if np.sum(group_mask) > 0:
                    group_rates[f'group_{group}'] = np.mean(y_pred[group_mask])
            
            # Calculate maximum difference
            rates = list(group_rates.values())
            if len(rates) > 1:
                max_diff = max(rates) - min(rates)
                return {
                    'group_rates': group_rates,
                    'max_difference': max_diff,
                    'is_fair': max_diff <= 0.1  # 10% threshold
                }
            
            return {'group_rates': group_rates, 'max_difference': 0, 'is_fair': True}
            
        except Exception as e:
            raise Exception(f"Error calculating demographic parity: {str(e)}")
    
    def plot_bias_metrics(self, y_pred, protected_attribute, y_true=None):
        """
        Create visualization of bias metrics.
        
        Args:
            y_pred: Model predictions
            protected_attribute: Protected attribute values
            y_true: True labels (optional)
            
        Returns:
            plotly.graph_objects.Figure: Bias metrics visualization
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Prediction Rates by Group',
                    'Confusion Matrix by Group',
                    'Bias Metrics Summary',
                    'Group Distribution'
                ],
                specs=[[{"type": "bar"}, {"type": "heatmap"}],
                       [{"type": "bar"}, {"type": "pie"}]]
            )
            
            # Calculate metrics
            groups = np.unique(protected_attribute)
            group_pred_rates = []
            group_names = []
            
            for group in groups:
                group_mask = protected_attribute == group
                pred_rate = np.mean(y_pred[group_mask])
                group_pred_rates.append(pred_rate)
                group_names.append(f'Group {group}')
            
            # Plot 1: Prediction rates by group
            fig.add_trace(
                go.Bar(x=group_names, y=group_pred_rates, name='Positive Prediction Rate'),
                row=1, col=1
            )
            
            # Plot 2: Confusion matrices (if y_true provided)
            if y_true is not None:
                for i, group in enumerate(groups):
                    group_mask = protected_attribute == group
                    if np.sum(group_mask) > 0:
                        cm = confusion_matrix(y_true[group_mask], y_pred[group_mask])
                        if cm.size > 0:
                            fig.add_trace(
                                go.Heatmap(
                                    z=cm,
                                    colorscale='Blues',
                                    showscale=(i == 0),
                                    text=cm,
                                    texttemplate="%{text}",
                                    textfont={"size": 12},
                                    name=f'Group {group}'
                                ),
                                row=1, col=2
                            )
            
            # Plot 3: Bias metrics
            disparate_impact = self.calculate_disparate_impact(y_pred, protected_attribute)
            stat_parity = self.calculate_statistical_parity(y_pred, protected_attribute)
            
            metrics_names = ['Disparate Impact', 'Statistical Parity Diff']
            metrics_values = [disparate_impact, abs(stat_parity)]
            colors = ['red' if disparate_impact < 0.8 else 'green',
                     'red' if abs(stat_parity) > 0.1 else 'green']
            
            fig.add_trace(
                go.Bar(
                    x=metrics_names,
                    y=metrics_values,
                    marker_color=colors,
                    name='Bias Metrics'
                ),
                row=2, col=1
            )
            
            # Plot 4: Group distribution
            group_counts = [np.sum(protected_attribute == group) for group in groups]
            fig.add_trace(
                go.Pie(
                    labels=group_names,
                    values=group_counts,
                    name='Group Distribution'
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                title_text="Bias Detection Analysis",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error creating bias visualization: {str(e)}")
    
    def generate_bias_report(self, y_true, y_pred, protected_attribute):
        """
        Generate a comprehensive bias report.
        
        Args:
            y_true: True labels
            y_pred: Model predictions
            protected_attribute: Protected attribute values
            
        Returns:
            dict: Comprehensive bias report
        """
        try:
            report = {
                'disparate_impact': self.calculate_disparate_impact(y_pred, protected_attribute),
                'statistical_parity': self.calculate_statistical_parity(y_pred, protected_attribute),
                'equalized_odds': self.calculate_equalized_odds(y_true, y_pred, protected_attribute),
                'demographic_parity': self.calculate_demographic_parity(y_pred, protected_attribute)
            }
            
            # Add interpretations
            report['interpretations'] = self._generate_interpretations(report)
            
            return report
            
        except Exception as e:
            raise Exception(f"Error generating bias report: {str(e)}")
    
    def _generate_interpretations(self, metrics):
        """Generate human-readable interpretations of bias metrics."""
        interpretations = []
        
        di = metrics.get('disparate_impact', np.nan)
        if not np.isnan(di):
            if di < 0.8:
                interpretations.append(f"Disparate impact of {di:.3f} indicates potential bias (threshold: 0.8)")
            else:
                interpretations.append(f"Disparate impact of {di:.3f} is within acceptable range")
        
        sp = metrics.get('statistical_parity', np.nan)
        if not np.isnan(sp):
            if abs(sp) > 0.1:
                interpretations.append(f"Statistical parity difference of {sp:.3f} suggests unfair treatment")
            else:
                interpretations.append(f"Statistical parity difference of {sp:.3f} is acceptable")
        
        eo = metrics.get('equalized_odds', np.nan)
        if not np.isnan(eo):
            if eo > 0.1:
                interpretations.append(f"Equalized odds difference of {eo:.3f} indicates bias in error rates")
            else:
                interpretations.append(f"Equalized odds difference of {eo:.3f} is within acceptable range")
        
        return interpretations
