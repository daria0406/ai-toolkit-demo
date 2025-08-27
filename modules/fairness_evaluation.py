import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

class FairnessEvaluator:
    """
    A comprehensive fairness evaluation module for ML models.
    Implements multiple fairness metrics and visualization tools.
    """
    
    def __init__(self):
        self.fairness_thresholds = {
            'statistical_parity': 0.1,
            'equalized_odds': 0.1,
            'calibration': 0.1
        }
    
    def calculate_all_metrics(self, y_true, y_pred, y_proba, protected_attribute):
        """
        Calculate comprehensive fairness metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            protected_attribute: Protected attribute values
            
        Returns:
            dict: Dictionary containing all fairness metrics
        """
        try:
            metrics = {
                'group_fairness': self._calculate_group_fairness(y_true, y_pred, protected_attribute),
                'predictive_parity': self._calculate_predictive_parity(y_true, y_pred, protected_attribute),
                'individual_fairness': self._calculate_individual_fairness(y_proba, protected_attribute),
                'calibration': self._calculate_calibration(y_true, y_proba, protected_attribute)
            }
            
            return metrics
            
        except Exception as e:
            raise Exception(f"Error calculating fairness metrics: {str(e)}")
    
    def _calculate_group_fairness(self, y_true, y_pred, protected_attribute):
        """Calculate group fairness metrics."""
        groups = np.unique(protected_attribute)
        if len(groups) != 2:
            raise ValueError("Currently supports binary protected attributes only")
        
        group_0_mask = protected_attribute == groups[0]
        group_1_mask = protected_attribute == groups[1]
        
        # Selection rates
        selection_rate_0 = np.mean(y_pred[group_0_mask])
        selection_rate_1 = np.mean(y_pred[group_1_mask])
        
        # True positive rates
        tpr_0 = self._calculate_tpr(y_true[group_0_mask], y_pred[group_0_mask])
        tpr_1 = self._calculate_tpr(y_true[group_1_mask], y_pred[group_1_mask])
        
        # False positive rates
        fpr_0 = self._calculate_fpr(y_true[group_0_mask], y_pred[group_0_mask])
        fpr_1 = self._calculate_fpr(y_true[group_1_mask], y_pred[group_1_mask])
        
        return {
            'demographic_parity': abs(selection_rate_1 - selection_rate_0),
            'equalized_odds_tpr': abs(tpr_1 - tpr_0) if not (np.isnan(tpr_0) or np.isnan(tpr_1)) else np.nan,
            'equalized_odds_fpr': abs(fpr_1 - fpr_0) if not (np.isnan(fpr_0) or np.isnan(fpr_1)) else np.nan,
            'disparate_impact': (selection_rate_1 / selection_rate_0) if selection_rate_0 > 0 else np.nan
        }
    
    def _calculate_predictive_parity(self, y_true, y_pred, protected_attribute):
        """Calculate predictive parity metrics."""
        groups = np.unique(protected_attribute)
        group_0_mask = protected_attribute == groups[0]
        group_1_mask = protected_attribute == groups[1]
        
        # Positive predictive value (precision)
        ppv_0 = self._calculate_ppv(y_true[group_0_mask], y_pred[group_0_mask])
        ppv_1 = self._calculate_ppv(y_true[group_1_mask], y_pred[group_1_mask])
        
        # Negative predictive value
        npv_0 = self._calculate_npv(y_true[group_0_mask], y_pred[group_0_mask])
        npv_1 = self._calculate_npv(y_true[group_1_mask], y_pred[group_1_mask])
        
        return {
            'predictive_parity_positive': abs(ppv_1 - ppv_0) if not (np.isnan(ppv_0) or np.isnan(ppv_1)) else np.nan,
            'predictive_parity_negative': abs(npv_1 - npv_0) if not (np.isnan(npv_0) or np.isnan(npv_1)) else np.nan
        }
    
    def _calculate_individual_fairness(self, y_proba, protected_attribute):
        """Calculate individual fairness metrics."""
        # Simplified individual fairness: variance in predictions within groups
        groups = np.unique(protected_attribute)
        group_variances = []
        
        for group in groups:
            group_mask = protected_attribute == group
            if np.sum(group_mask) > 1:
                group_var = np.var(y_proba[group_mask])
                group_variances.append(group_var)
        
        return {
            'prediction_variance_ratio': max(group_variances) / min(group_variances) if len(group_variances) > 1 and min(group_variances) > 0 else 1.0
        }
    
    def _calculate_calibration(self, y_true, y_proba, protected_attribute):
        """Calculate calibration metrics."""
        groups = np.unique(protected_attribute)
        calibration_errors = []
        
        for group in groups:
            group_mask = protected_attribute == group
            if np.sum(group_mask) > 10:  # Need sufficient samples
                # Bin predictions and calculate calibration error
                bins = np.linspace(0, 1, 11)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                group_proba = y_proba[group_mask]
                group_true = y_true[group_mask]
                
                calibration_error = 0
                for i in range(len(bins) - 1):
                    bin_mask = (group_proba >= bins[i]) & (group_proba < bins[i + 1])
                    if np.sum(bin_mask) > 0:
                        bin_accuracy = np.mean(group_true[bin_mask])
                        bin_confidence = np.mean(group_proba[bin_mask])
                        calibration_error += abs(bin_accuracy - bin_confidence) * np.sum(bin_mask)
                
                calibration_error /= len(group_proba)
                calibration_errors.append(calibration_error)
        
        return {
            'calibration_difference': abs(calibration_errors[1] - calibration_errors[0]) if len(calibration_errors) == 2 else 0
        }
    
    def _calculate_tpr(self, y_true, y_pred):
        """Calculate True Positive Rate."""
        if len(y_true) == 0:
            return np.nan
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        return tp / (tp + fn) if (tp + fn) > 0 else np.nan
    
    def _calculate_fpr(self, y_true, y_pred):
        """Calculate False Positive Rate."""
        if len(y_true) == 0:
            return np.nan
        
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        return fp / (fp + tn) if (fp + tn) > 0 else np.nan
    
    def _calculate_ppv(self, y_true, y_pred):
        """Calculate Positive Predictive Value (Precision)."""
        if len(y_true) == 0:
            return np.nan
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        return tp / (tp + fp) if (tp + fp) > 0 else np.nan
    
    def _calculate_npv(self, y_true, y_pred):
        """Calculate Negative Predictive Value."""
        if len(y_true) == 0:
            return np.nan
        
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        return tn / (tn + fn) if (tn + fn) > 0 else np.nan
    
    def create_fairness_dashboard(self, y_true, y_pred, y_proba, protected_attribute):
        """
        Create comprehensive fairness dashboard.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            y_proba: Prediction probabilities
            protected_attribute: Protected attribute values
            
        Returns:
            plotly.graph_objects.Figure: Fairness dashboard
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    'ROC Curves by Group',
                    'Precision-Recall Curves by Group',
                    'Calibration Plot',
                    'Confusion Matrices',
                    'Prediction Distribution',
                    'Fairness Metrics Heatmap'
                ],
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "heatmap"}],
                       [{"type": "histogram"}, {"type": "heatmap"}]]
            )
            
            groups = np.unique(protected_attribute)
            colors = ['blue', 'red']
            
            # ROC curves
            for i, group in enumerate(groups):
                group_mask = protected_attribute == group
                if np.sum(group_mask) > 0 and len(np.unique(y_true[group_mask])) > 1:
                    fpr, tpr, _ = roc_curve(y_true[group_mask], y_proba[group_mask])
                    auc_score = auc(fpr, tpr)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=fpr, y=tpr,
                            mode='lines',
                            name=f'Group {group} (AUC={auc_score:.3f})',
                            line=dict(color=colors[i])
                        ),
                        row=1, col=1
                    )
            
            # Add diagonal line for ROC
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Precision-Recall curves
            for i, group in enumerate(groups):
                group_mask = protected_attribute == group
                if np.sum(group_mask) > 0 and len(np.unique(y_true[group_mask])) > 1:
                    precision, recall, _ = precision_recall_curve(y_true[group_mask], y_proba[group_mask])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=recall, y=precision,
                            mode='lines',
                            name=f'Group {group} PR',
                            line=dict(color=colors[i])
                        ),
                        row=1, col=2
                    )
            
            # Calibration plot
            for i, group in enumerate(groups):
                group_mask = protected_attribute == group
                if np.sum(group_mask) > 10:
                    # Create calibration bins
                    bins = np.linspace(0, 1, 11)
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    
                    group_proba = y_proba[group_mask]
                    group_true = y_true[group_mask]
                    
                    bin_accuracies = []
                    for j in range(len(bins) - 1):
                        bin_mask = (group_proba >= bins[j]) & (group_proba < bins[j + 1])
                        if np.sum(bin_mask) > 0:
                            bin_accuracy = np.mean(group_true[bin_mask])
                            bin_accuracies.append(bin_accuracy)
                        else:
                            bin_accuracies.append(bin_centers[j])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=bin_centers, y=bin_accuracies,
                            mode='markers+lines',
                            name=f'Group {group} Calibration',
                            marker=dict(color=colors[i])
                        ),
                        row=2, col=1
                    )
            
            # Perfect calibration line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    name='Perfect Calibration'
                ),
                row=2, col=1
            )
            
            # Confusion matrices
            for i, group in enumerate(groups):
                group_mask = protected_attribute == group
                if np.sum(group_mask) > 0:
                    cm = confusion_matrix(y_true[group_mask], y_pred[group_mask])
                    
                    fig.add_trace(
                        go.Heatmap(
                            z=cm,
                            colorscale='Blues',
                            showscale=(i == 0),
                            text=cm,
                            texttemplate="%{text}",
                            name=f'Group {group} CM'
                        ),
                        row=2, col=2
                    )
            
            # Prediction distributions
            for i, group in enumerate(groups):
                group_mask = protected_attribute == group
                fig.add_trace(
                    go.Histogram(
                        x=y_proba[group_mask],
                        name=f'Group {group} Predictions',
                        opacity=0.7,
                        marker_color=colors[i]
                    ),
                    row=3, col=1
                )
            
            # Fairness metrics heatmap
            metrics = self.calculate_all_metrics(y_true, y_pred, y_proba, protected_attribute)
            
            # Flatten metrics for heatmap
            metric_names = []
            metric_values = []
            
            for category, cat_metrics in metrics.items():
                for metric_name, value in cat_metrics.items():
                    if not np.isnan(value):
                        metric_names.append(f"{category}:{metric_name}")
                        metric_values.append(value)
            
            if metric_names:
                # Reshape for heatmap
                n_metrics = len(metric_names)
                heatmap_data = np.array(metric_values).reshape(1, -1)
                
                fig.add_trace(
                    go.Heatmap(
                        z=heatmap_data,
                        x=metric_names,
                        colorscale='RdYlBu_r',
                        showscale=True,
                        text=np.round(heatmap_data, 3),
                        texttemplate="%{text}"
                    ),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                height=1200,
                title_text="Comprehensive Fairness Dashboard",
                showlegend=True
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
            fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
            fig.update_xaxes(title_text="Recall", row=1, col=2)
            fig.update_yaxes(title_text="Precision", row=1, col=2)
            fig.update_xaxes(title_text="Mean Predicted Probability", row=2, col=1)
            fig.update_yaxes(title_text="Fraction of Positives", row=2, col=1)
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error creating fairness dashboard: {str(e)}")
    
    def generate_recommendations(self, metrics):
        """
        Generate fairness improvement recommendations.
        
        Args:
            metrics: Dictionary of calculated fairness metrics
            
        Returns:
            list: List of recommendation strings
        """
        recommendations = []
        
        try:
            # Check demographic parity
            demo_parity = metrics['group_fairness'].get('demographic_parity', 0)
            if demo_parity > 0.1:
                recommendations.append(
                    f"🚨 Demographic parity violation detected ({demo_parity:.3f}). "
                    "Consider post-processing techniques or threshold optimization."
                )
            
            # Check equalized odds
            eq_odds_tpr = metrics['group_fairness'].get('equalized_odds_tpr', 0)
            if not np.isnan(eq_odds_tpr) and eq_odds_tpr > 0.1:
                recommendations.append(
                    f"⚠️ Equalized odds violation in TPR ({eq_odds_tpr:.3f}). "
                    "Model shows different error rates across groups."
                )
            
            # Check calibration
            calibration_diff = metrics['calibration'].get('calibration_difference', 0)
            if calibration_diff > 0.1:
                recommendations.append(
                    f"📊 Calibration difference detected ({calibration_diff:.3f}). "
                    "Consider calibration techniques like Platt scaling."
                )
            
            # Check disparate impact
            disparate_impact = metrics['group_fairness'].get('disparate_impact', 1.0)
            if not np.isnan(disparate_impact) and disparate_impact < 0.8:
                recommendations.append(
                    f"⚖️ Disparate impact below 0.8 ({disparate_impact:.3f}). "
                    "This may indicate discriminatory impact."
                )
            
            # General recommendations
            if not recommendations:
                recommendations.append("✅ No major fairness violations detected. Continue monitoring.")
            else:
                recommendations.append(
                    "💡 Consider implementing fairness-aware algorithms or bias mitigation techniques."
                )
            
            return recommendations
            
        except Exception as e:
            return [f"Error generating recommendations: {str(e)}"]
