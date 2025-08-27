import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance

# Handle SHAP import gracefully
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

class TransparencyAnalyzer:
    """
    A comprehensive model transparency and interpretability module.
    Implements LIME, SHAP, and other interpretability techniques.
    """
    
    def __init__(self):
        self.lime_explainer = None
        self.shap_explainer = None
        
    def plot_feature_importance(self, model, feature_names):
        """
        Plot feature importance from tree-based models.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            
        Returns:
            plotly.graph_objects.Figure: Feature importance plot
        """
        try:
            if not hasattr(model, 'feature_importances_'):
                raise ValueError("Model does not have feature_importances_ attribute")
            
            importances = model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            fig = go.Figure(data=go.Bar(
                x=[feature_names[i] for i in indices],
                y=[importances[i] for i in indices],
                marker_color='skyblue'
            ))
            
            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Features",
                yaxis_title="Importance",
                xaxis_tickangle=-45
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error plotting feature importance: {str(e)}")
    
    def explain_with_lime(self, model, X_test, instance_idx, num_features=10):
        """
        Generate LIME explanation for a specific instance.
        
        Args:
            model: Trained model
            X_test: Test data
            instance_idx: Index of instance to explain
            num_features: Number of features to show
            
        Returns:
            plotly.graph_objects.Figure: LIME explanation plot
        """
        try:
            # Convert to numpy array if needed
            if hasattr(X_test, 'values'):
                X_array = X_test.values
                feature_names = X_test.columns.tolist()
            else:
                X_array = np.array(X_test)
                feature_names = [f'Feature_{i}' for i in range(X_array.shape[1])]
            
            # Initialize LIME explainer if not exists
            if self.lime_explainer is None:
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_array,
                    feature_names=feature_names,
                    class_names=['Negative', 'Positive'],
                    mode='classification'
                )
            
            # Get explanation
            instance = X_array[instance_idx]
            explanation = self.lime_explainer.explain_instance(
                instance,
                model.predict_proba,
                num_features=num_features
            )
            
            # Extract explanation data
            exp_data = explanation.as_list()
            features = [item[0] for item in exp_data]
            values = [item[1] for item in exp_data]
            
            # Create plot
            colors = ['red' if v < 0 else 'green' for v in values]
            
            fig = go.Figure(data=go.Bar(
                x=values,
                y=features,
                orientation='h',
                marker_color=colors
            ))
            
            fig.update_layout(
                title=f"LIME Explanation - Instance {instance_idx}",
                xaxis_title="Feature Impact",
                yaxis_title="Features",
                height=max(400, len(features) * 30)
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error generating LIME explanation: {str(e)}")
    
    def explain_with_shap(self, model, X_test, max_samples=100):
        """
        Generate SHAP explanations.
        
        Args:
            model: Trained model
            X_test: Test data
            max_samples: Maximum number of samples to use
            
        Returns:
            plotly.graph_objects.Figure: SHAP summary plot
        """
        try:
            if not SHAP_AVAILABLE:
                raise Exception("SHAP library is not available. Please install with: pip install shap")
            
            # Convert to numpy array if needed
            if hasattr(X_test, 'values'):
                X_array = X_test.values
                feature_names = X_test.columns.tolist()
            else:
                X_array = np.array(X_test)
                feature_names = [f'Feature_{i}' for i in range(X_array.shape[1])]
            
            # Limit samples for performance
            if X_array.shape[0] > max_samples:
                X_sample = X_array[:max_samples]
            else:
                X_sample = X_array
            
            # Initialize SHAP explainer
            try:
                # Try TreeExplainer first (for tree-based models)
                self.shap_explainer = shap.TreeExplainer(model)
                shap_values = self.shap_explainer.shap_values(X_sample)
                
                # For binary classification, get positive class values
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
                    
            except Exception:
                # Fallback to KernelExplainer
                self.shap_explainer = shap.KernelExplainer(
                    model.predict_proba, 
                    X_sample[:min(50, len(X_sample))]  # Use subset for background
                )
                shap_values = self.shap_explainer.shap_values(X_sample)
                
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
            
            # Calculate mean absolute SHAP values for each feature
            mean_shap_values = np.mean(np.abs(shap_values), axis=0)
            
            # Sort features by importance
            indices = np.argsort(mean_shap_values)[::-1]
            
            # Create summary plot
            fig = go.Figure()
            
            # Add bar plot for mean importance
            fig.add_trace(go.Bar(
                x=[feature_names[i] for i in indices],
                y=[mean_shap_values[i] for i in indices],
                name='Mean |SHAP Value|',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="SHAP Feature Importance Summary",
                xaxis_title="Features",
                yaxis_title="Mean |SHAP Value|",
                xaxis_tickangle=-45,
                height=500
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error generating SHAP explanation: {str(e)}")
    
    def plot_decision_boundary_2d(self, model, X_test, y_test, feature1_idx, feature2_idx):
        """
        Plot 2D decision boundary for two selected features.
        
        Args:
            model: Trained model
            X_test: Test data
            y_test: Test labels
            feature1_idx: Index of first feature
            feature2_idx: Index of second feature
            
        Returns:
            plotly.graph_objects.Figure: Decision boundary plot
        """
        try:
            # Convert to numpy array if needed
            if hasattr(X_test, 'values'):
                X_array = X_test.values
                feature_names = X_test.columns.tolist()
            else:
                X_array = np.array(X_test)
                feature_names = [f'Feature_{i}' for i in range(X_array.shape[1])]
            
            # Extract the two features
            X_2d = X_array[:, [feature1_idx, feature2_idx]]
            
            # Create a mesh grid
            h = 0.02  # Step size in the mesh
            x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
            y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            
            # Create feature matrix for prediction
            # Use mean values for other features
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            
            # For prediction, we need to create full feature vectors
            # Use mean values for features not being visualized
            full_features = np.zeros((mesh_points.shape[0], X_array.shape[1]))
            full_features[:, feature1_idx] = mesh_points[:, 0]
            full_features[:, feature2_idx] = mesh_points[:, 1]
            
            # Fill other features with mean values
            for i in range(X_array.shape[1]):
                if i not in [feature1_idx, feature2_idx]:
                    full_features[:, i] = np.mean(X_array[:, i])
            
            # Get predictions
            Z = model.predict_proba(full_features)[:, 1]
            Z = Z.reshape(xx.shape)
            
            # Create the plot
            fig = go.Figure()
            
            # Add contour plot for decision boundary
            fig.add_trace(go.Contour(
                x=np.arange(x_min, x_max, h),
                y=np.arange(y_min, y_max, h),
                z=Z,
                colorscale='RdYlBu',
                opacity=0.6,
                showscale=True,
                name='Decision Boundary'
            ))
            
            # Add scatter plot for data points
            colors = ['blue' if label == 0 else 'red' for label in y_test]
            
            fig.add_trace(go.Scatter(
                x=X_2d[:, 0],
                y=X_2d[:, 1],
                mode='markers',
                marker=dict(
                    color=colors,
                    size=8,
                    opacity=0.8,
                    line=dict(width=1, color='black')
                ),
                name='Data Points'
            ))
            
            fig.update_layout(
                title=f"Decision Boundary: {feature_names[feature1_idx]} vs {feature_names[feature2_idx]}",
                xaxis_title=feature_names[feature1_idx],
                yaxis_title=feature_names[feature2_idx],
                height=600,
                width=800
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error plotting decision boundary: {str(e)}")
    
    def analyze_model_behavior(self, model, X_test, feature_names=None):
        """
        Comprehensive model behavior analysis.
        
        Args:
            model: Trained model
            X_test: Test data
            feature_names: List of feature names
            
        Returns:
            dict: Dictionary containing various analysis results
        """
        try:
            if feature_names is None:
                if hasattr(X_test, 'columns'):
                    feature_names = X_test.columns.tolist()
                else:
                    feature_names = [f'Feature_{i}' for i in range(X_test.shape[1])]
            
            analysis = {}
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                analysis['feature_importance'] = dict(zip(feature_names, model.feature_importances_))
            
            # Permutation importance
            try:
                # Use a subset for performance
                X_subset = X_test[:min(100, len(X_test))]
                y_subset = model.predict(X_subset)  # Use predictions as "true" labels
                
                perm_importance = permutation_importance(model, X_subset, y_subset, n_repeats=5, random_state=42)
                analysis['permutation_importance'] = dict(zip(feature_names, perm_importance.importances_mean))
            except Exception as e:
                analysis['permutation_importance_error'] = str(e)
            
            # Prediction statistics
            predictions = model.predict_proba(X_test)[:, 1]
            analysis['prediction_stats'] = {
                'mean': np.mean(predictions),
                'std': np.std(predictions),
                'min': np.min(predictions),
                'max': np.max(predictions),
                'quartiles': np.percentile(predictions, [25, 50, 75]).tolist()
            }
            
            return analysis
            
        except Exception as e:
            raise Exception(f"Error in model behavior analysis: {str(e)}")
    
    def create_interpretability_dashboard(self, model, X_test, y_test, instance_idx=0):
        """
        Create comprehensive interpretability dashboard.
        
        Args:
            model: Trained model
            X_test: Test data
            y_test: Test labels
            instance_idx: Instance to explain
            
        Returns:
            plotly.graph_objects.Figure: Interpretability dashboard
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Feature Importance',
                    'LIME Explanation',
                    'Prediction Distribution',
                    'Model Confidence'
                ]
            )
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_names = X_test.columns.tolist() if hasattr(X_test, 'columns') else [f'Feature_{i}' for i in range(X_test.shape[1])]
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:10]  # Top 10
                
                fig.add_trace(
                    go.Bar(
                        x=[feature_names[i] for i in indices],
                        y=[importances[i] for i in indices],
                        name='Feature Importance'
                    ),
                    row=1, col=1
                )
            
            # Prediction distribution
            predictions = model.predict_proba(X_test)[:, 1]
            fig.add_trace(
                go.Histogram(
                    x=predictions,
                    nbinsx=30,
                    name='Prediction Distribution'
                ),
                row=2, col=1
            )
            
            # Model confidence (entropy-based)
            proba = model.predict_proba(X_test)
            entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
            fig.add_trace(
                go.Histogram(
                    x=entropy,
                    nbinsx=30,
                    name='Prediction Entropy'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                title_text="Model Interpretability Dashboard",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error creating interpretability dashboard: {str(e)}")
