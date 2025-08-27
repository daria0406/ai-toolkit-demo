import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from modules.data_loader import DataLoader

def load_sample_model(data, dataset_name="Adult Income Dataset"):
    """
    Train a sample model for demonstration purposes.
    
    Args:
        data: Input dataset
        dataset_name: Name of the dataset
        
    Returns:
        tuple: (model, X_test, y_test, protected_attribute)
    """
    try:
        data_loader = DataLoader()
        
        # Determine target and protected columns based on dataset
        if "Adult" in dataset_name:
            target_col = 'income'
            protected_col = 'protected_race'
        elif "German" in dataset_name:
            target_col = 'credit_risk'
            protected_col = 'protected_age'
        else:
            # For custom datasets, assume last column is target
            target_col = data.columns[-1]
            # Try to find a protected attribute or create one
            protected_col = None
            for col in data.columns:
                if 'protected' in col.lower():
                    protected_col = col
                    break
            if protected_col is None:
                # Create a synthetic protected attribute
                data['protected_synthetic'] = np.random.randint(0, 2, len(data))
                protected_col = 'protected_synthetic'
        
        # Prepare data
        X_train, X_test, y_train, y_test, protected_train, protected_test = data_loader.prepare_ml_data(
            data, target_col, protected_col
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully! Accuracy: {accuracy:.3f}")
        
        return model, X_test, y_test, protected_test
        
    except Exception as e:
        raise Exception(f"Error training sample model: {str(e)}")

def calculate_model_metrics(model, X_test, y_test):
    """
    Calculate comprehensive model performance metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary of metrics
    """
    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score
        )
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'average_precision': average_precision_score(y_test, y_proba)
        }
        
        return metrics
        
    except Exception as e:
        raise Exception(f"Error calculating model metrics: {str(e)}")

def generate_synthetic_biased_data(n_samples=1000, bias_strength=0.3):
    """
    Generate synthetic dataset with controllable bias for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        bias_strength: Strength of bias (0 = no bias, 1 = maximum bias)
        
    Returns:
        pandas.DataFrame: Synthetic biased dataset
    """
    try:
        np.random.seed(42)
        
        # Generate features
        data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
            'feature_3': np.random.normal(0, 1, n_samples),
            'feature_4': np.random.normal(0, 1, n_samples),
            'protected_attr': np.random.binomial(1, 0.3, n_samples)  # 30% minority group
        })
        
        # Generate target with bias
        # Base probability from features
        base_prob = 1 / (1 + np.exp(-(
            0.5 * data['feature_1'] + 
            0.3 * data['feature_2'] + 
            0.2 * data['feature_3'] + 
            0.1 * data['feature_4']
        )))
        
        # Add bias based on protected attribute
        biased_prob = base_prob.copy()
        minority_mask = data['protected_attr'] == 1
        
        # Reduce probability for minority group
        biased_prob[minority_mask] = biased_prob[minority_mask] * (1 - bias_strength)
        
        # Generate target
        data['target'] = np.random.binomial(1, biased_prob, n_samples)
        
        return data
        
    except Exception as e:
        raise Exception(f"Error generating synthetic biased data: {str(e)}")

def validate_fairness_metrics(metrics_dict, thresholds=None):
    """
    Validate fairness metrics against standard thresholds.
    
    Args:
        metrics_dict: Dictionary of calculated fairness metrics
        thresholds: Custom thresholds (optional)
        
    Returns:
        dict: Validation results
    """
    if thresholds is None:
        thresholds = {
            'demographic_parity': 0.1,
            'equalized_odds': 0.1,
            'disparate_impact': 0.8,
            'calibration_difference': 0.1
        }
    
    validation_results = {}
    
    try:
        # Check each metric category
        for category, cat_metrics in metrics_dict.items():
            if isinstance(cat_metrics, dict):
                for metric_name, value in cat_metrics.items():
                    if not np.isnan(value):
                        # Check against thresholds
                        if metric_name in thresholds:
                            threshold = thresholds[metric_name]
                            
                            if metric_name == 'disparate_impact':
                                is_fair = value >= threshold
                            else:
                                is_fair = abs(value) <= threshold
                            
                            validation_results[f"{category}_{metric_name}"] = {
                                'value': value,
                                'threshold': threshold,
                                'is_fair': is_fair,
                                'status': 'PASS' if is_fair else 'FAIL'
                            }
        
        return validation_results
        
    except Exception as e:
        raise Exception(f"Error validating fairness metrics: {str(e)}")

def create_bias_mitigation_recommendations(validation_results):
    """
    Generate specific recommendations for bias mitigation based on validation results.
    
    Args:
        validation_results: Results from validate_fairness_metrics
        
    Returns:
        list: List of specific recommendations
    """
    recommendations = []
    
    try:
        failed_metrics = [k for k, v in validation_results.items() if v['status'] == 'FAIL']
        
        if not failed_metrics:
            recommendations.append("✅ All fairness metrics passed validation!")
            return recommendations
        
        # Specific recommendations based on failed metrics
        for failed_metric in failed_metrics:
            if 'demographic_parity' in failed_metric:
                recommendations.append(
                    "🎯 Demographic Parity Issue: Consider post-processing techniques like "
                    "threshold optimization or re-sampling methods to balance selection rates."
                )
            
            elif 'equalized_odds' in failed_metric:
                recommendations.append(
                    "⚖️ Equalized Odds Issue: Implement fairness-aware algorithms like "
                    "adversarial debiasing or constraint-based optimization."
                )
            
            elif 'disparate_impact' in failed_metric:
                recommendations.append(
                    "📊 Disparate Impact Issue: Review feature selection and consider "
                    "removing or transforming features that may encode bias."
                )
            
            elif 'calibration' in failed_metric:
                recommendations.append(
                    "🎯 Calibration Issue: Apply calibration techniques like Platt scaling "
                    "or isotonic regression separately for each group."
                )
        
        # General recommendations
        recommendations.extend([
            "💡 Consider using fairness-aware ML algorithms during training.",
            "🔍 Implement continuous monitoring of fairness metrics in production.",
            "📚 Review data collection processes to identify sources of bias.",
            "🤝 Engage domain experts and affected communities in the evaluation process."
        ])
        
        return recommendations
        
    except Exception as e:
        return [f"Error generating recommendations: {str(e)}"]

def export_analysis_results(results_dict, filename="ai_safety_analysis.json"):
    """
    Export analysis results to JSON file.
    
    Args:
        results_dict: Dictionary containing analysis results
        filename: Output filename
        
    Returns:
        str: Success message
    """
    try:
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        # Convert results
        converted_results = convert_numpy(results_dict)
        
        # Add metadata
        import datetime
        converted_results['metadata'] = {
            'timestamp': datetime.datetime.now().isoformat(),
            'analysis_type': 'AI Safety Toolkit',
            'version': '1.0'
        }
        
        # Export to JSON
        with open(filename, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        return f"Analysis results exported successfully to {filename}"
        
    except Exception as e:
        raise Exception(f"Error exporting analysis results: {str(e)}")
