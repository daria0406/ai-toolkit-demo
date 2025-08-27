import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    Data loading utilities for AI safety demonstration datasets.
    Provides access to common fairness evaluation datasets.
    """
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
    
    def load_adult_dataset(self):
        """
        Load the Adult Income dataset from OpenML.
        
        Returns:
            pandas.DataFrame: Processed Adult dataset
        """
        try:
            # Fetch Adult dataset from OpenML
            adult = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
            
            # Get features and target
            X = adult.data
            y = adult.target
            
            # Combine features and target
            data = X.copy()
            data['income'] = y
            
            # Basic preprocessing
            data = self._preprocess_adult_dataset(data)
            
            return data
            
        except Exception as e:
            # Fallback: create a synthetic dataset with similar structure
            return self._create_synthetic_adult_dataset()
    
    def _preprocess_adult_dataset(self, data):
        """Preprocess the Adult dataset."""
        try:
            # Handle missing values
            data = data.replace('?', np.nan)
            
            # Drop rows with missing values
            data = data.dropna()
            
            # Encode categorical variables
            categorical_columns = data.select_dtypes(include=['object']).columns
            categorical_columns = categorical_columns.drop('income')  # Exclude target
            
            for col in categorical_columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    data[col] = self.encoders[col].fit_transform(data[col])
                else:
                    data[col] = self.encoders[col].transform(data[col])
            
            # Encode target variable
            data['income'] = (data['income'] == '>50K').astype(int)
            
            # Create protected attribute (race-based, simplified)
            if 'race' in data.columns:
                data['protected_race'] = (data['race'] == 4).astype(int)  # Assuming 4 represents White
            else:
                # If race column not available, use age as proxy
                data['protected_race'] = (data['age'] >= data['age'].median()).astype(int)
            
            return data
            
        except Exception as e:
            raise Exception(f"Error preprocessing Adult dataset: {str(e)}")
    
    def _create_synthetic_adult_dataset(self):
        """Create synthetic Adult-like dataset."""
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'age': np.random.normal(40, 12, n_samples).astype(int),
            'workclass': np.random.randint(0, 8, n_samples),
            'education': np.random.randint(0, 16, n_samples),
            'marital_status': np.random.randint(0, 7, n_samples),
            'occupation': np.random.randint(0, 14, n_samples),
            'relationship': np.random.randint(0, 6, n_samples),
            'race': np.random.randint(0, 5, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'hours_per_week': np.random.normal(40, 12, n_samples).astype(int)
        })
        
        # Create correlated income based on features
        income_score = (
            0.3 * data['age'] / 100 +
            0.2 * data['education'] / 16 +
            0.2 * data['hours_per_week'] / 80 +
            0.1 * data['sex'] +
            np.random.normal(0, 0.1, n_samples)
        )
        
        data['income'] = (income_score > 0.3).astype(int)
        data['protected_race'] = (data['race'] == 0).astype(int)
        
        return data
    
    def load_german_credit_dataset(self):
        """
        Load the German Credit dataset from OpenML.
        
        Returns:
            pandas.DataFrame: Processed German Credit dataset
        """
        try:
            # Fetch German Credit dataset from OpenML
            german = fetch_openml(name='credit-g', version=1, as_frame=True, parser='auto')
            
            # Get features and target
            X = german.data
            y = german.target
            
            # Combine features and target
            data = X.copy()
            data['credit_risk'] = y
            
            # Basic preprocessing
            data = self._preprocess_german_credit_dataset(data)
            
            return data
            
        except Exception as e:
            # Fallback: create a synthetic dataset
            return self._create_synthetic_german_credit_dataset()
    
    def _preprocess_german_credit_dataset(self, data):
        """Preprocess the German Credit dataset."""
        try:
            # Encode categorical variables
            categorical_columns = data.select_dtypes(include=['object']).columns
            categorical_columns = categorical_columns.drop('credit_risk')  # Exclude target
            
            for col in categorical_columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    data[col] = self.encoders[col].fit_transform(data[col])
                else:
                    data[col] = self.encoders[col].transform(data[col])
            
            # Convert target to binary (good credit = 1, bad credit = 0)
            data['credit_risk'] = (data['credit_risk'] == 'good').astype(int)
            
            # Create protected attribute based on age
            if 'age' in data.columns:
                data['protected_age'] = (data['age'] >= 35).astype(int)
            else:
                # If age not available, create based on other features
                data['protected_age'] = np.random.randint(0, 2, len(data))
            
            return data
            
        except Exception as e:
            raise Exception(f"Error preprocessing German Credit dataset: {str(e)}")
    
    def _create_synthetic_german_credit_dataset(self):
        """Create synthetic German Credit-like dataset."""
        np.random.seed(42)
        n_samples = 800
        
        data = pd.DataFrame({
            'duration': np.random.randint(6, 72, n_samples),
            'credit_amount': np.random.lognormal(8, 1, n_samples).astype(int),
            'installment_commitment': np.random.randint(1, 5, n_samples),
            'residence_since': np.random.randint(1, 5, n_samples),
            'age': np.random.randint(18, 80, n_samples),
            'existing_credits': np.random.randint(1, 5, n_samples),
            'num_dependents': np.random.randint(1, 3, n_samples),
            'job': np.random.randint(0, 4, n_samples),
            'housing': np.random.randint(0, 3, n_samples),
            'savings_status': np.random.randint(0, 5, n_samples)
        })
        
        # Create correlated credit risk
        risk_score = (
            -0.01 * data['duration'] +
            -0.00001 * data['credit_amount'] +
            0.1 * (data['age'] > 35) +
            0.1 * data['savings_status'] / 5 +
            np.random.normal(0, 0.2, n_samples)
        )
        
        data['credit_risk'] = (risk_score > 0).astype(int)
        data['protected_age'] = (data['age'] >= 35).astype(int)
        
        return data
    
    def prepare_ml_data(self, data, target_column, protected_column, test_size=0.3):
        """
        Prepare data for machine learning.
        
        Args:
            data: Input dataframe
            target_column: Name of target column
            protected_column: Name of protected attribute column
            test_size: Fraction of data for testing
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, protected_train, protected_test)
        """
        try:
            # Separate features, target, and protected attribute
            X = data.drop([target_column, protected_column], axis=1)
            y = data[target_column]
            protected = data[protected_column]
            
            # Split the data
            X_train, X_test, y_train, y_test, protected_train, protected_test = train_test_split(
                X, y, protected, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scale numerical features
            numerical_columns = X.select_dtypes(include=[np.number]).columns
            if len(numerical_columns) > 0:
                scaler = StandardScaler()
                X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
                X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
                self.scalers['features'] = scaler
            
            return X_train, X_test, y_train, y_test, protected_train, protected_test
            
        except Exception as e:
            raise Exception(f"Error preparing ML data: {str(e)}")
    
    def get_dataset_info(self, data):
        """
        Get comprehensive information about the dataset.
        
        Args:
            data: Input dataframe
            
        Returns:
            dict: Dataset information
        """
        try:
            info = {
                'shape': data.shape,
                'columns': data.columns.tolist(),
                'dtypes': data.dtypes.to_dict(),
                'missing_values': data.isnull().sum().to_dict(),
                'numerical_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': data.select_dtypes(include=['object']).columns.tolist(),
                'summary_stats': data.describe().to_dict()
            }
            
            return info
            
        except Exception as e:
            raise Exception(f"Error getting dataset info: {str(e)}")
