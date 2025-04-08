import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import optuna
from typing import Dict, Tuple, Any
import joblib

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = float('-inf')
        
    def prepare_data(self, features: pd.DataFrame, target: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""
        X = self.scaler.fit_transform(features)
        y = target.values
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def create_xgboost_model(self, trial: optuna.Trial) -> xgb.XGBRegressor:
        """Create XGBoost model with hyperparameter optimization"""
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'objective': 'reg:squarederror'
        }
        return xgb.XGBRegressor(**param)
    
    def create_lightgbm_model(self, trial: optuna.Trial) -> lgb.LGBMRegressor:
        """Create LightGBM model with hyperparameter optimization"""
        param = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'objective': 'regression'
        }
        return lgb.LGBMRegressor(**param)
    
    def create_neural_network(self, input_dim: int) -> Sequential:
        """Create Neural Network model"""
        model = Sequential([
            Dense(256, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, 
                 X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Objective function for Optuna optimization"""
        model_type = trial.suggest_categorical('model_type', ['xgboost', 'lightgbm'])
        
        if model_type == 'xgboost':
            model = self.create_xgboost_model(trial)
            # XGBoost training without early stopping
            model.fit(X_train, y_train)
        else:
            model = self.create_lightgbm_model(trial)
            # LightGBM training with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50)]
            )
        
        y_pred = model.predict(X_val)
        score = r2_score(y_val, y_pred)
        
        if score > self.best_score:
            self.best_score = score
            self.best_model = model
            
        return score
    
    def train_models(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """Train all models and select the best one"""
        X_train, X_test, y_train, y_test = self.prepare_data(features, target)
        
        # Create and train neural network
        nn_model = self.create_neural_network(X_train.shape[1])
        nn_model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            verbose=0
        )
        self.models['neural_network'] = nn_model
        
        # Optimize tree-based models
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_test, y_test),
            n_trials=50
        )
        
        # Evaluate best model
        best_model = self.best_model
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'best_model': best_model,
            'neural_network': nn_model,
            'scaler': self.scaler,
            'metrics': {
                'mse': mse,
                'r2': r2
            }
        }
    
    def save_models(self, models: Dict[str, Any], path: str = 'models'):
        """Save trained models and scaler"""
        joblib.dump(models['best_model'], f'{path}/best_model.joblib')
        joblib.dump(models['scaler'], f'{path}/scaler.joblib')
        models['neural_network'].save(f'{path}/neural_network.h5')
        
    def load_models(self, path: str = 'models') -> Dict[str, Any]:
        """Load trained models and scaler"""
        return {
            'best_model': joblib.load(f'{path}/best_model.joblib'),
            'scaler': joblib.load(f'{path}/scaler.joblib'),
            'neural_network': tf.keras.models.load_model(f'{path}/neural_network.h5')
        } 