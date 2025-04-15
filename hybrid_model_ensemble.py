import pandas as pd
import numpy as np
import joblib
import logging
import os
import traceback
from typing import Dict, List, Union, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import pytorch_forecasting
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from torch.utils.data import DataLoader

# Configure logging
logger = logging.getLogger('hybrid_model_ensemble')

class HybridModelEnsemble:
    """
    Advanced hybrid model ensemble for Dream11 predictions.
    Combines multiple model types with weighted voting and meta-learning.
    Now includes Temporal Fusion Transformer (TFT) for time-series forecasting.
    """
    
    def __init__(self, data_dir="dataset", models_dir="models"):
        """
        Initialize the HybridModelEnsemble
        
        Args:
            data_dir (str): Directory containing data files
            models_dir (str): Directory to store/load models
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.scaler = None
        self.base_models = {}
        self.ensemble_model = None
        self.meta_learner = None
        self.feature_importances = {}
        self.tft_model: Optional[TemporalFusionTransformer] = None 
        self.tft_training_data: Optional[TimeSeriesDataSet] = None
        self.tft_scaler_params: Optional[Dict] = None
        self.pitch_specific_ensembles = {}
        self.model_weights = {
            'xgboost': 0.3,
            'gradient_boosting': 0.25,
            'random_forest': 0.25,
            'neural_network': 0.2,
        }
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
    
    def load_base_models(self):
        """Load existing base models from disk"""
        try:
            # Load scaler
            scaler_path = os.path.join(self.models_dir, 'scaler.joblib')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")
            
            # Load base models
            model_files = {
                'xgboost': 'xgboost_model.joblib',
                'gradient_boosting': 'gradient_boosting_model.joblib',
                'random_forest': 'random_forest_model.joblib',
                'decision_tree': 'decision_tree_model.joblib',
                'knn': 'knn_model.joblib'
            }
            
            for model_name, file_name in model_files.items():
                file_path = os.path.join(self.models_dir, file_name)
                if os.path.exists(file_path):
                    self.base_models[model_name] = joblib.load(file_path)
                    logger.info(f"Loaded {model_name} model")
            
            # Load neural network if available
            nn_path = os.path.join(self.models_dir, 'neural_network_model.h5')
            if os.path.exists(nn_path):
                try:
                    self.base_models['neural_network'] = load_model(nn_path)
                    logger.info("Loaded neural network model")
                except Exception as e:
                    logger.error(f"Error loading neural network: {e}")
            
            # Load pitch-specific models
            for pitch_type in ['balanced', 'batting_friendly', 'bowling_friendly']:
                model_path = os.path.join(self.models_dir, f'{pitch_type}_model.pkl')
                if os.path.exists(model_path):
                    self.base_models[f'{pitch_type}_model'] = joblib.load(model_path)
                    logger.info(f"Loaded {pitch_type} pitch-specific model")
            
            if not self.base_models:
                logger.warning("No base models found. Ensemble cannot be created without training.")
                return False
            
            logger.info(f"Successfully loaded {len(self.base_models)} base models")
            return True
            
        except Exception as e:
            logger.error(f"Error loading base models: {e}")
            traceback.print_exc()
            return False
    
    def create_ensemble(self):
        """Create ensemble model from loaded base models"""
        try:
            if not self.base_models:
                logger.error("No base models available to create ensemble")
                return False
            
            # Filter models that support scikit-learn API
            sklearn_models = {}
            for name, model in self.base_models.items():
                if hasattr(model, 'predict') and name != 'neural_network':
                    sklearn_models[name] = model
            
            if not sklearn_models:
                logger.error("No compatible scikit-learn models found for ensemble")
                return False
            
            # Create weighted voting ensemble
            estimators = [(name, model) for name, model in sklearn_models.items()]
            weights = [self.model_weights.get(name, 0.1) for name, _ in estimators]
            
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            
            self.ensemble_model = VotingRegressor(estimators=estimators, weights=weights)
            logger.info(f"Created weighted voting ensemble with {len(estimators)} models")
            
            # Create meta-learner (stacking ensemble)
            meta_estimators = [(name, model) for name, model in sklearn_models.items()]
            self.meta_learner = StackingRegressor(
                estimators=meta_estimators,
                final_estimator=Ridge(alpha=1.0),
                cv=5
            )
            logger.info(f"Created stacking ensemble with {len(meta_estimators)} models and Ridge meta-learner")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating ensemble: {e}")
            traceback.print_exc()
            return False
    
    def fit_ensemble(self, X_train, y_train, X_val=None, y_val=None):
        """Fit the ensemble model and meta-learner"""
        try:
            if not self.ensemble_model or not self.meta_learner:
                if not self.create_ensemble():
                    logger.error("Failed to create ensembles")
                    return False
            
            # Scale features if scaler exists
            if self.scaler:
                X_train_scaled = self.scaler.transform(X_train)
                X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
            else:
                self.scaler = StandardScaler()
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
                
                # Save scaler
                joblib.dump(self.scaler, os.path.join(self.models_dir, 'ensemble_scaler.joblib'))
            
            # Fit weighted voting ensemble
            logger.info("Fitting weighted voting ensemble...")
            self.ensemble_model.fit(X_train_scaled, y_train)
            
            # Fit stacking ensemble
            logger.info("Fitting stacking ensemble (meta-learner)...")
            self.meta_learner.fit(X_train_scaled, y_train)
            
            # Evaluate if validation data is provided
            if X_val is not None and y_val is not None:
                # Evaluate weighted ensemble
                y_pred_ensemble = self.ensemble_model.predict(X_val_scaled)
                r2_ensemble = r2_score(y_val, y_pred_ensemble)
                mae_ensemble = mean_absolute_error(y_val, y_pred_ensemble)
                
                # Evaluate meta-learner
                y_pred_meta = self.meta_learner.predict(X_val_scaled)
                r2_meta = r2_score(y_val, y_pred_meta)
                mae_meta = mean_absolute_error(y_val, y_pred_meta)
                
                logger.info(f"Weighted Ensemble - R²: {r2_ensemble:.4f}, MAE: {mae_ensemble:.2f}")
                logger.info(f"Meta-Learner - R²: {r2_meta:.4f}, MAE: {mae_meta:.2f}")
            
            # Save ensemble models
            self.save_ensemble_models()
            
            return True
            
        except Exception as e:
            logger.error(f"Error fitting ensemble: {e}")
            traceback.print_exc()
            return False
    
    def create_time_series_component(self, input_dim):
        """Create time-series model component for form prediction"""
        try:
            # Create a simple sequential neural network for time series
            model = Sequential([
                Input(shape=(input_dim,)),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1, activation='linear')
            ])
            
            model.compile(
                optimizer='adam',
                loss='mean_squared_error',
                metrics=['mae']
            )
            
            self.time_series_model = model
            logger.info("Created time-series model component")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating time-series component: {e}")
            traceback.print_exc()
            return False
    
    def fit_time_series_component(self, X_time_series, y_time_series, validation_split=0.2):
        """Fit the time-series model component for form prediction"""
        try:
            if not self.time_series_model:
                logger.error("Time-series model not created. Call create_time_series_component first.")
                return False
            
            # Define early stopping to prevent overfitting
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Fit the model
            self.time_series_model.fit(
                X_time_series, y_time_series,
                validation_split=validation_split,
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Save the time-series model
            self.time_series_model.save(os.path.join(self.models_dir, 'time_series_model.h5'))
            logger.info("Time-series model fitted and saved")
            
            return True
            
        except Exception as e:
            logger.error(f"Error fitting time-series component: {e}")
            traceback.print_exc()
            return False
    
    def predict_with_ensemble(self, X, use_meta_learner=True, use_tft=True):
        """
        Make predictions using the ensemble model, meta-learner, or TFT.
        Prioritizes TFT if available and requested.
        """
        # --- Try TFT First ---
        if use_tft and self.tft_model and self.tft_training_data:
            try:
                # Check if X contains necessary time-series data for TFT
                required_cols = ['player_id', 'match_date', 'time_idx']
                if all(col in X.columns for col in required_cols):
                    logger.info("Input data contains required columns for TFT prediction")
                    
                    # Prepare data for TFT prediction
                    # Sort by player_id and match_date to ensure correct sequence
                    X_tft = X.copy()
                    X_tft['match_date'] = pd.to_datetime(X_tft['match_date'])
                    X_tft = X_tft.sort_values(['player_id', 'match_date'])
                    
                    # Ensure time_idx is properly set
                    if 'time_idx' not in X_tft.columns:
                        X_tft['time_idx'] = X_tft.groupby('player_id').cumcount()
                    
                    # Make TFT predictions
                    tft_predictions = self.predict_with_tft(X_tft, self.tft_training_data)
                    if tft_predictions is not None:
                        logger.info(f"Successfully made {len(tft_predictions)} predictions using TFT model")
                        return tft_predictions
                    else:
                        logger.warning("TFT prediction returned None. Falling back to other ensembles.")
                else:
                    missing = [col for col in required_cols if col not in X.columns]
                    logger.warning(f"Input data missing required columns for TFT: {missing}. Falling back to other ensembles.")
            except Exception as e:
                logger.error(f"Error during TFT prediction: {e}")
                logger.warning("TFT prediction failed. Falling back to other ensembles.")
        # --- End TFT ---
                
        try:
            # Fallback to existing ensemble logic if TFT fails or is not used
            if not self.ensemble_model and not self.meta_learner:
                logger.error("No ensemble models available. Load or create ensembles first.")
                return None
            
            # Scale features
            if self.scaler:
                X_scaled = self.scaler.transform(X)
            else:
                logger.error("No scaler available. Features cannot be scaled.")
                return None
            
            # Choose prediction model
            if use_meta_learner and self.meta_learner:
                predictions = self.meta_learner.predict(X_scaled)
                logger.info(f"Predicted {len(predictions)} samples using meta-learner")
            elif self.ensemble_model:
                predictions = self.ensemble_model.predict(X_scaled)
                logger.info(f"Predicted {len(predictions)} samples using weighted ensemble")
            else:
                logger.error("No ensemble models available for prediction")
                return None
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting with ensemble: {e}")
            traceback.print_exc()
            return None
    
    def save_ensemble_models(self):
        """Save trained ensemble models to disk"""
        try:
            # Save weighted ensemble
            if self.ensemble_model:
                joblib.dump(self.ensemble_model, os.path.join(self.models_dir, 'weighted_ensemble.joblib'))
                logger.info("Saved weighted ensemble model")
            
            # Save meta-learner
            if self.meta_learner:
                joblib.dump(self.meta_learner, os.path.join(self.models_dir, 'meta_learner.joblib'))
                logger.info("Saved meta-learner model")
            
            # Save scaler if exists
            if self.scaler:
                joblib.dump(self.scaler, os.path.join(self.models_dir, 'ensemble_scaler.joblib'))
                logger.info("Saved ensemble scaler")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving ensemble models: {e}")
            traceback.print_exc()
            return False
    
    def load_ensemble_models(self):
        """Load trained ensemble models from disk"""
        loaded_any = False
        try:
            # Load weighted ensemble
            ensemble_path = os.path.join(self.models_dir, 'weighted_ensemble.joblib')
            if os.path.exists(ensemble_path):
                self.ensemble_model = joblib.load(ensemble_path)
                logger.info("Loaded weighted ensemble model")
            
            # Load meta-learner
            meta_path = os.path.join(self.models_dir, 'meta_learner.joblib')
            if os.path.exists(meta_path):
                self.meta_learner = joblib.load(meta_path)
                logger.info("Loaded meta-learner model")
            
            # Load scaler
            scaler_path = os.path.join(self.models_dir, 'ensemble_scaler.joblib')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded ensemble scaler")
            
            # Load time-series model if available
            ts_path = os.path.join(self.models_dir, 'time_series_model.h5')
            if os.path.exists(ts_path):
                self.time_series_model = load_model(ts_path)
                logger.info("Loaded time-series model component")
            
            # --- Load TFT Model ---
            if self.load_tft_model():
                 loaded_any = True
            # --- End Load TFT Model ---

            if not self.ensemble_model and not self.meta_learner and not self.tft_model:
                logger.warning("No ensemble or TFT models found")
                return False
            
            logger.info("Finished loading models.")
            return loaded_any # Return True if at least one model (ensemble or TFT) was loaded
            
        except Exception as e:
            logger.error(f"Error loading ensemble models: {e}")
            traceback.print_exc()
            return False
    
    def create_pitch_specific_ensembles(self, X, y, pitch_types=['balanced', 'batting_friendly', 'bowling_friendly']):
        """Create and train pitch-specific ensemble models"""
        try:
            for pitch_type in pitch_types:
                logger.info(f"Creating pitch-specific ensemble for {pitch_type}")
                
                # Create estimators for each pitch type
                estimators = []
                for name, model in self.base_models.items():
                    if hasattr(model, 'predict') and name != 'neural_network':
                        # Clone the model to avoid modifying original
                        estimators.append((name, model))
                
                if not estimators:
                    logger.warning(f"No base models available for {pitch_type} ensemble")
                    continue
                
                # Create voting regressor for this pitch type
                voting_ensemble = VotingRegressor(estimators=estimators)
                
                # Fit the ensemble
                if self.scaler:
                    X_scaled = self.scaler.transform(X)
                else:
                    self.scaler = StandardScaler()
                    X_scaled = self.scaler.fit_transform(X)
                
                voting_ensemble.fit(X_scaled, y)
                
                # Save the pitch-specific ensemble
                self.pitch_specific_ensembles[pitch_type] = voting_ensemble
                joblib.dump(voting_ensemble, os.path.join(self.models_dir, f'{pitch_type}_ensemble.joblib'))
                logger.info(f"Saved {pitch_type} pitch-specific ensemble")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating pitch-specific ensembles: {e}")
            traceback.print_exc()
            return False
    
    def predict_with_pitch_specific_ensemble(self, X, pitch_type='balanced'):
        """Make predictions using a pitch-specific ensemble model"""
        try:
            # Load pitch-specific ensemble if not already loaded
            if pitch_type not in self.pitch_specific_ensembles:
                ensemble_path = os.path.join(self.models_dir, f'{pitch_type}_ensemble.joblib')
                if os.path.exists(ensemble_path):
                    self.pitch_specific_ensembles[pitch_type] = joblib.load(ensemble_path)
                    logger.info(f"Loaded {pitch_type} pitch-specific ensemble")
                else:
                    logger.error(f"No ensemble available for pitch type: {pitch_type}")
                    # Fall back to general ensemble
                    return self.predict_with_ensemble(X)
            
            # Scale features
            if self.scaler:
                X_scaled = self.scaler.transform(X)
            else:
                logger.error("No scaler available. Features cannot be scaled.")
                return None
            
            # Make predictions
            predictions = self.pitch_specific_ensembles[pitch_type].predict(X_scaled)
            logger.info(f"Predicted {len(predictions)} samples using {pitch_type} pitch-specific ensemble")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting with pitch-specific ensemble: {e}")
            traceback.print_exc()
            # Fall back to general ensemble
            return self.predict_with_ensemble(X)

    def prepare_tft_data(self, player_history_df: pd.DataFrame, target_metric: str = 'fantasy_points', time_idx_col: str = 'match_date'):
        """
        Prepare data for the Temporal Fusion Transformer model.
        
        Args:
            player_history_df (pd.DataFrame): DataFrame with historical player data. 
                                             Must contain player_id, target_metric, time_idx_col,
                                             and other relevant time-varying features.
            target_metric (str): The target variable to predict (e.g., 'fantasy_points').
            time_idx_col (str): Column representing the time index (e.g., 'match_date' or 'match_number').

        Returns:
            TimeSeriesDataSet or None: Prepared dataset for TFT training/validation.
        """
        logger.info("Preparing data for Temporal Fusion Transformer...")
        try:
            if player_history_df is None or player_history_df.empty:
                logger.error("Player history data is missing or empty.")
                return None

            # Ensure correct data types
            player_history_df[time_idx_col] = pd.to_datetime(player_history_df[time_idx_col], errors='coerce')
            player_history_df.dropna(subset=[time_idx_col], inplace=True)
            player_history_df = player_history_df.sort_values(['player_id', time_idx_col])

            # Create time index relative to each player
            player_history_df['time_idx'] = player_history_df.groupby('player_id').cumcount()
            
            # --- Define TFT features ---
            # Adjust these based on available columns in player_history_df
            static_categoricals = ['player_id', 'role'] 
            static_reals = ['credits'] # Example static real feature
            
            time_varying_known_categoricals = ['opposition_team_code'] # Example: opponent is known in advance
            time_varying_known_reals = ['venue_code', 'is_home'] # Example: venue, home status known

            time_varying_unknown_categoricals = [] # Example: match outcome, player role in match
            time_varying_unknown_reals = [
                target_metric, 'runs', 'wickets', 'strike_rate', 'economy', # Target + other performance metrics
                # Add other relevant unknown real features
            ]
            
            # Filter DataFrame to include only necessary columns + target
            all_req_cols = (
                [time_idx_col, 'time_idx', 'player_id', target_metric] + 
                [col for col in static_categoricals if col != 'player_id'] + 
                static_reals + 
                time_varying_known_categoricals + 
                time_varying_known_reals + 
                time_varying_unknown_categoricals + 
                [col for col in time_varying_unknown_reals if col != target_metric]
            )
            
            missing_cols = [col for col in all_req_cols if col not in player_history_df.columns]
            if missing_cols:
                logger.warning(f"Missing required columns for TFT data preparation: {missing_cols}. Proceeding without them.")
                # Filter out missing columns from feature lists
                static_categoricals = [c for c in static_categoricals if c in player_history_df.columns or c == 'player_id']
                static_reals = [c for c in static_reals if c in player_history_df.columns]
                time_varying_known_categoricals = [c for c in time_varying_known_categoricals if c in player_history_df.columns]
                time_varying_known_reals = [c for c in time_varying_known_reals if c in player_history_df.columns]
                time_varying_unknown_categoricals = [c for c in time_varying_unknown_categoricals if c in player_history_df.columns]
                time_varying_unknown_reals = [c for c in time_varying_unknown_reals if c in player_history_df.columns or c == target_metric]

            # Ensure categorical columns are string type for TFT
            for col in static_categoricals + time_varying_known_categoricals + time_varying_unknown_categoricals:
                if col in player_history_df.columns:
                    player_history_df[col] = player_history_df[col].astype(str)
            
            # Handle potential NaN in target before creating dataset
            player_history_df[target_metric] = player_history_df[target_metric].fillna(0) 
            # Ensure numeric types for real features - fill NaNs appropriately
            for col in static_reals + time_varying_known_reals + time_varying_unknown_reals:
                 if col in player_history_df.columns:
                      player_history_df[col] = pd.to_numeric(player_history_df[col], errors='coerce').fillna(0) # Example: fill NaNs with 0

            # Define dataset parameters
            max_encoder_length = 5 # How many past matches to look at
            max_prediction_length = 1 # Predict the next match

            # Create the TimeSeriesDataSet
            training_cutoff = player_history_df["time_idx"].max() - max_prediction_length
            
            # Store scaler parameters for potential use during prediction
            target_normalizer = GroupNormalizer(groups=["player_id"], transformation="softplus")
            
            dataset = TimeSeriesDataSet(
                player_history_df[lambda x: x.time_idx <= training_cutoff],
                time_idx="time_idx",
                target=target_metric,
                group_ids=["player_id"],
                max_encoder_length=max_encoder_length,
                max_prediction_length=max_prediction_length,
                static_categoricals=[col for col in static_categoricals if col != 'player_id'], # player_id is group_id
                static_reals=[col for col in static_reals if col in player_history_df.columns],
                time_varying_known_categoricals=[col for col in time_varying_known_categoricals if col in player_history_df.columns],
                time_varying_known_reals=[col for col in time_varying_known_reals if col in player_history_df.columns],
                time_varying_unknown_categoricals=[col for col in time_varying_unknown_categoricals if col in player_history_df.columns],
                time_varying_unknown_reals=[col for col in time_varying_unknown_reals if col in player_history_df.columns],
                target_normalizer=target_normalizer,
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
            )
            
            self.tft_training_data = dataset # Store for training
            logger.info(f"TFT TimeSeriesDataSet created successfully with {len(dataset)} samples.")
            return dataset

        except Exception as e:
            logger.error(f"Error preparing TFT data: {e}", exc_info=True)
            traceback.print_exc()
            return None

    def train_tft_component(self, dataset: TimeSeriesDataSet, epochs=30, batch_size=64, learning_rate=0.001, hidden_size=32, attention_head_size=2, dropout=0.1):
        """
        Train the Temporal Fusion Transformer model component.
        
        Args:
            dataset (TimeSeriesDataSet): Prepared TFT dataset.
            epochs (int): Number of training epochs.
            batch_size (int): Training batch size.
            learning_rate (float): Learning rate for the optimizer.
            hidden_size (int): Hidden layer size for TFT.
            attention_head_size (int): Number of attention heads.
            dropout (float): Dropout rate.

        Returns:
            bool: True if training was successful, False otherwise.
        """
        if dataset is None:
            logger.error("Cannot train TFT model without a valid dataset.")
            return False
            
        logger.info(f"Starting TFT model training for {epochs} epochs...")
        try:
            # Create validation dataset
            validation = TimeSeriesDataSet.from_dataset(dataset, player_history_df, predict=True, stop_randomization=True)
            
            # Create dataloaders
            train_dataloader = dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0) # Set num_workers > 0 if possible
            val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

            # Initialize trainer
            # Check for available GPUs? For now, assume CPU training (gpus=0)
            trainer = pl.Trainer(
                max_epochs=epochs,
                gpus=0, # Set to 1 or more if GPUs are available and configured
                gradient_clip_val=0.1,
                limit_train_batches=30, # Limit batches for faster training (remove for full run)
                callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")],
                logger=pl.loggers.TensorBoardLogger(save_dir=os.path.join(self.models_dir, "tft_logs")),
            )

            # Initialize TFT model
            tft = TemporalFusionTransformer.from_dataset(
                dataset,
                learning_rate=learning_rate,
                hidden_size=hidden_size,
                attention_head_size=attention_head_size,
                dropout=dropout,
                hidden_continuous_size=hidden_size // 2, # Example configuration
                output_size=7,  # Number of quantiles to predict
                loss=pytorch_forecasting.metrics.QuantileLoss(),
                log_interval=10, # Log every 10 batches
                reduce_on_plateau_patience=4,
            )
            logger.info(f"Number of parameters in TFT model: {tft.size()/1e3:.1f}k")

            # Train the model
            trainer.fit(
                tft,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )

            # Load best model checkpoint
            best_model_path = trainer.checkpoint_callback.best_model_path
            self.tft_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
            logger.info(f"TFT training finished. Best model loaded from {best_model_path}")

            # Save the final trained model separately
            tft_save_path = os.path.join(self.models_dir, "tft_model.ckpt")
            trainer.save_checkpoint(tft_save_path)
            logger.info(f"Final TFT model saved to {tft_save_path}")
            
            return True

        except Exception as e:
            logger.error(f"Error training TFT component: {e}", exc_info=True)
            traceback.print_exc()
            return False

    def load_tft_model(self):
        """Load a pre-trained TFT model from disk."""
        tft_model_path = os.path.join(self.models_dir, "tft_model.ckpt")
        if os.path.exists(tft_model_path):
            try:
                self.tft_model = TemporalFusionTransformer.load_from_checkpoint(tft_model_path)
                logger.info(f"Loaded TFT model from {tft_model_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading TFT model from checkpoint: {e}", exc_info=True)
                self.tft_model = None
                return False
        else:
            logger.warning(f"TFT model checkpoint not found at {tft_model_path}")
            self.tft_model = None
            return False
            
    def predict_with_tft(self, prediction_data_df: pd.DataFrame, dataset: TimeSeriesDataSet) -> Optional[np.ndarray]:
        """
        Make predictions using the trained TFT model.
        
        Args:
            prediction_data_df (pd.DataFrame): Dataframe containing the features for the time points to predict. 
                                                Needs historical context (encoder data).
            dataset (TimeSeriesDataSet): The *original* TimeSeriesDataSet object used for training 
                                         (contains metadata like scalers).

        Returns:
            np.ndarray or None: Array of predicted points (usually median quantile), or None if prediction fails.
        """
        if self.tft_model is None:
            logger.error("TFT model is not loaded or trained. Cannot predict.")
            return None
        if dataset is None:
             logger.error("Original training TimeSeriesDataSet is required for TFT prediction.")
             return None

        logger.info(f"Predicting with TFT model for {len(prediction_data_df)} data points...")
        try:
            # Ensure data types match training data schema (use dataset metadata)
            # Convert categorical columns to string type for TFT
            for col in prediction_data_df.select_dtypes(include=['object', 'category']).columns:
                prediction_data_df[col] = prediction_data_df[col].astype(str)
                
            # Handle missing values in numeric columns
            for col in prediction_data_df.select_dtypes(include=['float64', 'int64']).columns:
                prediction_data_df[col] = prediction_data_df[col].fillna(0)

            # Create prediction dataloader
            # Important: Use the 'dataset' object to create the dataloader for prediction
            # It ensures correct processing and scaling based on training data.
            pred_dataloader = dataset.to_dataloader(train=False, data=prediction_data_df, batch_size=128) # Use a larger batch size for prediction

            # Make predictions
            # Set model to evaluation mode
            self.tft_model.eval() 
            
            raw_predictions = self.tft_model.predict(pred_dataloader, return_index=True, return_decoder_lengths=True)
            
            # Often, the median quantile (index 3 for 7 quantiles) is used as the point prediction
            # predictions.output shape is (n_samples, n_timesteps_out, n_quantiles)
            # We predict only 1 step ahead, so index 0 for timestep
            median_predictions = raw_predictions.output[:, 0, 3].numpy() 

            logger.info(f"TFT prediction successful for {len(median_predictions)} players.")
            return median_predictions

        except Exception as e:
            logger.error(f"Error predicting with TFT model: {e}", exc_info=True)
            traceback.print_exc()
            return None
            
    def combine_predictions(self, X, ensemble_weight=0.4, tft_weight=0.6):
        """
        Combines predictions from TFT model and ensemble models for improved accuracy.
        
        Args:
            X (pd.DataFrame): Input features for prediction
            ensemble_weight (float): Weight for ensemble model predictions (0-1)
            tft_weight (float): Weight for TFT model predictions (0-1)
            
        Returns:
            np.ndarray or None: Combined predictions or None if prediction fails
        """
        try:
            # Normalize weights
            total_weight = ensemble_weight + tft_weight
            ensemble_weight = ensemble_weight / total_weight
            tft_weight = tft_weight / total_weight
            
            logger.info(f"Combining predictions with weights: Ensemble={ensemble_weight:.2f}, TFT={tft_weight:.2f}")
            
            # Get TFT predictions
            tft_predictions = None
            if self.tft_model and self.tft_training_data:
                # Check if X contains necessary time-series data for TFT
                required_cols = ['player_id', 'match_date', 'time_idx']
                if all(col in X.columns for col in required_cols):
                    # Prepare data for TFT prediction
                    X_tft = X.copy()
                    X_tft['match_date'] = pd.to_datetime(X_tft['match_date'])
                    X_tft = X_tft.sort_values(['player_id', 'match_date'])
                    
                    # Ensure time_idx is properly set
                    if 'time_idx' not in X_tft.columns:
                        X_tft['time_idx'] = X_tft.groupby('player_id').cumcount()
                    
                    # Make TFT predictions
                    tft_predictions = self.predict_with_tft(X_tft, self.tft_training_data)
                    logger.info(f"Got TFT predictions for {len(tft_predictions) if tft_predictions is not None else 0} samples")
                else:
                    missing = [col for col in required_cols if col not in X.columns]
                    logger.warning(f"Input data missing required columns for TFT: {missing}")
            
            # Get ensemble predictions
            ensemble_predictions = None
            if self.meta_learner or self.ensemble_model:
                # Scale features
                if self.scaler:
                    X_scaled = self.scaler.transform(X.drop(columns=['player_id', 'match_date', 'time_idx'], errors='ignore'))
                    
                    # Choose prediction model
                    if self.meta_learner:
                        ensemble_predictions = self.meta_learner.predict(X_scaled)
                    elif self.ensemble_model:
                        ensemble_predictions = self.ensemble_model.predict(X_scaled)
                    
                    logger.info(f"Got ensemble predictions for {len(ensemble_predictions) if ensemble_predictions is not None else 0} samples")
            
            # Combine predictions
            if tft_predictions is not None and ensemble_predictions is not None:
                # Ensure same length
                min_len = min(len(tft_predictions), len(ensemble_predictions))
                combined_predictions = (tft_weight * tft_predictions[:min_len] + 
                                       ensemble_weight * ensemble_predictions[:min_len])
                logger.info(f"Combined {min_len} predictions from both models")
                return combined_predictions
            elif tft_predictions is not None:
                logger.info("Using only TFT predictions")
                return tft_predictions
            elif ensemble_predictions is not None:
                logger.info("Using only ensemble predictions")
                return ensemble_predictions
            else:
                logger.error("No predictions available from either model")
                return None
                
        except Exception as e:
            logger.error(f"Error combining predictions: {e}")
            traceback.print_exc()
            return None