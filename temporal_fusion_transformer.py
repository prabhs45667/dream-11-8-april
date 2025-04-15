import pandas as pd
import numpy as np
import os
import logging
import torch
import traceback
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('temporal_fusion_transformer')

class TemporalFusionTransformer:
    """
    Implementation of Temporal Fusion Transformer for Dream11 fantasy cricket predictions.
    This model enhances prediction accuracy by capturing temporal patterns in player performance.
    """
    
    def __init__(self, data_dir="dataset", models_dir="models"):
        """
        Initialize the TFT model
        
        Args:
            data_dir (str): Directory containing data files
            models_dir (str): Directory for saving/loading models
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.model = None
        self.trainer = None
        self.training_data = None
        self.validation_data = None
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.join(models_dir, "tft"), exist_ok=True)
        
        # Try to import required libraries
        try:
            global pl, TimeSeriesDataSet, TemporalFusionTransformer
            import pytorch_lightning as pl
            from pytorch_forecasting import TimeSeriesDataSet
            from pytorch_forecasting import TemporalFusionTransformer
            logger.info("Successfully imported pytorch_forecasting and pytorch_lightning")
        except ImportError as e:
            logger.error(f"Error importing required libraries: {e}")
            logger.info("Please install required packages: pip install pytorch-forecasting pytorch-lightning")
            raise ImportError("Required packages not installed. Run: pip install pytorch-forecasting pytorch-lightning")
    
    def create_time_series_dataset(self, player_data):
        """
        Create time series dataset from player data
        
        Args:
            player_data (pd.DataFrame): DataFrame containing player match history
            
        Returns:
            TimeSeriesDataSet: Dataset for training TFT model
        """
        try:
            # Ensure match_date is datetime
            player_data['match_date'] = pd.to_datetime(player_data['match_date'])
            
            # Sort by player_id and match_date
            player_data = player_data.sort_values(['player_id', 'match_date'])
            
            # Create time index within each player series
            player_data['time_idx'] = player_data.groupby('player_id').cumcount()
            
            # Identify available features
            time_varying_known_reals = []
            time_varying_unknown_reals = ['fantasy_points']
            
            # Add available features to appropriate categories
            for feature in ['venue_code', 'opposition_strength', 'is_home']:
                if feature in player_data.columns:
                    time_varying_known_reals.append(feature)
                    
            for feature in ['runs', 'wickets', 'strike_rate', 'economy']:
                if feature in player_data.columns:
                    time_varying_unknown_reals.append(feature)
            
            # Create TimeSeriesDataSet
            training = TimeSeriesDataSet(
                player_data,
                time_idx="time_idx",
                target="fantasy_points",
                group_ids=["player_id"],
                max_encoder_length=5,  # Last 5 matches
                max_prediction_length=1,  # Next match
                time_varying_known_reals=time_varying_known_reals,
                time_varying_unknown_reals=time_varying_unknown_reals
            )
            
            logger.info(f"Created time series dataset with {len(player_data)} records")
            return training
            
        except Exception as e:
            logger.error(f"Error creating time series dataset: {e}")
            return None
    
    def prepare_data(self):
        """
        Prepare data for TFT model training
        
        Returns:
            bool: True if data preparation was successful, False otherwise
        """
        try:
            # Load historical player data
            player_history_file = os.path.join(self.data_dir, "player_history.csv")
            if not os.path.exists(player_history_file):
                logger.error(f"Player history file not found at {player_history_file}")
                return False
                
            player_data = pd.read_csv(player_history_file)
            
            # Check if required columns exist
            required_columns = ['match_date', 'fantasy_points']
            missing_columns = [col for col in required_columns if col not in player_data.columns]
            
            # Add player_id column if missing (using player_name as player_id)
            if 'player_id' not in player_data.columns and 'player_name' in player_data.columns:
                logger.info("Creating player_id column from player_name")
                player_data['player_id'] = player_data['player_name']
            elif 'player_id' not in player_data.columns:
                logger.error("Neither player_id nor player_name column found in player history")
                return False
            
            if missing_columns:
                logger.error(f"Missing required columns in player history: {missing_columns}")
                return False
            
            # Create time series dataset
            training_data = self.create_time_series_dataset(player_data)
            if training_data is None:
                return False
                
            # Create validation dataset with appropriate min_prediction_idx
            # Calculate a more appropriate min_prediction_idx that ensures more players are included
            # Most players should have at least 2 matches to be included in validation
            min_required_history = 2  # Minimum number of matches required for validation
            
            # Get counts of matches per player
            player_counts = player_data.groupby('player_id').size()
            
            # Find a reasonable min_prediction_idx that includes players with sufficient history
            # Use the 25th percentile of time_idx to ensure most players are included
            try:
                reasonable_idx = max(min_required_history, 
                                    int(player_data.groupby('player_id')['time_idx'].max().quantile(0.25)))
            except Exception as idx_error:
                logger.warning(f"Error calculating reasonable min_prediction_idx: {idx_error}")
                reasonable_idx = min_required_history  # Fallback to minimum required history
            
            # Create validation dataset with the calculated min_prediction_idx
            try:
                validation_data = TimeSeriesDataSet.from_dataset(
                    training_data, 
                    player_data, 
                    min_prediction_idx=reasonable_idx
                )
                
                # Create data loaders
                self.train_dataloader = training_data.to_dataloader(train=True, batch_size=32)
                self.val_dataloader = validation_data.to_dataloader(train=False, batch_size=32)
                
                # Save references to datasets
                self.training_data = training_data
                self.validation_data = validation_data
                
                logger.info("Data preparation completed successfully with validation dataset")
                return True
                
            except Exception as e:
                logger.warning(f"Could not create validation dataset with calculated min_prediction_idx: {e}")
                logger.info("Falling back to using training dataset for validation")
                
                # Fallback: use training dataset for validation
                self.train_dataloader = training_data.to_dataloader(train=True, batch_size=32)
                self.val_dataloader = self.train_dataloader  # Use same data for validation as fallback
                
                # Save references to datasets
                self.training_data = training_data
                self.validation_data = self.training_data  # Use training data as validation data
                
                logger.info("Data preparation completed successfully with fallback validation")
                return True
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            traceback.print_exc()  # Print full traceback for debugging
            return False
    
    def train_model(self, max_epochs=30):
        """
        Train the TFT model
        
        Args:
            max_epochs (int): Maximum number of training epochs
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            # Prepare data if not already done
            if self.training_data is None:
                success = self.prepare_data()
                if not success:
                    return False
            
            # Initialize trainer with updated parameters for compatibility
            try:
                # Try with newer PyTorch Lightning API
                self.trainer = pl.Trainer(
                    max_epochs=max_epochs,
                    accelerator='cpu',  # Explicitly use CPU
                    gradient_clip_val=0.1,
                    limit_train_batches=50,  # Limit batches for faster training
                )
            except TypeError:
                # Fall back to older PyTorch Lightning API
                logger.info("Falling back to older PyTorch Lightning API")
                self.trainer = pl.Trainer(
                    max_epochs=max_epochs,
                    gpus=0,  # CPU only
                    gradient_clip_val=0.1,
                    limit_train_batches=50,  # Limit batches for faster training
                )
            
            # Initialize model
            self.model = TemporalFusionTransformer.from_dataset(
                self.training_data,
                hidden_size=32,
                attention_head_size=2,
                dropout=0.1,
                learning_rate=0.001,
                log_interval=10,
                reduce_on_plateau_patience=4
            )
            
            # Train model
            self.trainer.fit(
                self.model,
                train_dataloaders=self.train_dataloader,
                val_dataloaders=self.val_dataloader
            )
            
            # Save model
            model_path = os.path.join(self.models_dir, "tft", "tft_model.ckpt")
            self.trainer.save_checkpoint(model_path)
            logger.info(f"Model saved to {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def load_model(self):
        """
        Load a trained TFT model
        
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            model_path = os.path.join(self.models_dir, "tft", "tft_model.ckpt")
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at {model_path}")
                # Try to train a new model if data is available
                if self.training_data is None:
                    success = self.prepare_data()
                    if success:
                        logger.info("Attempting to train a new model since no saved model was found")
                        train_success = self.train_model(max_epochs=5)  # Quick training with fewer epochs
                        if train_success:
                            return True
                return False
                
            # Prepare data if not already done (needed for model initialization)
            if self.training_data is None:
                success = self.prepare_data()
                if not success:
                    return False
            
            # Load model
            self.model = TemporalFusionTransformer.load_from_checkpoint(model_path)
            logger.info(f"Model loaded from {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            traceback.print_exc()  # Print full traceback for debugging
            return False
    
    def predict(self, player_data):
        """
        Make predictions using the trained TFT model
        
        Args:
            player_data (pd.DataFrame): Player data for prediction
            
        Returns:
            pd.DataFrame: Player data with predictions
        """
        try:
            # Ensure model is loaded
            if self.model is None:
                success = self.load_model()
                if not success:
                    logger.warning("Failed to load model for prediction, using fallback prediction")
                    # Use a simple fallback prediction based on player's average fantasy points
                    return self._fallback_prediction(player_data)
            
            # Ensure data is properly formatted
            player_data_copy = player_data.copy()
            
            # Convert date to datetime if it exists
            if 'match_date' in player_data_copy.columns:
                player_data_copy['match_date'] = pd.to_datetime(player_data_copy['match_date'])
            
            # Add player_id column if missing (using player_name as player_id)
            if 'player_id' not in player_data_copy.columns and 'player_name' in player_data_copy.columns:
                logger.info("Creating player_id column from player_name for prediction data")
                player_data_copy['player_id'] = player_data_copy['player_name']
            elif 'player_id' not in player_data_copy.columns:
                logger.error("Neither player_id nor player_name column found in prediction data")
                return player_data
            
            # Sort by player_id and match_date
            if 'match_date' in player_data_copy.columns:
                player_data_copy = player_data_copy.sort_values(['player_id', 'match_date'])
            
            # Create time index if it doesn't exist
            if 'time_idx' not in player_data_copy.columns:
                logger.info("Creating time_idx for prediction data")
                player_data_copy['time_idx'] = player_data_copy.groupby('player_id').cumcount()
            
            # Handle missing values in numeric columns
            for col in player_data_copy.select_dtypes(include=['float64', 'int64']).columns:
                player_data_copy[col] = player_data_copy[col].fillna(0)
            
            # Convert categorical columns to string
            for col in player_data_copy.select_dtypes(include=['object', 'category']).columns:
                player_data_copy[col] = player_data_copy[col].astype(str)
            
            # Prepare data for prediction
            prediction_data = self.create_time_series_dataset(player_data_copy)
            if prediction_data is None:
                logger.error("Failed to create time series dataset for prediction")
                return player_data
                
            # Create prediction dataloader
            prediction_dataloader = prediction_data.to_dataloader(train=False, batch_size=32)
            
            # Make predictions
            try:
                # Get raw predictions from model
                raw_predictions = self.model.predict(prediction_dataloader, return_index=True, return_decoder_lengths=True)
                
                # Create result dataframe with prediction column
                result_df = player_data.copy()
                result_df['tft_predicted_points'] = np.nan
                
                # Safe extraction of predictions
                try:
                    # Get indices from raw_predictions to map back to original data
                    indices = raw_predictions.index
                    
                    # Extract predictions based on tensor shape
                    if hasattr(raw_predictions.output, 'shape'):
                        # Get tensor shape to determine extraction method
                        output_shape = raw_predictions.output.shape
                        logger.info(f"Prediction output shape: {output_shape}")
                        
                        # Extract predictions based on shape
                        if len(output_shape) == 3 and output_shape[2] > 3:
                            # Standard case: [batch, time, quantiles]
                            predictions = raw_predictions.output[:, 0, 3].cpu().numpy()
                        elif len(output_shape) == 3:
                            # Fewer quantiles: use middle quantile or mean
                            middle_idx = output_shape[2] // 2
                            predictions = raw_predictions.output[:, 0, middle_idx].cpu().numpy()
                        elif len(output_shape) == 2 and output_shape[1] > 1:
                            # 2D case: [batch, quantiles]
                            middle_idx = output_shape[1] // 2
                            predictions = raw_predictions.output[:, middle_idx].cpu().numpy()
                        else:
                            # Single prediction per sample
                            predictions = raw_predictions.output.cpu().numpy().flatten()
                    else:
                        # If output doesn't have shape attribute, try to convert to numpy directly
                        predictions = raw_predictions.output.cpu().numpy().flatten()
                    
                    # Map predictions back to original dataframe using indices
                    if hasattr(indices, 'get_index'):
                        # Get the group IDs from the index
                        groups = indices.get_index()
                        
                        # Create a mapping from group ID to prediction
                        pred_map = {}
                        for i, group in enumerate(groups):
                            if i < len(predictions):
                                # Extract player_id from group
                                if isinstance(group, dict) and '__group_id__player_id' in group:
                                    player_id = group['__group_id__player_id']
                                    pred_map[player_id] = predictions[i]
                        
                        # Apply predictions to result dataframe
                        for idx, row in result_df.iterrows():
                            if row['player_id'] in pred_map:
                                result_df.loc[idx, 'tft_predicted_points'] = pred_map[row['player_id']]
                    else:
                        # Fallback: assign predictions directly if indices don't have get_index method
                        # This is less accurate but better than nothing
                        for i, idx in enumerate(indices):
                            if i < len(predictions):
                                # Make sure idx is a valid index
                                try:
                                    if isinstance(idx, (int, np.integer)) and idx < len(result_df):
                                        result_df.loc[idx, 'tft_predicted_points'] = predictions[i]
                                except TypeError:
                                    # Skip if idx is not a valid index type
                                    continue
                    
                    # Count how many predictions were made
                    pred_count = result_df['tft_predicted_points'].notna().sum()
                    logger.info(f"Made predictions for {pred_count} players")
                    
                    # If no predictions were made, fall back to simple prediction
                    if pred_count == 0:
                        logger.warning("No predictions were mapped to players, using fallback")
                        return self._fallback_prediction(player_data)
                    
                    return result_df
                    
                except Exception as tensor_error:
                    logger.warning(f"Error processing model predictions: {tensor_error}")
                    traceback.print_exc()
                    return self._fallback_prediction(player_data)
            except Exception as pred_error:
                logger.error(f"Error during prediction: {pred_error}")
                logger.info("Returning original data without predictions")
                return player_data
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            traceback.print_exc()
            return player_data

    def _fallback_prediction(self, player_data):
        """
        Simple fallback prediction when model is not available
        
        Args:
            player_data (pd.DataFrame): Player data for prediction
            
        Returns:
            pd.DataFrame: Player data with simple predictions
        """
        try:
            result_df = player_data.copy()
            
            # Add player_id if it doesn't exist
            if 'player_id' not in result_df.columns and 'player_name' in result_df.columns:
                result_df['player_id'] = result_df['player_name']
            
            # Calculate average fantasy points per player as a simple prediction
            if 'fantasy_points' in result_df.columns and 'player_id' in result_df.columns:
                # Group by player_id and calculate mean fantasy points
                player_avg = result_df.groupby('player_id')['fantasy_points'].mean().reset_index()
                player_avg.rename(columns={'fantasy_points': 'tft_predicted_points'}, inplace=True)
                
                # Merge with original data
                result_df = result_df.merge(player_avg, on='player_id', how='left')
                
                logger.info(f"Generated fallback predictions for {len(player_avg)} players")
            else:
                # If fantasy_points column doesn't exist, just add a placeholder column
                result_df['tft_predicted_points'] = np.nan
                logger.warning("Could not generate fallback predictions due to missing columns")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in fallback prediction: {e}")
            traceback.print_exc()
            # Return original data if fallback fails
            player_data['tft_predicted_points'] = np.nan
            return player_data

# Example usage
if __name__ == "__main__":
    # Initialize TFT model
    tft = TemporalFusionTransformer()
    
    # Prepare data and train model
    success = tft.prepare_data()
    if success:
        tft.train_model(max_epochs=10)  # Reduced epochs for testing
    
    # Load player data for prediction
    player_data = pd.read_csv(os.path.join("dataset", "player_history.csv"))
    
    # Make predictions
    predictions = tft.predict(player_data)
    
    # Print sample predictions
    if 'player_id' in predictions.columns and 'tft_predicted_points' in predictions.columns:
        print(predictions[["player_id", "fantasy_points", "tft_predicted_points"]].head())
    else:
        available_cols = [col for col in ["player_id", "player_name", "fantasy_points", "tft_predicted_points"] if col in predictions.columns]
        print(predictions[available_cols].head())