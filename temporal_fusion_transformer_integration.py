import pandas as pd
import numpy as np
import os
import logging
import torch
from typing import Dict, List, Tuple, Optional
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformerModel
from pytorch_forecasting.data import GroupNormalizer
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tft_integration')

def integrate_tft_with_ensemble(hybrid_model, player_history_df, target_col='fantasy_points'):
    """
    Integrates the Temporal Fusion Transformer model with the hybrid ensemble model.
    
    Args:
        hybrid_model: The HybridModelEnsemble instance
        player_history_df (pd.DataFrame): Historical player data
        target_col (str): Target column to predict
        
    Returns:
        bool: True if integration was successful, False otherwise
    """
    try:
        logger.info("Starting TFT integration with hybrid ensemble model...")
        
        # Step 1: Prepare data for TFT
        tft_dataset = hybrid_model.prepare_tft_data(player_history_df, target_metric=target_col)
        if tft_dataset is None:
            logger.error("Failed to prepare TFT dataset")
            return False
            
        # Step 2: Train TFT model
        success = hybrid_model.train_tft_component(
            dataset=tft_dataset,
            epochs=30,
            batch_size=64,
            learning_rate=0.001,
            hidden_size=32,
            attention_head_size=2,
            dropout=0.1
        )
        
        if not success:
            logger.error("Failed to train TFT model")
            return False
            
        logger.info("TFT model successfully trained and integrated with ensemble")
        return True
        
    except Exception as e:
        logger.error(f"Error integrating TFT with ensemble: {e}")
        return False

def evaluate_tft_performance(hybrid_model, test_data, target_col='fantasy_points'):
    """
    Evaluates the performance of the TFT model compared to other models in the ensemble.
    
    Args:
        hybrid_model: The HybridModelEnsemble instance
        test_data (pd.DataFrame): Test data for evaluation
        target_col (str): Target column to predict
        
    Returns:
        dict: Performance metrics for TFT and ensemble models
    """
    try:
        logger.info("Evaluating TFT model performance...")
        
        # Prepare test data for TFT
        X_test = test_data.drop(columns=[target_col])
        y_test = test_data[target_col]
        
        # Get predictions from TFT model
        tft_predictions = hybrid_model.predict_with_ensemble(X_test, use_tft=True, use_meta_learner=False)
        
        # Get predictions from meta-learner ensemble
        ensemble_predictions = hybrid_model.predict_with_ensemble(X_test, use_tft=False, use_meta_learner=True)
        
        # Calculate metrics
        from sklearn.metrics import r2_score, mean_absolute_error
        
        metrics = {}
        
        if tft_predictions is not None:
            metrics['tft_r2'] = r2_score(y_test, tft_predictions)
            metrics['tft_mae'] = mean_absolute_error(y_test, tft_predictions)
            logger.info(f"TFT Model - R²: {metrics['tft_r2']:.4f}, MAE: {metrics['tft_mae']:.2f}")
        
        if ensemble_predictions is not None:
            metrics['ensemble_r2'] = r2_score(y_test, ensemble_predictions)
            metrics['ensemble_mae'] = mean_absolute_error(y_test, ensemble_predictions)
            logger.info(f"Ensemble Model - R²: {metrics['ensemble_r2']:.4f}, MAE: {metrics['ensemble_mae']:.2f}")
        
        # Calculate improvement
        if 'tft_r2' in metrics and 'ensemble_r2' in metrics:
            improvement = metrics['tft_r2'] - metrics['ensemble_r2']
            metrics['r2_improvement'] = improvement
            logger.info(f"TFT R² improvement: {improvement:.4f} ({improvement*100:.2f}%)")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating TFT performance: {e}")
        return {}

def prepare_player_data_for_tft(player_data):
    """
    Prepares player data specifically for TFT prediction.
    
    Args:
        player_data (pd.DataFrame): Raw player data
        
    Returns:
        pd.DataFrame: Processed data ready for TFT
    """
    try:
        # Ensure required columns exist
        required_cols = ['player_id', 'match_date', 'fantasy_points']
        missing_cols = [col for col in required_cols if col not in player_data.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return None
        
        # Process data
        df = player_data.copy()
        
        # Convert date to datetime
        df['match_date'] = pd.to_datetime(df['match_date'])
        
        # Sort by player and date
        df = df.sort_values(['player_id', 'match_date'])
        
        # Create time index
        df['time_idx'] = df.groupby('player_id').cumcount()
        
        # Handle missing values
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col] = df[col].fillna(0)
        
        # Convert categorical columns to string
        for col in df.select_dtypes(include=['object', 'category']).columns:
            df[col] = df[col].astype(str)
        
        logger.info(f"Prepared player data for TFT with {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error preparing player data for TFT: {e}")
        return None

# Example usage
if __name__ == "__main__":
    from hybrid_model_ensemble import HybridModelEnsemble
    
    # Initialize hybrid model
    hybrid_model = HybridModelEnsemble()
    
    # Load player history data
    player_history = pd.read_csv(os.path.join("dataset", "player_history.csv"))
    
    # Prepare data for TFT
    prepared_data = prepare_player_data_for_tft(player_history)
    
    # Integrate TFT with ensemble
    if prepared_data is not None:
        success = integrate_tft_with_ensemble(hybrid_model, prepared_data)
        
        if success:
            # Split data for evaluation
            from sklearn.model_selection import train_test_split
            train_data, test_data = train_test_split(prepared_data, test_size=0.2, random_state=42)
            
            # Evaluate TFT performance
            metrics = evaluate_tft_performance(hybrid_model, test_data)
            
            # Print improvement summary
            if 'r2_improvement' in metrics:
                print(f"\nTFT Model R² Improvement: {metrics['r2_improvement']*100:.2f}%")
                print(f"Previous Ensemble R²: {metrics.get('ensemble_r2', 0):.4f}")
                print(f"TFT Model R²: {metrics.get('tft_r2', 0):.4f}")