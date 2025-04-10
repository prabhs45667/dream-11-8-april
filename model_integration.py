import pandas as pd
import numpy as np
import os
import sys
import logging
from pitch_specific_models import PitchSpecificModelTrainer
from feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_integration')

class Dream11ModelIntegrator:
    """
    Integrates pitch-specific models (with fallback) with the Dream11 application
    """
    
    def __init__(self, data_dir="dataset", models_dir="models"):
        """
        Initialize the model integrator
        
        Args:
            data_dir (str): Directory containing data files
            models_dir (str): Directory containing model files
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        
        # Initialize model trainer (which handles loading/training models)
        self.model_trainer = PitchSpecificModelTrainer(data_dir=data_dir, models_dir=models_dir)
        
        # Feature engineer needed for prediction if pitch-specific models are used
        self.feature_engineer = FeatureEngineer(data_dir=data_dir) 
        
        # Ensure models are loaded or trained on initialization
        self.load_or_train_models()
        
    def load_or_train_models(self):
        """Load trained models or train new ones if not found"""
        logger.info("Attempting to load models...")
        models_loaded = self.model_trainer.load_models()
        
        if not models_loaded:
            logger.warning("Failed to load one or more models. Training new models.")
            self.train_models()
        else:
            logger.info("All required models loaded successfully.")
            # Verify fallback model is loaded
            if self.model_trainer.models.get('fallback') is None:
                 logger.warning("Fallback model specifically not found. Retraining needed.")
                 self.train_models()

    def train_models(self):
        """Train new pitch-specific and fallback models"""
        logger.info("Training new pitch-specific and fallback models")
        results = self.model_trainer.train_models()
        
        if results:
            logger.info("Models trained successfully")
            # Optionally log metrics
        else:
            logger.error("Failed to train models")
            # Consider how to handle this - maybe raise an exception?
            
    def predict_player_points(self, player_data, match_info):
        """
        Predict fantasy points for players, using pitch-specific model first,
        then the trained fallback model if needed.
        
        Args:
            player_data (pd.DataFrame): Player data with features
            match_info (dict): Match information including venue and teams
            
        Returns:
            pd.DataFrame: Player data with predicted points, or original data with NaNs if all predictions fail.
        """
        try:
            # Extract match information
            venue = match_info.get('venue', 'default')
            home_team = match_info.get('home_team', None)
            away_team = match_info.get('away_team', None)
            pitch_type = match_info.get('pitch_type', 'balanced')
            
            # Standardize common input column names needed for prediction/feature eng
            df = player_data.copy()
            field_mapping = {
                'Player Name': 'player_name',
                'Player Type': 'role',
                'Team': 'team',
                'Credits': 'credits'
            }
            for source, target in field_mapping.items():
                if source in df.columns and target not in df.columns:
                    df[target] = df[source]
            
            # Add match info needed by feature engineer / prediction
            df['venue'] = venue
            df['home_team'] = home_team
            df['away_team'] = away_team
            df['pitch_type'] = pitch_type
            
            logger.info(f"Attempting prediction using '{pitch_type}' model.")
            # First, try predicting with the pitch-specific model
            result_df = self.model_trainer.predict(df.copy(), pitch_type=pitch_type, use_fallback=False)

            # Check if prediction failed (returned original data or has NaNs)
            prediction_failed = result_df is None or result_df['predicted_points'].isnull().all()

            if prediction_failed:
                logger.warning(f"Prediction failed with '{pitch_type}' model. Attempting fallback model.")
                # Try predicting with the fallback model
                result_df = self.model_trainer.predict(df.copy(), use_fallback=True)
                
                # Check if fallback prediction also failed
                fallback_failed = result_df is None or result_df['predicted_points'].isnull().all()
                if fallback_failed:
                    logger.error("Fallback prediction also failed. Returning data without predictions.")
                    # Return original data frame but ensure predicted_points column exists with NaNs
                    player_data['predicted_points'] = np.nan
                    return player_data 
                else:
                    logger.info("Successfully predicted points using the fallback model.")
            else:
                 logger.info(f"Successfully predicted points using the '{pitch_type}' model.")

            # Merge predictions back into the original structure if needed, 
            # or just return the result_df which now contains predictions.
            # Ensure player identifier is present for potential merging later if necessary.
            if 'player_name' not in result_df.columns and 'Player Name' in result_df.columns:
                 result_df['player_name'] = result_df['Player Name']
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error during player point prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return original data with NaNs in case of unexpected error
            player_data['predicted_points'] = np.nan
            return player_data
            
    def identify_team_balance_requirements(self, match_info):
        """
        Identify the optimal team balance based on match conditions
        (This logic remains the same as before)
        
        Args:
            match_info (dict): Match information including venue and teams
            
        Returns:
            dict: Team balance requirements
        """
        try:
            # Extract match information
            venue = match_info.get('venue', 'default')
            pitch_type = match_info.get('pitch_type', 'balanced')
            
            # Default balance requirements
            balance_requirements = {
                'balanced': {
                    'WK': {'min': 1, 'max': 2, 'optimal': 1},
                    'BAT': {'min': 3, 'max': 5, 'optimal': 4},
                    'AR': {'min': 1, 'max': 4, 'optimal': 3},
                    'BOWL': {'min': 3, 'max': 5, 'optimal': 3}
                },
                'batting_friendly': {
                    'WK': {'min': 1, 'max': 3, 'optimal': 2},
                    'BAT': {'min': 4, 'max': 6, 'optimal': 5},
                    'AR': {'min': 1, 'max': 3, 'optimal': 2},
                    'BOWL': {'min': 2, 'max': 4, 'optimal': 2}
                },
                'bowling_friendly': {
                    'WK': {'min': 1, 'max': 2, 'optimal': 1},
                    'BAT': {'min': 3, 'max': 4, 'optimal': 3},
                    'AR': {'min': 1, 'max': 3, 'optimal': 2},
                    'BOWL': {'min': 4, 'max': 6, 'optimal': 5}
                }
            }
            
            # Get requirements for the specified pitch type
            if pitch_type not in balance_requirements:
                logger.warning(f"Unknown pitch type: {pitch_type}. Using 'balanced' instead.")
                pitch_type = 'balanced'
                
            requirements = balance_requirements[pitch_type]
            
            logger.info(f"Team balance requirements for {pitch_type} pitch: "
                       f"WK={requirements['WK']['optimal']}, "
                       f"BAT={requirements['BAT']['optimal']}, "
                       f"AR={requirements['AR']['optimal']}, "
                       f"BOWL={requirements['BOWL']['optimal']}")
                       
            return requirements
            
        except Exception as e:
            logger.error(f"Error identifying team balance requirements: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def suggest_captain_vice_captain(self, selected_team, match_info):
        """
        Suggest captain and vice-captain based on match conditions and predicted points.
        (This logic remains the same as before)
        
        Args:
            selected_team (pd.DataFrame): DataFrame of selected players with predicted_points
            match_info (dict): Match information including venue and teams
            
        Returns:
            tuple: (captain_player_series, vice_captain_player_series) or (None, None)
        """
        try:
            # Extract match information
            pitch_type = match_info.get('pitch_type', 'balanced')
            
            # Role weights for different pitch types
            role_weights = {
                'balanced': {
                    'WK': 0.8, 'BAT': 1.0, 'AR': 1.1, 'BOWL': 1.0,
                    'Wicket Keeper': 0.8, 'Batsman': 1.0, 'All Rounder': 1.1, 'Bowler': 1.0 
                },
                'batting_friendly': {
                    'WK': 1.0, 'BAT': 1.2, 'AR': 1.0, 'BOWL': 0.7,
                     'Wicket Keeper': 1.0, 'Batsman': 1.2, 'All Rounder': 1.0, 'Bowler': 0.7
                },
                'bowling_friendly': {
                    'WK': 0.7, 'BAT': 0.8, 'AR': 1.0, 'BOWL': 1.2,
                     'Wicket Keeper': 0.7, 'Batsman': 0.8, 'All Rounder': 1.0, 'Bowler': 1.2
                }
            }
            
            # Get weights for the specified pitch type
            if pitch_type not in role_weights:
                logger.warning(f"Unknown pitch type: {pitch_type}. Using 'balanced' weights.")
                weights = role_weights['balanced']
            else:
                 weights = role_weights[pitch_type]
            
            # Ensure selected_team is a DataFrame
            if not isinstance(selected_team, pd.DataFrame):
                logger.error("selected_team must be a Pandas DataFrame.")
                return None, None
            
            team_df = selected_team.copy()

            # Ensure we have required columns
            if 'predicted_points' not in team_df.columns or team_df['predicted_points'].isnull().all():
                logger.warning("No valid predicted points found in selected team. Cannot suggest C/VC.")
                return None, None
                
            # Standardize role column if needed
            if 'role' not in team_df.columns and 'Player Type' in team_df.columns:
                team_df['role'] = team_df['Player Type']
            elif 'role' not in team_df.columns:
                 logger.warning("Role information missing. Cannot apply role weights for C/VC.")
                 # Proceed without weights if role is missing
                 team_df['captain_score'] = team_df['predicted_points']
            else:
                 # Apply role weights safely
                 team_df['captain_score'] = team_df.apply(
                    lambda x: x['predicted_points'] * weights.get(x['role'], 1.0), # Default weight 1.0 if role not in map
                    axis=1
                 )

            # Handle potential NaN scores after weighting
            team_df.dropna(subset=['captain_score'], inplace=True)
            
            # Sort by captain score
            captain_candidates = team_df.sort_values('captain_score', ascending=False)
            
            # Select captain and vice-captain
            if len(captain_candidates) >= 2:
                captain = captain_candidates.iloc[0]
                vice_captain = captain_candidates.iloc[1]
                
                # Extract player names for logging
                captain_name = captain.get('player_name', captain.get('Player Name', 'Unknown'))
                vice_captain_name = vice_captain.get('player_name', vice_captain.get('Player Name', 'Unknown'))
                
                logger.info(f"Suggested Captain: {captain_name}, Vice-Captain: {vice_captain_name} based on {pitch_type} weights.")
                
                return captain, vice_captain
            elif len(captain_candidates) == 1:
                 captain = captain_candidates.iloc[0]
                 logger.warning("Only one player available, suggesting as Captain, no Vice-Captain.")
                 return captain, None
            else:
                logger.warning("Not enough players with valid scores to select captain and vice-captain")
                return None, None
                
        except Exception as e:
            logger.error(f"Error suggesting captain and vice-captain: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

# Main function for testing integration logic
def main():
    # Initialize integrator
    integrator = Dream11ModelIntegrator()
    
    # Test with sample data
    # Create sample player data
    sample_players = pd.DataFrame({
        'Player Name': ['Virat Kohli', 'Rohit Sharma', 'Jasprit Bumrah', 'MS Dhoni', 'Hardik Pandya',
                       'Ravindra Jadeja', 'KL Rahul', 'Mohammed Shami', 'Rishabh Pant', 'Shreyas Iyer'],
        'Team': ['RCB', 'MI', 'MI', 'CSK', 'MI', 'CSK', 'PBKS', 'GT', 'DC', 'KKR'],
        'Player Type': ['BAT', 'BAT', 'BOWL', 'WK', 'AR', 'AR', 'BAT', 'BOWL', 'WK', 'BAT'],
        'Credits': [10.0, 9.5, 9.0, 8.5, 9.5, 9.0, 8.5, 8.0, 8.5, 8.0]
    })
    
    # Sample match info
    match_info_balanced = {
        'venue': 'MA Chidambaram Stadium',
        'home_team': 'CSK',
        'away_team': 'MI',
        'pitch_type': 'balanced'
    }
    
    # Predict player points
    predicted_df_balanced = integrator.predict_player_points(sample_players, match_info_balanced)
    
    if predicted_df_balanced is not None and 'predicted_points' in predicted_df_balanced.columns:
        print("\nPredicted Player Points (Balanced Pitch):")
        print(predicted_df_balanced[['Player Name', 'Player Type', 'Team', 'Credits', 'predicted_points']].sort_values('predicted_points', ascending=False).to_string())
        
        # Suggest C/VC
        captain, vc = integrator.suggest_captain_vice_captain(predicted_df_balanced, match_info_balanced)
        if captain is not None:
            print(f"\nSuggested Captain (Balanced): {captain.get('Player Name')}")
        if vc is not None:
             print(f"Suggested VC (Balanced): {vc.get('Player Name')}")

    else:
         print("\nPrediction failed for Balanced Pitch.")
    
    # Test batting-friendly pitch
    match_info_batting = {
        'venue': 'Wankhede Stadium',
        'home_team': 'MI', 
        'away_team': 'RCB',
        'pitch_type': 'batting_friendly'
    }
    predicted_df_batting = integrator.predict_player_points(sample_players, match_info_batting)
    
    if predicted_df_batting is not None and 'predicted_points' in predicted_df_batting.columns:
        print("\nPredicted Player Points (Batting-Friendly Pitch):")
        print(predicted_df_batting[['Player Name', 'Player Type', 'Team', 'Credits', 'predicted_points']].sort_values('predicted_points', ascending=False).to_string())
    else:
         print("\nPrediction failed for Batting-Friendly Pitch.")

    # Test edge case: Force fallback (e.g., if pitch type model was known bad)
    print("\nTesting Fallback Model explicitly:")
    predicted_df_fallback = integrator.model_trainer.predict(sample_players.copy(), use_fallback=True)
    if predicted_df_fallback is not None and 'predicted_points' in predicted_df_fallback.columns:
         print("\nPredicted Player Points (Fallback Model):")
         print(predicted_df_fallback[['Player Name', 'Player Type', 'Team', 'Credits', 'predicted_points']].sort_values('predicted_points', ascending=False).to_string())
    else:
         print("\nPrediction failed using Fallback Model.")

    # Test team balance requirements (no change needed here)
    # ...
    
if __name__ == "__main__":
    # Need feature_engineering.py to exist for this test to run
    try:
         # Check if feature_engineering.py exists before running main
         if os.path.exists("feature_engineering.py"):
              main()
         else:
              print("Skipping model_integration test: feature_engineering.py not found.")
    except ImportError as e:
         print(f"Skipping model_integration test due to import error: {e}")
    except Exception as e:
        print(f"An error occurred during model_integration test: {e}") 