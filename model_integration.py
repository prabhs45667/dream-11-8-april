import pandas as pd
import numpy as np
import os
import sys
import logging
import random  # For mock sentiment
from pitch_specific_models import PitchSpecificModelTrainer
from feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_integration')

# Simple mock NLP enhancer that doesn't require external dependencies
class MockNlpEnhancer:
    """
    A mock NLP enhancer that simulates sentiment analysis without external dependencies.
    This is used when the full NlpFeatureEnhancer is not available due to missing dependencies.
    """
    def __init__(self):
        logger.info("Initializing Mock NLP Enhancer (no external dependencies required)")
        # Players with simulated positive sentiment
        self.positive_players = ['Virat Kohli', 'MS Dhoni', 'Jasprit Bumrah', 'Rohit Sharma', 'KL Rahul']
        # Players with simulated negative sentiment
        self.negative_players = ['Rishabh Pant', 'Shreyas Iyer']
        # Players with simulated injuries
        self.injured_players = ['Hardik Pandya']
        
    def run_pipeline(self, predictions_df):
        """
        Apply mock NLP adjustments to player predictions.
        
        Args:
            predictions_df: DataFrame with player predictions
            
        Returns:
            DataFrame with mock NLP adjustments
        """
        if not isinstance(predictions_df, pd.DataFrame):
            logger.error("Invalid input to mock NLP enhancer")
            return predictions_df
            
        logger.info("Applying mock NLP sentiment analysis to predictions")
        enhanced_df = predictions_df.copy()
        
        # Add NLP columns
        enhanced_df['nlp_sentiment_score'] = 0.0
        enhanced_df['nlp_adjustment_factor'] = 1.0
        enhanced_df['nlp_is_injured'] = False
        
        # Get player name column
        player_column = 'Player Name' if 'Player Name' in enhanced_df.columns else 'player_name'
        if player_column not in enhanced_df.columns:
            logger.warning("No player name column found in predictions DataFrame")
            return enhanced_df
            
        # Apply adjustments based on player names
        total_adjustments = 0
        
        for idx, row in enhanced_df.iterrows():
            player_name = row[player_column]
            
            # Apply positive sentiment
            if any(pos_player.lower() in player_name.lower() for pos_player in self.positive_players):
                enhanced_df.loc[idx, 'nlp_sentiment_score'] = random.uniform(0.6, 0.9)
                enhanced_df.loc[idx, 'nlp_adjustment_factor'] = 1.10  # +10%
                enhanced_df.loc[idx, 'predicted_points'] *= 1.10
                total_adjustments += 1
                
            # Apply negative sentiment
            elif any(neg_player.lower() in player_name.lower() for neg_player in self.negative_players):
                enhanced_df.loc[idx, 'nlp_sentiment_score'] = random.uniform(-0.9, -0.6)
                enhanced_df.loc[idx, 'nlp_adjustment_factor'] = 0.90  # -10%
                enhanced_df.loc[idx, 'predicted_points'] *= 0.90
                total_adjustments += 1
                
            # Apply injury flag
            if any(inj_player.lower() in player_name.lower() for inj_player in self.injured_players):
                enhanced_df.loc[idx, 'nlp_is_injured'] = True
                enhanced_df.loc[idx, 'nlp_adjustment_factor'] = 0.70  # -30% for injured
                enhanced_df.loc[idx, 'predicted_points'] *= 0.70
                total_adjustments += 1
                
        logger.info(f"Mock NLP enhancement applied {total_adjustments} adjustments to player predictions")
        return enhanced_df

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
        
        # Initialize NLP Feature Enhancer for sentiment analysis
        try:
            # Try to import the full NLP enhancer first
            from nlp_feature_enhancer import NlpFeatureEnhancer
            self.nlp_enhancer = NlpFeatureEnhancer()
            logger.info("Full NLP Feature Enhancer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing NLP enhancer: {e}")
            logger.info("Using mock NLP enhancer instead (simulated sentiment)")
            self.nlp_enhancer = MockNlpEnhancer()
        
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
            
    def get_available_pitch_types(self):
        """Return the list of pitch types for which models are available."""
        # Ensure models are loaded
        if not hasattr(self.model_trainer, 'models') or not self.model_trainer.models:
            logger.warning("Models not loaded yet in model_trainer. Attempting to load.")
            self.load_or_train_models()
        
        # Check again after loading
        if not hasattr(self.model_trainer, 'models') or not self.model_trainer.models:
            logger.error("Models dictionary is still missing or empty after load attempt.")
            return [] # Return empty list if models cannot be confirmed
            
        # Return keys excluding 'fallback' as it's not a user-selectable pitch type
        available_types = [ptype for ptype in self.model_trainer.models.keys() if ptype != 'fallback']
        logger.debug(f"Found available pitch types: {available_types}")
        return available_types
            
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
                    # Log fallback predictions
                    logger.info(f"Fallback Predictions:\n{result_df[['player_name', 'predicted_points']].to_string()}") # LOG PREDICTIONS
            else:
                 logger.info(f"Successfully predicted points using the '{pitch_type}' model.")
                 # Log pitch-specific predictions
                 logger.info(f"'{pitch_type}' Predictions:\n{result_df[['player_name', 'predicted_points']].to_string()}") # LOG PREDICTIONS

            # Merge predictions back into the original structure if needed, 
            # or just return the result_df which now contains predictions.
            # Ensure player identifier is present for potential merging later if necessary.
            if 'player_name' not in result_df.columns and 'Player Name' in result_df.columns:
                 result_df['player_name'] = result_df['Player Name']
            
            # Apply NLP enhancement to adjust predictions based on news sentiment
            try:
                logger.info("Applying NLP enhancement to adjust predictions based on news sentiment")
                enhanced_df = self.nlp_enhancer.run_pipeline(result_df)
                
                if enhanced_df is not None and not enhanced_df.empty:
                    logger.info(f"NLP enhancement applied successfully to {len(enhanced_df)} player predictions")
                    
                    # Log some examples of adjustments
                    adjusted_players = enhanced_df[enhanced_df['nlp_adjustment_factor'] != 1.0]
                    if not adjusted_players.empty:
                        sample_size = min(5, len(adjusted_players))
                        sample_df = adjusted_players.head(sample_size)
                        for _, row in sample_df.iterrows():
                            player_name = row.get('player_name', row.get('Player Name', 'Unknown'))
                            factor = row.get('nlp_adjustment_factor', 1.0)
                            is_injured = row.get('nlp_is_injured', False)
                            logger.info(f"  - {player_name}: adjustment factor={factor:.2f}, injured={is_injured}")
                    
                    return enhanced_df
                else:
                    logger.warning("NLP enhancement returned empty DataFrame. Using original predictions.")
                    return result_df
            except Exception as e:
                logger.error(f"Error during NLP enhancement: {str(e)}, using original predictions")
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

    def prepare_optimization_problem(self, player_data, match_info):
        """
        Prepare the optimization problem structure for team selection
        
        Args:
            player_data (pd.DataFrame): Player data with predicted points
            match_info (dict): Match information including venue and teams
            
        Returns:
            dict: Problem structure for the team optimizer
        """
        try:
            # Ensure we have predicted points
            if 'predicted_points' not in player_data.columns or player_data['predicted_points'].isnull().all():
                logger.error("Player data missing predicted points, cannot prepare optimization problem")
                return None
                
            # Extract role requirements based on pitch type
            role_requirements = self.identify_team_balance_requirements(match_info)
            if not role_requirements:
                logger.warning("Could not determine role requirements. Using default requirements.")
                role_requirements = {
                    'WK': {'min': 1, 'max': 4},
                    'BAT': {'min': 3, 'max': 5},
                    'AR': {'min': 1, 'max': 4},
                    'BOWL': {'min': 3, 'max': 5}
                }
            
            # Convert role requirements from dict of dicts to dict of tuples (min, max)
            role_reqs = {
                role: (requirements['min'], requirements['max'])
                for role, requirements in role_requirements.items()
            }
                
            # Extract team names
            home_team = match_info.get('home_team')
            away_team = match_info.get('away_team')
            if not home_team or not away_team:
                logger.warning("Missing home_team or away_team in match_info. Using default values.")
                home_team = 'Team1'
                away_team = 'Team2'
                
            # Create players dictionary
            players_dict = {}
            for idx, player in player_data.iterrows():
                # Try different column names to handle various data formats
                player_name = player.get('Player Name', player.get('player_name', player.get('Player', str(idx))))
                team = player.get('Team', player.get('team', 'Unknown'))
                role = player.get('role', player.get('Role', player.get('Player Type', 'BAT')))
                credits = float(player.get('Credits', player.get('credits', 8.0)))
                points = float(player.get('predicted_points', 0))
                
                # Normalize role to standard format
                std_role = self._standardize_role(role)
                
                # Create player entry
                players_dict[player_name] = {
                    'name': player_name,
                    'team': team,
                    'role': std_role,
                    'credits': credits,
                    'points': points
                }
                
                # Add form and consistency if available
                if 'fantasy_points_last_5' in player:
                    players_dict[player_name]['fantasy_points_last_5'] = player['fantasy_points_last_5']
                if 'fantasy_points_std_dev' in player:
                    players_dict[player_name]['fantasy_points_std_dev'] = player['fantasy_points_std_dev']
                    
            # Create problem structure
            problem = {
                'players': players_dict,
                'role_requirements': role_reqs,
                'max_credits': 100,
                'team_ratio_limit': 7,
                'max_players': 11
            }
            
            logger.info(f"Prepared optimization problem with {len(players_dict)} players")
            return problem
            
        except Exception as e:
            logger.error(f"Error preparing optimization problem: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def _standardize_role(self, role):
        """Standardize player role to one of: WK, BAT, AR, BOWL"""
        role_str = str(role).upper()
        if 'WK' in role_str or 'KEEPER' in role_str:
            return 'WK'
        elif 'BAT' in role_str:
            return 'BAT'
        elif 'BOWL' in role_str:
            return 'BOWL'
        elif 'ALL' in role_str or 'AR' in role_str:
            return 'AR'
        else:
            return 'BAT'  # Default to batsman

    def generate_team(self, player_data, match_info):
        """
        Generate a team using optimization or fallback to greedy selection
        
        Args:
            player_data (pd.DataFrame): Player data with predicted points
            match_info (dict): Match information including venue and teams
            
        Returns:
            dict: Selected team result
        """
        try:
            # Prepare optimization problem
            problem = self.prepare_optimization_problem(player_data, match_info)
            if not problem:
                logger.error("Failed to prepare optimization problem")
                return None
                
            # Initialize team optimizer
            from team_optimizer import TeamOptimizer
            optimizer = TeamOptimizer()
            
            # Attempt optimization
            logger.info("Attempting team optimization...")
            team_result = optimizer.optimize_team(problem)
            
            if team_result and 'players' in team_result and team_result['players']:
                logger.info("Team optimization successful")
                return team_result
                
            # If optimization fails, try greedy approach
            logger.warning("Optimization failed, falling back to greedy selection")
            home_team = match_info.get('home_team', 'Team1')
            away_team = match_info.get('away_team', 'Team2')
            
            # Prepare role requirements for greedy selection
            role_requirements = {}
            if 'role_requirements' in problem:
                role_requirements = problem['role_requirements']
            
            # Perform greedy selection, passing the player_data directly (not the nested problem)
            # This avoids the issue with "missing 'players' key" in the fallback
            greedy_result = optimizer._greedy_team_selection(
                player_data,  # Pass the original player_data DataFrame instead of problem
                home_team, 
                away_team, 
                role_requirements
            )
            
            if greedy_result:
                logger.info("Greedy selection successful")
                return greedy_result
            else:
                logger.error("Greedy selection also failed")
                return None
                
        except Exception as e:
            logger.error(f"Error generating team: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

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
        display_cols = ['Player Name', 'Player Type', 'Team', 'Credits', 'predicted_points']
        
        # Check if NLP columns exist
        nlp_cols = ['nlp_sentiment_score', 'nlp_adjustment_factor', 'nlp_is_injured']
        has_nlp_data = all(col in predicted_df_balanced.columns for col in nlp_cols)
        
        if has_nlp_data:
            display_cols.extend(nlp_cols)
            print("NLP enhancement detected in predictions!")
            
            # Display players with NLP adjustments
            adjusted_players = predicted_df_balanced[predicted_df_balanced['nlp_adjustment_factor'] != 1.0]
            if not adjusted_players.empty:
                print("\nPlayers with NLP adjustments:")
                for _, row in adjusted_players.iterrows():
                    player_name = row.get('Player Name', row.get('player_name', 'Unknown'))
                    factor = row.get('nlp_adjustment_factor', 1.0)
                    sentiment = row.get('nlp_sentiment_score', 0.0)
                    is_injured = row.get('nlp_is_injured', False)
                    print(f"  - {player_name}: adjustment={factor:.2f}, sentiment={sentiment:.2f}, injured={is_injured}")
        
        print(predicted_df_balanced[display_cols].sort_values('predicted_points', ascending=False).to_string())
        
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
        
        display_cols = ['Player Name', 'Player Type', 'Team', 'Credits', 'predicted_points']
        
        # Check if NLP columns exist
        nlp_cols = ['nlp_sentiment_score', 'nlp_adjustment_factor', 'nlp_is_injured']
        has_nlp_data = all(col in predicted_df_batting.columns for col in nlp_cols)
        
        if has_nlp_data:
            display_cols.extend(nlp_cols)
        
        print(predicted_df_batting[display_cols].sort_values('predicted_points', ascending=False).to_string())
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