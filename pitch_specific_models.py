import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression  # Added for fallback
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pitch_specific_models')

class PitchSpecificModelTrainer:
    """
    Trains specialized machine learning models for different pitch types 
    and a baseline fallback model to predict fantasy cricket points
    """
    
    def __init__(self, data_dir="dataset", models_dir="models"):
        """
        Initialize the pitch-specific model trainer
        
        Args:
            data_dir (str): Directory containing training data
            models_dir (str): Directory to save trained models
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize the feature engineer
        self.feature_engineer = FeatureEngineer(data_dir=data_dir)
        
        # Initialize models dictionary (including fallback)
        self.models = {
            'balanced': None,
            'batting_friendly': None,
            'bowling_friendly': None,
            'fallback': None  # Added for the baseline model
        }
        
    def load_training_data(self, match_data_file=None, squad_file=None):
        """
        Load and prepare training data
        
        Args:
            match_data_file (str): Path to match data file with fantasy points
            squad_file (str): Path to squad data file with player details
            
        Returns:
            pd.DataFrame: DataFrame containing enhanced player data for training
        """
        if match_data_file is None:
            match_data_file = os.path.join(self.data_dir, "fantasy_match_data.csv")
            
        if squad_file is None:
            squad_file = os.path.join(self.data_dir, "SquadPlayerNames_IndianT20League - SquadData_AllTeams.csv")
            
        try:
            # Load match data with fantasy points if available
            if os.path.exists(match_data_file):
                match_df = pd.read_csv(match_data_file)
                logger.info(f"Loaded match data from {match_data_file}: {match_df.shape[0]} records")
            else:
                logger.warning(f"Match data file not found: {match_data_file}")
                
                # Create synthetic match data if real data is not available
                # This is just for development and testing purposes
                match_df = self._create_synthetic_match_data(squad_file)
                logger.info(f"Created synthetic match data: {match_df.shape[0]} records")
                
            # Combine all pitch data for fallback model training later
            self.combined_data_for_fallback = match_df.copy()
                
            # Apply feature engineering to the data
            # We'll create three versions of the data, one for each pitch type
            data_by_pitch_type = {}
            
            # Home and away teams for feature engineering
            all_teams = match_df['team'].unique() if 'team' in match_df.columns else ['CSK', 'MI', 'RCB', 'KKR']
            home_team = all_teams[0] if len(all_teams) > 0 else 'CSK'
            away_team = all_teams[1] if len(all_teams) > 1 else 'MI'
            
            # Stadiums for feature engineering
            stadiums = {
                'balanced': 'MA Chidambaram Stadium',
                'batting_friendly': 'Wankhede Stadium',
                'bowling_friendly': 'Eden Gardens'
            }
            
            # Generate enhanced data for each pitch type
            for pitch_type, venue in stadiums.items():
                # Clone the match data
                pitch_df = match_df.copy()
                
                # Add pitch type column
                pitch_df['pitch_type'] = pitch_type
                
                # Apply feature engineering
                enhanced_df = self.feature_engineer.enhance_player_features(
                    pitch_df, home_team=home_team, away_team=away_team, venue=venue
                )
                
                # Store the enhanced data
                data_by_pitch_type[pitch_type] = enhanced_df
                
            return data_by_pitch_type
            
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def _create_synthetic_match_data(self, squad_file):
        """
        Create synthetic match data for training when real data is unavailable
        
        Args:
            squad_file (str): Path to squad data file
            
        Returns:
            pd.DataFrame: Synthetic match data
        """
        try:
            # Load squad data
            squad_df = pd.read_csv(squad_file)
            logger.info(f"Loaded squad data from {squad_file}: {squad_df.shape[0]} players")
            
            # Standardize column names
            if 'Player Name' in squad_df.columns:
                squad_df.rename(columns={'Player Name': 'player_name'}, inplace=True)
                
            if 'Player Type' in squad_df.columns:
                squad_df.rename(columns={'Player Type': 'role'}, inplace=True)
                
            if 'Credits' in squad_df.columns:
                squad_df.rename(columns={'Credits': 'credits'}, inplace=True)
                
            if 'Team' in squad_df.columns:
                squad_df.rename(columns={'Team': 'team'}, inplace=True)
                
            # Define role factors for different pitch types
            role_factors = {
                'balanced': {
                    'BAT': 12, 'BOWL': 12, 'AR': 13, 'WK': 11,
                    'Batsman': 12, 'Bowler': 12, 'All Rounder': 13, 'Wicket Keeper': 11
                },
                'batting_friendly': {
                    'BAT': 15, 'BOWL': 9, 'AR': 12, 'WK': 14,
                    'Batsman': 15, 'Bowler': 9, 'All Rounder': 12, 'Wicket Keeper': 14
                },
                'bowling_friendly': {
                    'BAT': 9, 'BOWL': 15, 'AR': 12, 'WK': 8,
                    'Batsman': 9, 'Bowler': 15, 'All Rounder': 12, 'Wicket Keeper': 8
                }
            }
            
            # Create synthetic match data records
            matches = []
            
            # Generate 3 matches for each team combination with different pitch types
            teams = squad_df['team'].unique()
            pitch_types = ['balanced', 'batting_friendly', 'bowling_friendly']
            
            for home_team in teams:
                for away_team in teams:
                    if home_team != away_team:
                        for pitch_type in pitch_types:
                            # Filter players from both teams
                            team_players = squad_df[
                                (squad_df['team'] == home_team) | (squad_df['team'] == away_team)
                            ].copy()
                            
                            # Generate fantasy points based on credits, role, and pitch type
                            for idx, player in team_players.iterrows():
                                # Get role factor based on pitch type
                                role = player['role']
                                role_factor = role_factors[pitch_type].get(role, 12)
                                
                                # Generate fantasy points with some randomness
                                # base_points = credits * role_factor * random_factor
                                base_points = player['credits'] * role_factor * (0.9 + 0.2 * np.random.random())
                                
                                # Add noise to make it realistic
                                noise = np.random.normal(0, 10)
                                fantasy_points = max(0, base_points + noise)
                                
                                # Create match record
                                match_record = {
                                    'match_id': f"{home_team}_vs_{away_team}_{pitch_type}",
                                    'player_name': player['player_name'],
                                    'team': player['team'],
                                    'role': player['role'],
                                    'credits': player['credits'],
                                    'fantasy_points': fantasy_points,
                                    'venue': f"{home_team} Stadium",
                                    'pitch_type': pitch_type,
                                    'home_team': home_team,
                                    'away_team': away_team
                                }
                                
                                matches.append(match_record)
            
            # Create DataFrame
            match_df = pd.DataFrame(matches)
            
            return match_df
            
        except Exception as e:
            logger.error(f"Error creating synthetic match data: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
            
    def prepare_features_targets(self, data_by_pitch_type):
        """
        Prepare features and targets for model training
        
        Args:
            data_by_pitch_type (dict): Dictionary of DataFrames by pitch type
            
        Returns:
            dict: Dictionary containing X_train, X_test, y_train, y_test for each pitch type
        """
        try:
            features_targets = {}
            
            # Define base features for fallback model
            fallback_base_cols = ['credits']
            fallback_cat_cols = ['role', 'team']
            fallback_feature_cols = fallback_base_cols + fallback_cat_cols

            combined_X_list = []
            combined_y_list = []

            for pitch_type, df in data_by_pitch_type.items():
                logger.info(f"Preparing features for {pitch_type} pitch type")
                
                # Define feature columns to use for pitch-specific models
                categorical_cols = ['role', 'pitch_type', 'team']
                role_indicators = [col for col in df.columns if col.startswith('is_')]
                team_features = [col for col in df.columns if col.startswith('vs_team_')]
                venue_features = [col for col in df.columns if col in [
                    'venue_advantage', 'pitch_factor', 'pitch_is_batting_friendly',
                    'pitch_is_bowling_friendly', 'pitch_is_balanced'
                ]]
                form_features = [col for col in df.columns if col in [
                    'recent_form', 'last_3_avg', 'last_5_avg', 'form_trend'
                ]]
                interaction_features = [col for col in df.columns if '_interaction' in col]
                numeric_cols = ['credits']
                
                # Combine all features for pitch-specific model
                feature_cols = numeric_cols + categorical_cols + role_indicators + \
                               team_features + venue_features + form_features + interaction_features
                feature_cols = list(set(col for col in feature_cols if col in df.columns))
                
                # Target column
                target_col = 'fantasy_points'
                if target_col not in df.columns:
                    logger.error(f"Target column '{target_col}' not found in data for {pitch_type}")
                    continue
                
                # Extract features and target for pitch-specific model
                X = df[feature_cols]
                y = df[target_col]
                X = pd.get_dummies(X, columns=[c for c in categorical_cols if c in X.columns], drop_first=True)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Store for pitch-specific model
                features_targets[pitch_type] = {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'feature_cols': feature_cols
                }
                logger.info(f"{pitch_type} features: {len(feature_cols)} columns, {X_train.shape[0]} training samples")

                # Prepare data for combined fallback model
                fallback_cols_present = [col for col in fallback_feature_cols if col in df.columns]
                if set(fallback_cols_present) == set(fallback_feature_cols):
                    X_fallback = df[fallback_feature_cols]
                    y_fallback = df[target_col]
                    X_fallback = pd.get_dummies(X_fallback, columns=[c for c in fallback_cat_cols if c in X_fallback.columns], drop_first=True)
                    combined_X_list.append(X_fallback)
                    combined_y_list.append(y_fallback)
                else:
                    logger.warning(f"Missing base columns for fallback model in {pitch_type} data.")

            # Prepare combined data for fallback model
            if combined_X_list:
                X_combined = pd.concat(combined_X_list, ignore_index=True)
                y_combined = pd.concat(combined_y_list, ignore_index=True)
                
                # Align columns (handle cases where some dummy vars might be missing in splits)
                X_combined = X_combined.fillna(0)
                X_train_fb, X_test_fb, y_train_fb, y_test_fb = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)
                
                # Store for fallback model
                features_targets['fallback'] = {
                    'X_train': X_train_fb,
                    'X_test': X_test_fb,
                    'y_train': y_train_fb,
                    'y_test': y_test_fb,
                    'feature_cols': fallback_feature_cols
                }
                logger.info(f"Fallback features prepared using {X_train_fb.shape[0]} training samples")
            else:
                logger.error("Could not prepare data for the fallback model.")
                
            return features_targets
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def train_models(self, features_targets=None, data=None):
        """
        Train machine learning models for each pitch type and a fallback model
        
        Args:
            features_targets (dict): Dictionary of prepared features and targets
            data (dict): Dictionary of raw data by pitch type, used if features_targets is None
            
        Returns:
            dict: Results with model performance metrics
        """
        # Prepare features if not provided
        if features_targets is None:
            if data is None:
                data = self.load_training_data()
            
            if data is None:
                logger.error("No training data available")
                return None
                
            features_targets = self.prepare_features_targets(data)
            
        if features_targets is None:
            logger.error("Failed to prepare features and targets")
            return None
            
        results = {}
        
        try:
            # Train models for each pitch type + fallback
            for model_key, ft_dict in features_targets.items():
                logger.info(f"Training model for {model_key} type")
                
                X_train = ft_dict['X_train']
                y_train = ft_dict['y_train']
                X_test = ft_dict['X_test']
                y_test = ft_dict['y_test']
                
                # Select model type based on key
                if model_key == 'fallback':
                    # Simple Linear Regression for fallback
                    model = LinearRegression()
                elif model_key == 'balanced':
                    model = RandomForestRegressor(
                        n_estimators=100, max_depth=15, min_samples_split=5,
                        min_samples_leaf=2, random_state=42, n_jobs=-1
                    )
                elif model_key == 'batting_friendly':
                    model = GradientBoostingRegressor(
                        n_estimators=100, max_depth=10, learning_rate=0.1,
                        subsample=0.8, random_state=42
                    )
                else: # bowling_friendly
                    model = GradientBoostingRegressor(
                        n_estimators=100, max_depth=8, learning_rate=0.05,
                        subsample=0.7, random_state=42
                    )
                
                # Align columns before training/prediction if needed
                if 'fallback' in model_key and 'fallback' in features_targets:
                   common_cols = list(set(X_train.columns) & set(X_test.columns))
                   X_train = X_train[common_cols]
                   X_test = X_test[common_cols]
                   # Ensure test set has all columns from train set (add missing with 0)
                   for col in X_train.columns:
                       if col not in X_test.columns:
                           X_test[col] = 0
                   X_test = X_test[X_train.columns] # Ensure same order

                # Train the model
                model.fit(X_train, y_train)
                
                # Evaluate on training set
                train_pred = model.predict(X_train)
                train_r2 = r2_score(y_train, train_pred)
                train_mae = mean_absolute_error(y_train, train_pred)
                
                # Evaluate on test set
                test_pred = model.predict(X_test)
                test_r2 = r2_score(y_test, test_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                
                # Feature importance (if applicable)
                top_features = []
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(X_train.columns, model.feature_importances_))
                    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                elif hasattr(model, 'coef_'): # For Linear Regression
                    coef_importance = dict(zip(X_train.columns, model.coef_))
                    top_features = sorted(coef_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

                # Store the model
                self.models[model_key] = model
                
                # Store results
                results[model_key] = {
                    'model_type': model.__class__.__name__,
                    'train_r2': train_r2,
                    'train_mae': train_mae,
                    'test_r2': test_r2,
                    'test_mae': test_mae,
                    'num_features': X_train.shape[1],
                    'top_features': top_features
                }
                
                logger.info(f"{model_key} model: Test R² = {test_r2:.4f}, MAE = {test_mae:.2f}")
                
            # Save models
            self.save_models()
            
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def save_models(self):
        """Save trained models to disk"""
        try:
            for model_key, model in self.models.items():
                if model is not None:
                    model_path = os.path.join(self.models_dir, f"{model_key}_model.pkl")
                    joblib.dump(model, model_path)
                    logger.info(f"Saved {model_key} model to {model_path}")
                    
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def load_models(self):
        """Load trained models from disk"""
        try:
            loaded_any = False
            for model_key in self.models.keys(): # Includes 'fallback'
                model_path = os.path.join(self.models_dir, f"{model_key}_model.pkl")
                if os.path.exists(model_path):
                    self.models[model_key] = joblib.load(model_path)
                    logger.info(f"Loaded {model_key} model from {model_path}")
                    loaded_any = True
            return loaded_any
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def predict(self, player_data, pitch_type='balanced', use_fallback=False):
        """
        Predict fantasy points using the appropriate model or fallback
        
        Args:
            player_data (pd.DataFrame): Player data with required features
            pitch_type (str): Pitch type for prediction (balanced, batting_friendly, bowling_friendly)
            use_fallback (bool): Whether to force using the fallback model
            
        Returns:
            pd.DataFrame: Player data with predicted points
        """
        try:
            model_key = 'fallback' if use_fallback else pitch_type
            
            # Validate pitch type if not using fallback
            if not use_fallback and pitch_type not in ['balanced', 'batting_friendly', 'bowling_friendly']:
                logger.warning(f"Invalid pitch type: {pitch_type}. Using 'balanced' instead.")
                model_key = 'balanced'
                
            # Get the appropriate model
            model = self.models.get(model_key)
            
            if model is None:
                # Try to load the specific model
                model_path = os.path.join(self.models_dir, f"{model_key}_model.pkl")
                if os.path.exists(model_path):
                     self.models[model_key] = joblib.load(model_path)
                     model = self.models[model_key]
                     logger.info(f"Loaded {model_key} model for prediction.")
                else:
                    logger.warning(f"Model for {model_key} not found. Trying fallback.")
                    # Try loading fallback if the specific model failed
                    model_key = 'fallback'
                    model = self.models.get(model_key)
                    if model is None:
                         model_path_fb = os.path.join(self.models_dir, f"{model_key}_model.pkl")
                         if os.path.exists(model_path_fb):
                              self.models[model_key] = joblib.load(model_path_fb)
                              model = self.models[model_key]
                              logger.info(f"Loaded {model_key} model for prediction.")
                         else:
                              logger.error(f"Fallback model also not found. Cannot predict.")
                              # If fallback also fails, potentially return None or raise error
                              # For now, let's allow it to proceed, but log the error
                              # The feature preparation might still happen, but prediction will fail later
                              return player_data # Return original data if no model can predict

            logger.info(f"Using {model_key} model for prediction.")

            # Apply feature engineering if using a pitch-specific model
            df = player_data.copy()
            if model_key != 'fallback':
                 home_team = df['home_team'].iloc[0] if 'home_team' in df.columns else 'CSK'
                 away_team = df['away_team'].iloc[0] if 'away_team' in df.columns else 'MI'
                 venue = df['venue'].iloc[0] if 'venue' in df.columns else 'MA Chidambaram Stadium'
                 df['pitch_type'] = pitch_type # Ensure pitch type is set
                 enhanced_df = self.feature_engineer.enhance_player_features(
                     df, home_team=home_team, away_team=away_team, venue=venue
                 )
            else:
                 enhanced_df = df # Use base features for fallback

            # Prepare features based on the selected model (pitch-specific or fallback)
            if model_key == 'fallback':
                base_cols = ['credits']
                cat_cols = ['role', 'team']
                feature_cols = [col for col in base_cols + cat_cols if col in enhanced_df.columns]
            else:
                 # Use features defined during training preparation for pitch-specific
                 categorical_cols = ['role', 'pitch_type', 'team']
                 role_indicators = [col for col in enhanced_df.columns if col.startswith('is_')]
                 team_features = [col for col in enhanced_df.columns if col.startswith('vs_team_')]
                 venue_features = [col for col in enhanced_df.columns if col in [
                     'venue_advantage', 'pitch_factor', 'pitch_is_batting_friendly',
                     'pitch_is_bowling_friendly', 'pitch_is_balanced'
                 ]]
                 form_features = [col for col in enhanced_df.columns if col in [
                     'recent_form', 'last_3_avg', 'last_5_avg', 'form_trend'
                 ]]
                 interaction_features = [col for col in enhanced_df.columns if '_interaction' in col]
                 numeric_cols = ['credits']
                 feature_cols = numeric_cols + categorical_cols + role_indicators + \
                                team_features + venue_features + form_features + interaction_features
                 feature_cols = list(set(col for col in feature_cols if col in enhanced_df.columns))
                 cat_cols = [c for c in categorical_cols if c in feature_cols] # Update cat_cols for get_dummies

            X = enhanced_df[feature_cols]
            X = pd.get_dummies(X, columns=[c for c in cat_cols if c in X.columns], drop_first=True)
            
            # Ensure all model features are present
            model_features = None
            if hasattr(model, 'feature_names_in_'): # Scikit-learn >= 0.24
                model_features = model.feature_names_in_
            elif hasattr(model, 'n_features_'): # Older scikit-learn or Linear models
                 # This is less reliable, might need adjustment based on model type
                 pass # Cannot easily get names back for some models like LinearRegression
            
            if model_features is not None:
                missing_cols = set(model_features) - set(X.columns)
                for c in missing_cols:
                    X[c] = 0 # Add missing features as 0
                X = X[model_features] # Ensure order and presence
            else:
                 # Fallback or Linear Model case - need to ensure columns match if possible
                 # This part is tricky without stored feature names. 
                 # Assume the dummy creation is consistent for now.
                 logger.warning("Cannot verify feature names for the selected model. Assuming consistency.")

            # Make predictions
            predictions = model.predict(X)
            
            # Add predictions to enhanced DataFrame
            enhanced_df['predicted_points'] = predictions
            
            return enhanced_df
            
        except Exception as e:
            logger.error(f"Error predicting with model {model_key}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return original data + NaN predictions if error occurs during prediction
            player_data['predicted_points'] = np.nan
            return player_data
            
# Stand-alone execution for testing and training
if __name__ == "__main__":
    # Initialize trainer
    trainer = PitchSpecificModelTrainer()
    
    # Train models
    results = trainer.train_models()
    
    # Print results
    if results:
        print("\nPitch-Specific Model Training Results:")
        print("---------------------------------")
        for model_key, metrics in results.items():
            print(f"\n{model_key.title()} Model:")
            print(f"  - Model Type: {metrics['model_type']}")
            print(f"  - Training R² Score: {metrics['train_r2']:.4f}")
            print(f"  - Training MAE: {metrics['train_mae']:.2f}")
            print(f"  - Test R² Score: {metrics['test_r2']:.4f}")
            print(f"  - Test MAE: {metrics['test_mae']:.2f}")
            print(f"  - Features: {metrics['num_features']}")
            
            if metrics['top_features']:
                print("\n  Top 10 Important Features/Coefficients:")
                for feature, importance in metrics['top_features']:
                    print(f"    - {feature}: {importance:.4f}") 