import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression  # Added for fallback
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from feature_engineering import FeatureEngineer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import time

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
        
        # Initialize imputers dictionary
        self.imputers = {
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
                    imputer = None # No imputation needed for Linear Regression
                elif model_key == 'balanced':
                    model = RandomForestRegressor(
                        n_estimators=200, # Increased from 100
                        max_depth=20,     # Increased from 15
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,        # Use all available cores
                        bootstrap=True,   # Enable bootstrapping for better robustness
                        max_features='sqrt' # Use sqrt of features for better generalization
                    )
                    imputer = SimpleImputer(strategy='mean')
                elif model_key == 'batting_friendly':
                    model = GradientBoostingRegressor(
                        n_estimators=250, # Increased from 100
                        max_depth=12,     # Increased from 10
                        learning_rate=0.08, # Lowered for better convergence with more trees
                        subsample=0.85,   # Increased from 0.8
                        random_state=42,
                        verbose=0,
                        validation_fraction=0.1, # Use 10% for validation during training
                        n_iter_no_change=15,     # Early stopping after 15 iterations with no improvement
                        warm_start=True          # Use previous solution to fit additional estimators
                    )
                    imputer = SimpleImputer(strategy='mean')
                else: # bowling_friendly
                    model = GradientBoostingRegressor(
                        n_estimators=250, # Increased from 100
                        max_depth=10,     # Increased from 8
                        learning_rate=0.05,
                        subsample=0.8,    # Increased from 0.7
                        random_state=42,
                        verbose=0,
                        validation_fraction=0.1, # Use 10% for validation during training
                        n_iter_no_change=15,     # Early stopping after 15 iterations with no improvement
                        warm_start=True          # Use previous solution to fit additional estimators
                    )
                    imputer = SimpleImputer(strategy='mean')
                
                # Align columns before imputation/training if needed
                if 'fallback' not in model_key: # Align for pitch-specific models
                    # Get common columns between train and test
                    common_cols = list(set(X_train.columns) & set(X_test.columns))
                    X_train = X_train[common_cols]
                    X_test = X_test[common_cols]
                    
                    # Ensure test set has all columns from train set (add missing with 0)
                    for col in X_train.columns:
                        if col not in X_test.columns:
                            X_test[col] = 0
                    X_test = X_test[X_train.columns] # Ensure same order

                # Store column names before imputation (imputer converts to numpy array)
                X_train_columns = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None
                
                # Impute missing values if needed
                if imputer:
                    logger.info(f"Applying SimpleImputer (strategy='{imputer.strategy}') to {model_key} data")
                    X_train = imputer.fit_transform(X_train)
                    X_test = imputer.transform(X_test)
                    # Save the imputer along with the model
                    self.imputers[model_key] = imputer
                    
                # Evaluate with cross-validation before final training
                if model_key != 'fallback' and hasattr(model, 'fit'):
                    logger.info(f"Performing 5-fold cross-validation for {model_key} model")
                    start_time = time.time()
                    cv_scores = cross_val_score(model, X_train, y_train, 
                                               cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                               scoring='neg_mean_absolute_error', 
                                               n_jobs=-1)
                    cv_time = time.time() - start_time
                    logger.info(f"Cross-validation MAE: {-np.mean(cv_scores):.2f} ±{np.std(cv_scores):.2f}, completed in {cv_time:.2f}s")
                    
                # Train the model
                logger.info(f"Training final {model_key} model...")
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                logger.info(f"Model training completed in {training_time:.2f}s")
                
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
                if X_train_columns: # Ensure we have column names
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(X_train_columns, model.feature_importances_))
                        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    elif hasattr(model, 'coef_'): # For Linear Regression
                        coef_importance = dict(zip(X_train_columns, model.coef_))
                        top_features = sorted(coef_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                else:
                     logger.warning(f"Could not determine feature names for {model_key} model to calculate importance.")

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
                
            # Save models and imputers
            self.save_models()
            
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def save_models(self):
        """Save trained models and imputers to disk"""
        try:
            # Save models
            for model_key, model in self.models.items():
                if model is not None:
                    model_path = os.path.join(self.models_dir, f"{model_key}_model.pkl")
                    joblib.dump(model, model_path)
                    logger.info(f"Saved {model_key} model to {model_path}")
                    
            # Save imputers
            for model_key, imputer in self.imputers.items():
                if imputer is not None:
                    imputer_path = os.path.join(self.models_dir, f"{model_key}_imputer.pkl")
                    joblib.dump(imputer, imputer_path)
                    logger.info(f"Saved {model_key} imputer to {imputer_path}")
                    
        except Exception as e:
            logger.error(f"Error saving models/imputers: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def load_models(self):
        """Load trained models and imputers from disk"""
        try:
            loaded_any = False
            # Load models
            for model_key in self.models.keys(): # Includes 'fallback'
                model_path = os.path.join(self.models_dir, f"{model_key}_model.pkl")
                if os.path.exists(model_path):
                    self.models[model_key] = joblib.load(model_path)
                    logger.info(f"Loaded {model_key} model from {model_path}")
                    loaded_any = True
            
            # Load imputers
            for model_key in self.models.keys(): # Only load for models that need it
                 if model_key != 'fallback':
                    imputer_path = os.path.join(self.models_dir, f"{model_key}_imputer.pkl")
                    if os.path.exists(imputer_path):
                        self.imputers[model_key] = joblib.load(imputer_path)
                        logger.info(f"Loaded {model_key} imputer from {imputer_path}")
                    else:
                         # If imputer is missing, it might cause issues during prediction
                         logger.warning(f"Imputer for {model_key} not found at {imputer_path}. Prediction might fail if NaNs are present.")
                         self.imputers[model_key] = None # Explicitly set to None
            return loaded_any
            
        except Exception as e:
            logger.error(f"Error loading models/imputers: {str(e)}")
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
                
            # Get the appropriate model and imputer
            model = self.models.get(model_key)
            imputer = self.imputers.get(model_key)
            
            if model is None:
                logger.warning(f"Model for {model_key} not loaded. Trying to load.")
                # Try to load the specific model and imputer
                model_path = os.path.join(self.models_dir, f"{model_key}_model.pkl")
                imputer_path = os.path.join(self.models_dir, f"{model_key}_imputer.pkl")
                
                if os.path.exists(model_path):
                     self.models[model_key] = joblib.load(model_path)
                     model = self.models[model_key]
                     logger.info(f"Loaded {model_key} model for prediction.")
                     if model_key != 'fallback' and os.path.exists(imputer_path):
                         self.imputers[model_key] = joblib.load(imputer_path)
                         imputer = self.imputers[model_key]
                         logger.info(f"Loaded {model_key} imputer for prediction.")
                     elif model_key != 'fallback':
                          logger.warning(f"Imputer for {model_key} not found. Prediction might fail.")
                else:
                    logger.warning(f"Model for {model_key} not found. Trying fallback.")
                    # Try loading fallback if the specific model failed
                    model_key = 'fallback'
                    model = self.models.get(model_key)
                    imputer = None # Fallback doesn't use imputer
                    if model is None:
                         model_path_fb = os.path.join(self.models_dir, f"{model_key}_model.pkl")
                         if os.path.exists(model_path_fb):
                              self.models[model_key] = joblib.load(model_path_fb)
                              model = self.models[model_key]
                              logger.info(f"Loaded {model_key} model for prediction.")
                         else:
                              logger.error(f"Fallback model also not found. Cannot predict.")
                              return player_data # Return original data if no model can predict

            logger.info(f"Using {model_key} model for prediction.")

            # Apply feature engineering if needed (usually needed for all)
            df = player_data.copy()
            home_team = df['home_team'].iloc[0] if 'home_team' in df.columns else 'CSK'
            away_team = df['away_team'].iloc[0] if 'away_team' in df.columns else 'MI'
            venue = df['venue'].iloc[0] if 'venue' in df.columns else 'MA Chidambaram Stadium'
            df['pitch_type'] = pitch_type # Ensure pitch type is set for feature eng.
            enhanced_df = self.feature_engineer.enhance_player_features(
                df, home_team=home_team, away_team=away_team, venue=venue
            )
            
            if enhanced_df is None:
                 logger.error("Feature enhancement failed. Cannot proceed with prediction.")
                 return player_data # Return original data

            # Prepare features based on the selected model
            # Get the trained model's expected features (columns)
            # Need to handle case where model is loaded but features aren't stored
            # A more robust way is to save/load feature lists with models
            try:
                 # Assuming model was trained on a pandas DataFrame and retains feature names
                 if hasattr(model, 'feature_names_in_'):
                      model_features = list(model.feature_names_in_)
                 elif hasattr(model, 'feature_name_'): # Some models like XGBoost
                     model_features = list(model.feature_name_)
                 else:
                     # Fallback: attempt to load features from a saved file or use defaults
                     # For now, we'll assume a common feature set or risk error
                     # Let's try to infer from a loaded model if possible, otherwise use default set
                     logger.warning("Could not reliably determine model features. Using potentially incomplete set.")
                     # Default fallback - less robust
                     base_cols = ['credits']
                     cat_cols = ['role', 'team']
                     num_cols = ['recent_form', 'venue_avg', 'opposition_avg'] # Example numeric
                     model_features = [col for col in base_cols + cat_cols + num_cols if col in enhanced_df.columns]
                     if not model_features:
                         logger.error("No usable features found for prediction.")
                         return player_data
                     
            except Exception as e:
                 logger.error(f"Error determining model features: {e}. Using defaults.")
                 base_cols = ['credits']
                 cat_cols = ['role', 'team']
                 model_features = [col for col in base_cols + cat_cols if col in enhanced_df.columns]
                 if not model_features:
                      logger.error("No usable default features found for prediction.")
                      return player_data
                      
            # Ensure player data has all the features the model expects
            missing_cols = [col for col in model_features if col not in enhanced_df.columns]
            if missing_cols:
                logger.warning(f"Prediction data missing columns: {missing_cols}. Adding with 0.")
                for col in missing_cols:
                    enhanced_df[col] = 0
            
            # Select only the features the model was trained on, in the correct order
            prediction_features = enhanced_df[model_features]

            # Apply imputation if required for this model
            if imputer:
                try:
                    prediction_features_imputed = imputer.transform(prediction_features)
                    # Imputer returns numpy array, convert back to DataFrame with correct columns
                    prediction_features = pd.DataFrame(prediction_features_imputed, index=prediction_features.index, columns=model_features)
                except Exception as e:
                     logger.error(f"Error applying imputer during prediction: {e}. Prediction might be inaccurate.")
                     # Continue without imputation? Or return? For now, continue.

            # Make predictions
            predictions = model.predict(prediction_features)
            
            # Add predictions to the original DataFrame
            player_data['predicted_points'] = predictions
            return player_data
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return original data with maybe a flag or NaN for points?
            player_data['predicted_points'] = np.nan # Indicate prediction failure
            return player_data
            
# Stand-alone execution for testing and training
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train pitch-specific models for fantasy cricket')
    parser.add_argument('--iterations', type=int, default=1, 
                        help='Number of training iterations with different random seeds (default: 1)')
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Directory containing training data')
    parser.add_argument('--models_dir', type=str, default='models',
                        help='Directory to save trained models')
    args = parser.parse_args()
    
    # Train models with multiple random seeds if requested
    for iteration in range(args.iterations):
        print(f"\n=== Training Iteration {iteration + 1}/{args.iterations} ===")
        random_seed = 42 + iteration  # Use different seed for each iteration
        
        # Initialize trainer
        trainer = PitchSpecificModelTrainer(data_dir=args.data_dir, models_dir=args.models_dir)
        
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
                        
            print("\n=======================================")
            print(f"Training iteration {iteration + 1} completed successfully.") 