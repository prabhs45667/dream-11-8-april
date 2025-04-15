import pandas as pd
import numpy as np
import logging
import os
import traceback
from typing import Dict, List, Union, Optional, Tuple

# Configure logging
logger = logging.getLogger('venue_intelligence')

class VenueIntelligence:
    """
    Advanced venue intelligence for Dream11 predictions.
    Analyzes stadium characteristics and applies pitch-specific adjustments.
    """
    
    def __init__(self, data_dir="dataset"):
        """
        Initialize the VenueIntelligence module
        
        Args:
            data_dir (str): Directory containing data files
        """
        self.data_dir = data_dir
        self.venue_data = None
        self.venue_profiles = {}
        self.pitch_type_profiles = {
            'batting_friendly': {
                'avg_score': 180,
                'bat_multiplier': 1.25,
                'bowl_multiplier': 0.85,
                'boundary_percentage': 0.15,
                'spin_effectiveness': 0.7,
                'pace_effectiveness': 0.8
            },
            'bowling_friendly': {
                'avg_score': 140,
                'bat_multiplier': 0.85,
                'bowl_multiplier': 1.25,
                'boundary_percentage': 0.1,
                'spin_effectiveness': 1.2,
                'pace_effectiveness': 1.1
            },
            'balanced': {
                'avg_score': 160,
                'bat_multiplier': 1.0,
                'bowl_multiplier': 1.0,
                'boundary_percentage': 0.12,
                'spin_effectiveness': 1.0,
                'pace_effectiveness': 1.0
            }
        }
        self._load_venue_data()
        self._create_stadium_profiles()
    
    def _load_venue_data(self):
        """Load venue statistics if available"""
        try:
            # Try to load enhanced venue data first
            venue_file = os.path.join(self.data_dir, "venue_stats_enhanced.csv")
            if not os.path.exists(venue_file):
                # Fall back to regular venue stats
                venue_file = os.path.join(self.data_dir, "venue_stats.csv")
                
            if os.path.exists(venue_file):
                self.venue_data = pd.read_csv(venue_file)
                # Ensure the venue column name matches expected usage
                if 'venue_name' in self.venue_data.columns and 'venue' not in self.venue_data.columns:
                    self.venue_data = self.venue_data.rename(columns={'venue_name': 'venue'})
                elif 'venue' not in self.venue_data.columns:
                    logger.error("Venue stats file missing 'venue' or 'venue_name' column.")
                    self.venue_data = None
                    
                if self.venue_data is not None:
                    logger.info(f"Loaded venue data: {len(self.venue_data)} venues")
            else:
                logger.warning(f"No venue data file found. Creating default venue profiles.")
                self.venue_data = self._create_default_venue_data()
                
        except Exception as e:
            logger.error(f"Error loading venue data: {e}")
            traceback.print_exc()
            self.venue_data = self._create_default_venue_data()
    
    def _create_default_venue_data(self):
        """Create default venue data when no file is available"""
        # Create a basic DataFrame with common IPL venues
        default_venues = [
            "M. Chinnaswamy Stadium, Bangalore",
            "Eden Gardens, Kolkata",
            "Wankhede Stadium, Mumbai",
            "MA Chidambaram Stadium, Chennai",
            "Narendra Modi Stadium, Ahmedabad",
            "Arun Jaitley Stadium, Delhi"
        ]
        
        # Default characteristics for these venues
        venue_data = {
            'venue': default_venues,
            'avg_score': [180, 165, 175, 160, 170, 165],
            'is_batting_friendly': [True, False, True, False, True, False],
            'is_bowling_friendly': [False, True, False, True, False, True],
            'boundary_percentage': [0.15, 0.11, 0.14, 0.10, 0.13, 0.12],
            'spin_effectiveness': [0.7, 1.2, 0.8, 1.3, 0.9, 1.1],
            'pace_effectiveness': [1.1, 0.9, 1.0, 0.8, 1.0, 0.9]
        }
        
        return pd.DataFrame(venue_data)
    
    def _create_stadium_profiles(self):
        """Create detailed profiles for each stadium based on venue data"""
        if self.venue_data is None or self.venue_data.empty:
            logger.warning("No venue data available to create stadium profiles")
            return
            
        try:
            for _, row in self.venue_data.iterrows():
                venue_name = row['venue']
                profile = {}
                
                # Extract all available metrics
                for col in self.venue_data.columns:
                    if col != 'venue':
                        profile[col] = row[col]
                
                # Determine pitch type if not explicitly stated
                if 'pitch_type' not in profile:
                    if 'is_batting_friendly' in profile and profile['is_batting_friendly']:
                        profile['pitch_type'] = 'batting_friendly'
                    elif 'is_bowling_friendly' in profile and profile['is_bowling_friendly']:
                        profile['pitch_type'] = 'bowling_friendly'
                    else:
                        profile['pitch_type'] = 'balanced'
                
                # Store the complete profile
                self.venue_profiles[venue_name] = profile
                logger.debug(f"Created profile for venue: {venue_name}")
                
        except Exception as e:
            logger.error(f"Error creating stadium profiles: {e}")
            traceback.print_exc()

    def get_venue_characteristics(self, venue_name: str) -> Dict:
        """
        Get characteristics for a specific venue
        
        Args:
            venue_name (str): Name of the venue
            
        Returns:
            Dict: Dictionary of venue characteristics
        """
        # Normalize venue name for better matching
        venue_normalized = venue_name.strip().lower() if isinstance(venue_name, str) else ""
        
        # Try exact match first
        for venue, profile in self.venue_profiles.items():
            if venue.lower() == venue_normalized:
                return profile
        
        # Try partial match if exact match fails
        for venue, profile in self.venue_profiles.items():
            if venue_normalized in venue.lower() or venue.lower() in venue_normalized:
                logger.info(f"Using partial venue match: '{venue_name}' matched with '{venue}'")
                return profile
        
        # If no match found, return default balanced profile
        logger.warning(f"No profile found for venue '{venue_name}'. Using default balanced profile.")
        return self.pitch_type_profiles['balanced']

    def apply_venue_adjustments(self, player_data: pd.DataFrame, venue: str, pitch_type: str = None) -> pd.DataFrame:
        """
        Apply venue-specific adjustments to player predictions
        
        Args:
            player_data (pd.DataFrame): Player data with basic features
            venue (str): Match venue
            pitch_type (str, optional): Override pitch type
            
        Returns:
            pd.DataFrame: DataFrame with venue-adjusted predictions
        """
        if not isinstance(player_data, pd.DataFrame) or player_data.empty:
            logger.error("Invalid or empty DataFrame provided to apply_venue_adjustments")
            return player_data
            
        try:
            # Make a copy to avoid modifying the original
            df = player_data.copy()
            
            # Get venue characteristics
            venue_chars = self.get_venue_characteristics(venue)
            
            # If pitch_type is provided, override the venue's default pitch type
            if pitch_type and pitch_type in self.pitch_type_profiles:
                # Get the pitch profile but keep venue-specific metrics
                pitch_profile = self.pitch_type_profiles[pitch_type].copy()
                # Update with any venue-specific metrics
                for key, value in venue_chars.items():
                    if key not in pitch_profile:
                        pitch_profile[key] = value
                profile = pitch_profile
                
                # Update the pitch_type in the profile
                profile['pitch_type'] = pitch_type
            else:
                # Use venue's default profile
                profile = venue_chars
                
            # Add venue information to DataFrame
            df['venue_name'] = venue
            df['venue_avg_score'] = profile.get('avg_score', 160)
            df['pitch_type'] = profile.get('pitch_type', 'balanced')
            
            # Apply role-specific adjustments
            if 'role' in df.columns:
                for idx, player in df.iterrows():
                    role = player['role']
                    
                    # Initialize adjustment factor
                    bat_factor = profile.get('bat_multiplier', 1.0)
                    bowl_factor = profile.get('bowl_multiplier', 1.0)
                    
                    # Apply role-specific adjustments
                    if role == 'BAT':
                        df.loc[idx, 'venue_adjustment'] = bat_factor
                    elif role == 'BOWL':
                        df.loc[idx, 'venue_adjustment'] = bowl_factor
                    elif role == 'AR':
                        # All-rounders get a weighted average
                        df.loc[idx, 'venue_adjustment'] = (bat_factor + bowl_factor) / 2
                    elif role == 'WK':
                        # Wicket-keepers treated similar to batsmen
                        df.loc[idx, 'venue_adjustment'] = bat_factor
                    else:
                        # Default adjustment
                        df.loc[idx, 'venue_adjustment'] = 1.0
                    
                    # Add spin vs pace effectiveness if bowling style is available
                    if 'bowling_style' in df.columns:
                        bowling_style = player.get('bowling_style', '').lower()
                        if 'spin' in bowling_style or 'spinner' in bowling_style:
                            df.loc[idx, 'bowling_style_factor'] = profile.get('spin_effectiveness', 1.0)
                        elif any(pace_term in bowling_style for pace_term in ['fast', 'medium', 'seam', 'pace']):
                            df.loc[idx, 'bowling_style_factor'] = profile.get('pace_effectiveness', 1.0)
                        else:
                            df.loc[idx, 'bowling_style_factor'] = 1.0
            else:
                # If role information is not available, use a default adjustment
                df['venue_adjustment'] = 1.0
            
            # Add boundary percentage for batsmen
            df['boundary_percentage'] = profile.get('boundary_percentage', 0.12)
            
            # Add additional venue features based on match type if available
            if 'match_type' in df.columns:
                df = self._apply_match_type_adjustments(df, profile)
            
            return df
            
        except Exception as e:
            logger.error(f"Error applying venue adjustments: {e}")
            traceback.print_exc()
            # Return original data with minimal default additions
            player_data['venue_adjustment'] = 1.0
            player_data['boundary_percentage'] = 0.12
            return player_data

    def _apply_match_type_adjustments(self, df: pd.DataFrame, venue_profile: Dict) -> pd.DataFrame:
        """
        Apply match type specific adjustments (playoffs vs. league matches)
        
        Args:
            df (pd.DataFrame): Player data
            venue_profile (Dict): Venue characteristics
            
        Returns:
            pd.DataFrame: Adjusted DataFrame
        """
        try:
            # Define match type multipliers
            match_type_multipliers = {
                'League': 1.0,  # Base case
                'Qualifier': 1.15,  # 15% boost for qualifier matches
                'Eliminator': 1.2,  # 20% boost for eliminator matches
                'Final': 1.25  # 25% boost for final match
            }
            
            # Apply match type adjustments
            for idx, player in df.iterrows():
                match_type = player['match_type']
                
                # Apply multiplier based on match type if valid
                if match_type in match_type_multipliers:
                    multiplier = match_type_multipliers[match_type]
                    current_adjustment = df.loc[idx, 'venue_adjustment']
                    df.loc[idx, 'venue_adjustment'] = current_adjustment * multiplier
                    
                    # Add match type importance feature
                    df.loc[idx, 'match_importance'] = multiplier
            
            return df
            
        except Exception as e:
            logger.error(f"Error applying match type adjustments: {e}")
            traceback.print_exc()
            return df

    def get_pitch_report(self, venue: str, match_type: str = None) -> Dict:
        """
        Generate a comprehensive pitch report for UI display
        
        Args:
            venue (str): Match venue
            match_type (str, optional): Match type for additional context
            
        Returns:
            Dict: Detailed pitch report
        """
        try:
            # Get venue characteristics
            venue_chars = self.get_venue_characteristics(venue)
            
            # Add match type context if provided
            if match_type in ['Qualifier', 'Eliminator', 'Final']:
                importance = "high"
                pressure = "high"
            else:
                importance = "normal"
                pressure = "normal"
            
            # Determine expected first innings score
            avg_score = venue_chars.get('avg_score', 160)
            if match_type in ['Qualifier', 'Eliminator', 'Final']:
                # Playoff matches often see slightly lower scores due to pressure
                avg_score = avg_score * 0.95
            
            # Determine pitch type description
            pitch_type = venue_chars.get('pitch_type', 'balanced')
            if pitch_type == 'batting_friendly':
                description = "Batting-friendly pitch with good bounce and carry. Expect high scores."
                strategy = "Select more batsmen and aggressive all-rounders."
            elif pitch_type == 'bowling_friendly':
                description = "Bowling-friendly conditions with assistance for bowlers. Expect lower scores."
                strategy = "Select more bowlers and defensive all-rounders."
            else:
                description = "Balanced pitch offering something for both batters and bowlers."
                strategy = "Balanced team selection recommended."
            
            # Check if spin or pace is more effective
            spin_effectiveness = venue_chars.get('spin_effectiveness', 1.0)
            pace_effectiveness = venue_chars.get('pace_effectiveness', 1.0)
            
            if spin_effectiveness > pace_effectiveness:
                bowling_tip = "Spinners likely to be more effective on this pitch."
            elif pace_effectiveness > spin_effectiveness:
                bowling_tip = "Fast bowlers likely to be more effective on this pitch."
            else:
                bowling_tip = "Both spin and pace should be equally effective."
            
            # Compile report
            report = {
                'venue': venue,
                'pitch_type': pitch_type,
                'description': description,
                'strategy': strategy,
                'expected_score': round(avg_score),
                'match_importance': importance,
                'pressure': pressure,
                'bowling_tip': bowling_tip,
                'boundary_percentage': venue_chars.get('boundary_percentage', 0.12),
                'historical_data': {
                    'avg_score': venue_chars.get('avg_score', 160),
                    'spin_effectiveness': spin_effectiveness,
                    'pace_effectiveness': pace_effectiveness
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating pitch report: {e}")
            traceback.print_exc()
            # Return a basic default report
            return {
                'venue': venue,
                'pitch_type': 'balanced',
                'description': "Standard pitch conditions expected.",
                'strategy': "Balanced team selection recommended.",
                'expected_score': 160,
                'match_importance': 'normal',
                'pressure': 'normal',
                'bowling_tip': "Both spin and pace should be effective."
            } 