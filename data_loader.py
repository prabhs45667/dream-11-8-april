import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Utility class for loading various data files needed by the Dream11App
    """
    
    def __init__(self, data_dir="dataset"):
        """
        Initialize the data loader
        
        Args:
            data_dir (str): Directory containing data files
        """
        self.data_dir = data_dir
        
    def load_deliveries_data(self):
        """
        Load ball-by-ball deliveries data
        
        Returns:
            pd.DataFrame: Deliveries data or None if not available
        """
        try:
            # Look for deliveries file
            possible_files = [
                os.path.join(self.data_dir, "ipl_2025_deliveries.csv"),
                os.path.join(self.data_dir, "ipl_deliveries.csv"),
                os.path.join(self.data_dir, "deliveries.csv")
            ]
            
            for file_path in possible_files:
                if os.path.exists(file_path):
                    logger.info(f"Loading deliveries data from {file_path}")
                    deliveries_df = pd.read_csv(file_path)
                    logger.info(f"Loaded {len(deliveries_df)} delivery records")
                    return deliveries_df
                    
            logger.warning("No deliveries data file found. Searched for: " + str(possible_files))
            return None
            
        except Exception as e:
            logger.error(f"Error loading deliveries data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def load_squad_data(self, team=None):
        """
        Load squad data, optionally filtering for a specific team
        
        Args:
            team (str, optional): Team code to filter for
            
        Returns:
            pd.DataFrame: Squad data
        """
        try:
            # Look for squad data file
            possible_files = [
                os.path.join(self.data_dir, "SquadPlayerNames_IndianT20League - SquadData_AllTeams.csv"),
                os.path.join(self.data_dir, "SquadData_AllTeams.csv"),
                os.path.join(self.data_dir, "squads.csv")
            ]
            
            for file_path in possible_files:
                if os.path.exists(file_path):
                    logger.info(f"Loading squad data from {file_path}")
                    squad_df = pd.read_csv(file_path)
                    
                    # Filter for specific team if requested
                    if team and 'Team' in squad_df.columns:
                        squad_df = squad_df[squad_df['Team'] == team]
                        
                    logger.info(f"Loaded {len(squad_df)} squad records")
                    return squad_df
                    
            logger.warning("No squad data file found. Searched for: " + str(possible_files))
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error loading squad data: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame() 