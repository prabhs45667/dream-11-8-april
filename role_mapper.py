import pandas as pd

class RoleMapper:
    def __init__(self):
        self.role_mapping = {
            'WK': ['WK', 'WICKET KEEPER', 'WICKET-KEEPER', 'KEEPER', 'WICKETKEEPER', 'W/K', 'W/K'],
            'BAT': ['BAT', 'BATSMAN', 'BATSMEN', 'BATTER', 'BATTERS'],
            'AR': ['AR', 'ALL', 'ALL ROUNDER', 'ALL-ROUNDER', 'ALLROUNDER', 'ALL ROUNDERS', 'ALL-ROUNDERS'],
            'BOWL': ['BOWL', 'BOWLER', 'BOWLERS']
        }
        
        # Required roles for team creation
        self.required_roles = {
            'WK': (1, 1),   # Exactly 1 wicket-keeper
            'BAT': (3, 5),  # 3-5 batsmen
            'AR': (1, 3),   # 1-3 all-rounders
            'BOWL': (3, 5)  # 3-5 bowlers
        }
        
        # Team code mapping
        self.team_code_mapping = {
            'CSK': ['CSK', 'CHE'],  # CSK is also known as CHE in some datasets
            'MI': ['MI', 'MUM'],    # MI is also known as MUM in some datasets
            'RCB': ['RCB', 'BAN'],  # RCB is also known as BAN in some datasets
            'KKR': ['KKR', 'KOL'],  # KKR is also known as KOL in some datasets
            'PBKS': ['PBKS', 'PUN', 'KXI'], # PBKS is also known as PUN in some datasets
            'DC': ['DC', 'DEL'],    # DC is also known as DEL in some datasets
            'RR': ['RR', 'RAJ'],    # RR is also known as RAJ in some datasets
            'SRH': ['SRH', 'HYD'],  # SRH is also known as HYD in some datasets
            'GT': ['GT', 'GUJ'],    # GT is also known as GUJ in some datasets
            'LSG': ['LSG', 'LUC']   # LSG is also known as LUC in some datasets
        }
    
    def standardize_role(self, role):
        """Standardize player role to one of: WK, BAT, AR, BOWL"""
        try:
            if pd.isna(role):
                return 'BAT'  # Default to BAT if role is missing
                
            role = str(role).upper().strip()
            
            # Check each role category
            for std_role, variants in self.role_mapping.items():
                if role in variants:
                    return std_role
                    
            # If role contains any of the keywords, map accordingly
            if any(keyword in role for keyword in ['KEEPER', 'WK']):
                return 'WK'
            elif any(keyword in role for keyword in ['BAT', 'BATSMAN']):
                return 'BAT'
            elif any(keyword in role for keyword in ['ALL', 'ROUNDER']):
                return 'AR'
            elif any(keyword in role for keyword in ['BOWL']):
                return 'BOWL'
                
            return 'BAT'  # Default to BAT if role is unknown
            
        except Exception as e:
            print(f"Error standardizing role '{role}': {str(e)}")
            return 'BAT'  # Default to BAT on error
            
    def standardize_team_code(self, team_code):
        """Standardize team code to the primary code (e.g., CHE → CSK)"""
        try:
            if pd.isna(team_code):
                return None  # Return None for missing team codes
                
            team_code = str(team_code).upper().strip()
            
            # Check each team code mapping
            for std_code, variants in self.team_code_mapping.items():
                if team_code in variants:
                    return std_code
                    
            return team_code  # Return original code if no mapping found
            
        except Exception as e:
            print(f"Error standardizing team code '{team_code}': {str(e)}")
            return team_code  # Return original code on error
    
    def get_adjusted_role_requirements(self, available_roles):
        """
        Adjust role requirements based on available roles
        
        Parameters:
        -----------
        available_roles : dict
            Dictionary with role counts from available players
            
        Returns:
        --------
        dict
            Adjusted role requirements dictionary
        """
        requirements = self.required_roles.copy()
        
        # Check each required role
        for role, (min_req, max_req) in requirements.items():
            # If role is missing or has fewer than minimum required players
            if role not in available_roles or available_roles[role] < min_req:
                # Set minimum to 0 to make it optional
                requirements[role] = (0, max_req)
                
                # Increase other role minimums to compensate if possible
                if role != 'WK':  # WK is special, always need exactly 1
                    for other_role in ['BAT', 'BOWL']:
                        if other_role in available_roles and available_roles[other_role] > min_req:
                            curr_min, curr_max = requirements[other_role]
                            requirements[other_role] = (curr_min + 1, curr_max)
        
        return requirements 