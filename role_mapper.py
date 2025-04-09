import pandas as pd
import traceback

class RoleMapper:
    def __init__(self):
        self.role_mapping = {
            'WK': ['WK', 'WICKET KEEPER', 'WICKET-KEEPER', 'KEEPER', 'WICKETKEEPER', 'W/K', 'W/K', 'WICKET_KEEPER', 'KEEPER/BATSMAN'],
            'BAT': ['BAT', 'BATSMAN', 'BATSMEN', 'BATTER', 'BATTERS', 'BATTING', 'BATSMAN/KEEPER', 'BATS', 'BATSMAN/FIELDER'],
            'AR': ['AR', 'ALL', 'ALL ROUNDER', 'ALL-ROUNDER', 'ALLROUNDER', 'ALL ROUNDERS', 'ALL-ROUNDERS', 'ALL_ROUNDER', 'ALROUNDER', 'A/R'],
            'BOWL': ['BOWL', 'BOWLER', 'BOWLERS', 'BOWLING', 'FAST BOWLER', 'SPINNER', 'PACE BOWLER', 'SPIN BOWLER', 'FAST_BOWLER', 'SLOW_BOWLER']
        }
        
        # Required roles for team creation
        self.required_roles = {
            'WK': (1, 1),   # Exactly 1 wicket-keeper
            'BAT': (3, 5),  # 3-5 batsmen
            'AR': (1, 3),   # 1-3 all-rounders
            'BOWL': (3, 5)  # 3-5 bowlers
        }
        
        # Team code mapping (expanded)
        self.team_code_mapping = {
            'CSK': ['CSK', 'CHE', 'CHENNAI', 'CHENNAI SUPER KINGS', 'CHENNAI SUPER', 'CHENNAI SK'],
            'MI': ['MI', 'MUM', 'MUMBAI', 'MUMBAI INDIANS', 'MUMBAI I', 'MUM INDIANS'],
            'RCB': ['RCB', 'BAN', 'BANGALORE', 'BENGALURU', 'ROYAL CHALLENGERS', 'ROYAL CHALLENGERS BANGALORE', 'BANGALORE RC'],
            'KKR': ['KKR', 'KOL', 'KOLKATA', 'KOLKATA KNIGHT', 'KOLKATA KNIGHT RIDERS', 'KNIGHT RIDERS'],
            'PBKS': ['PBKS', 'PUN', 'KXI', 'PUNJAB', 'PUNJAB KINGS', 'KINGS XI', 'KINGS XI PUNJAB', 'KXIP'],
            'DC': ['DC', 'DEL', 'DELHI', 'DELHI CAPITALS', 'DELHI DAREDEVILS', 'DAREDEVILS'],
            'RR': ['RR', 'RAJ', 'RAJASTHAN', 'RAJASTHAN ROYALS', 'ROYALS'],
            'SRH': ['SRH', 'HYD', 'HYDERABAD', 'SUNRISERS', 'SUNRISERS HYDERABAD', 'DECCAN CHARGERS'],
            'GT': ['GT', 'GUJ', 'GUJARAT', 'GUJARAT TITANS', 'TITANS'],
            'LSG': ['LSG', 'LUC', 'LUCKNOW', 'LUCKNOW SUPER', 'LUCKNOW SUPER GIANTS', 'SUPER GIANTS']
        }
    
    def standardize_role(self, role):
        """Standardize player role to one of: WK, BAT, AR, BOWL"""
        try:
            if pd.isna(role) or role is None or str(role).strip() == '':
                print(f"Warning: Missing role value, defaulting to BAT")
                return 'BAT'  # Default to BAT if role is missing
                
            # Convert to string, uppercase and strip whitespace
            role_str = str(role).upper().strip()
            
            # Direct match check
            for std_role, variants in self.role_mapping.items():
                if role_str in variants:
                    return std_role
                    
            # Partial match check
            for std_role, variants in self.role_mapping.items():
                for variant in variants:
                    if variant in role_str or role_str in variant:
                        print(f"Partial role match: '{role_str}' -> '{std_role}' (matched with '{variant}')")
                        return std_role
                    
            # Keyword check
            if any(keyword in role_str for keyword in ['KEEPER', 'WK', 'WICKET']):
                return 'WK'
            elif any(keyword in role_str for keyword in ['BAT', 'BATSMAN', 'BATTER']):
                return 'BAT'
            elif any(keyword in role_str for keyword in ['ALL', 'ROUNDER', 'AR', 'A/R', 'A-R']):
                return 'AR'
            elif any(keyword in role_str for keyword in ['BOWL', 'SPINNER', 'PACE']):
                return 'BOWL'
                
            # Final fallback
            print(f"Warning: Unrecognized role '{role_str}', defaulting to BAT")
            return 'BAT'  # Default to BAT if role is unknown
            
        except Exception as e:
            print(f"Error standardizing role '{role}': {str(e)}")
            traceback.print_exc()
            return 'BAT'  # Default to BAT on error
            
    def standardize_team_code(self, team_code):
        """Standardize team code to the primary code (e.g., CHE → CSK)"""
        try:
            if pd.isna(team_code) or team_code is None or str(team_code).strip() == '':
                print(f"Warning: Missing team code, returning None")
                return None  # Return None for missing team codes
                
            # Convert to string, uppercase and strip whitespace
            team_str = str(team_code).upper().strip()
            
            # Direct match check
            for std_code, variants in self.team_code_mapping.items():
                if team_str in variants:
                    return std_code
            
            # Partial match check
            for std_code, variants in self.team_code_mapping.items():
                for variant in variants:
                    if (len(variant) >= 3 and variant in team_str) or (len(team_str) >= 3 and team_str in variant):
                        print(f"Partial team match: '{team_str}' -> '{std_code}' (matched with '{variant}')")
                        return std_code
                    
            print(f"Warning: Unrecognized team code '{team_str}', keeping original")
            return team_str  # Return original code if no mapping found
            
        except Exception as e:
            print(f"Error standardizing team code '{team_code}': {str(e)}")
            traceback.print_exc()
            return str(team_code) if team_code is not None else None  # Return original code on error
    
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
        try:
            if not available_roles:
                print("Warning: No available roles provided, using default requirements")
                return self.required_roles.copy()
                
            requirements = self.required_roles.copy()
            print(f"Original role requirements: {requirements}")
            print(f"Available roles: {available_roles}")
            
            # Check if we have at least one player from each role
            missing_roles = [role for role in requirements.keys() if role not in available_roles or available_roles[role] == 0]
            if missing_roles:
                print(f"Warning: Missing players for roles: {missing_roles}")
            
            # Check each required role
            for role, (min_req, max_req) in requirements.items():
                # If role is missing or has fewer than minimum required players
                if role not in available_roles or available_roles.get(role, 0) < min_req:
                    available_count = available_roles.get(role, 0)
                    
                    # If we have some players for this role but fewer than min_req
                    if role in available_roles and available_count > 0:
                        # Adjust minimum to what's available
                        requirements[role] = (available_count, max_req)
                        print(f"Adjusted {role} requirement from ({min_req}, {max_req}) to ({available_count}, {max_req}) based on availability")
                    else:
                        # Set minimum to 0 to make it optional
                        requirements[role] = (0, max_req)
                        print(f"Made {role} optional by setting min requirement to 0 (was {min_req})")
                        
                        # Increase other role minimums to compensate if possible
                        if role != 'WK':  # WK is special, always need exactly 1 if available
                            for other_role in ['BAT', 'BOWL']:
                                if other_role in available_roles and available_roles[other_role] > requirements[other_role][0]:
                                    curr_min, curr_max = requirements[other_role]
                                    requirements[other_role] = (curr_min + 1, curr_max)
                                    print(f"Increased {other_role} minimum requirement to {curr_min + 1} to compensate for missing {role}")
            
            # Special handling for WK: if no WK available, must use BAT
            if 'WK' not in available_roles or available_roles.get('WK', 0) == 0:
                print("No wicket-keepers available, adjusting requirements")
                if 'BAT' in available_roles and available_roles['BAT'] > requirements['BAT'][0]:
                    curr_min, curr_max = requirements['BAT']
                    requirements['BAT'] = (curr_min + 1, curr_max)
                    print(f"Increased BAT minimum to {curr_min + 1} to compensate for missing WK")
            
            print(f"Final adjusted role requirements: {requirements}")
            return requirements
            
        except Exception as e:
            print(f"Error adjusting role requirements: {str(e)}")
            traceback.print_exc()
            return self.required_roles.copy()  # Return default requirements on error 