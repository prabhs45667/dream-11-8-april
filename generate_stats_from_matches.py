import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
INPUT_FILE = os.path.join("dataset", "IPL_Matches_2008_2022.csv")
VENUE_OUTPUT_FILE = os.path.join("dataset", "venue_stats.csv")
TEAM_OUTPUT_FILE = os.path.join("dataset", "team_strengths.csv")

# --- Team Name Standardization ---
# Optional: Standardize team names if needed (example)
TEAM_NAME_MAPPING = {
    # 'Old Name': 'New Name'
    'Delhi Daredevils': 'DC', # Map old name to current code
    'Delhi Capitals': 'DC',   # Ensure current name maps to code
    'Chennai Super Kings': 'CSK', # Map full name to CSK code
    'Rising Pune Supergiant': 'RPS', # Example for other teams if needed
    'Rising Pune Supergiants': 'RPS',
    'Pune Warriors': 'PWI',
    'Gujarat Lions': 'GL',
    'Kochi Tuskers Kerala': 'KTK',
    'Deccan Chargers': 'DEC', # Example
    # Add other mappings as needed based on IPL_Matches_2008_2022.csv content
    # Use the codes consistent with SquadPlayerNames csv (e.g., CSK, DC)
}

def standardize_team_name(name):
    # Apply mapping, convert to string, uppercase, strip whitespace
    mapped_name = TEAM_NAME_MAPPING.get(str(name).strip(), str(name).strip())
    # Further map known variations to standard codes if needed
    if mapped_name == 'Chennai Super Kings': return 'CSK'
    if mapped_name == 'Kings XI Punjab': return 'PBKS' # Example
    if mapped_name == 'Punjab Kings': return 'PBKS'
    # Add more explicit final mappings if the main dict doesn't cover all cases
    return mapped_name.upper() # Return uppercase code

# --- Venue Statistics ---
def generate_venue_stats(df):
    """Calculates venue-specific statistics."""
    logging.info("Generating Venue Statistics...")
    if 'Venue' not in df.columns:
        logging.error("Input CSV missing 'Venue' column.")
        return None

    # Drop rows with missing Venue if any (shouldn't happen often)
    df_venue = df.dropna(subset=['Venue']).copy()
    if len(df_venue) != len(df):
        logging.warning(f"Dropped {len(df) - len(df_venue)} rows with missing Venue.")

    venue_stats = []

    for venue, group in df_venue.groupby('Venue'):
        stats = {'venue_name': venue}
        stats['match_count'] = len(group)

        # Filter out matches affected by 'method' (like D/L) or SuperOver for win % calc
        valid_results = group[(group['method'].isna() | (group['method'] == 'NA')) & (group['SuperOver'] == 'N')].copy()
        valid_count = len(valid_results)

        if valid_count > 0:
            # Win % Batting First
            won_bat_first = valid_results[valid_results['WonBy'] == 'Runs'].shape[0]
            stats['win_bat_first_pct'] = (won_bat_first / valid_count) * 100 if valid_count else 0

            # Win % Fielding First (Chasing)
            won_field_first = valid_results[valid_results['WonBy'] == 'Wickets'].shape[0]
            stats['win_field_first_pct'] = (won_field_first / valid_count) * 100 if valid_count else 0

            # Average Margins (only for wins)
            runs_margin = pd.to_numeric(valid_results.loc[valid_results['WonBy'] == 'Runs', 'Margin'], errors='coerce').dropna()
            stats['avg_margin_runs'] = runs_margin.mean() if not runs_margin.empty else 0

            wickets_margin = pd.to_numeric(valid_results.loc[valid_results['WonBy'] == 'Wickets', 'Margin'], errors='coerce').dropna()
            stats['avg_margin_wickets'] = wickets_margin.mean() if not wickets_margin.empty else 0
        else:
            stats['win_bat_first_pct'] = 0
            stats['win_field_first_pct'] = 0
            stats['avg_margin_runs'] = 0
            stats['avg_margin_wickets'] = 0

        # Toss Decision Stats (based on all matches at venue)
        toss_winner_count = group['TossWinner'].count() # Should be same as match_count
        if toss_winner_count > 0:
             chose_bat = group[group['TossDecision'] == 'bat'].shape[0]
             stats['toss_decision_bat_pct'] = (chose_bat / toss_winner_count) * 100
             stats['toss_decision_field_pct'] = 100 - stats['toss_decision_bat_pct']
        else:
             stats['toss_decision_bat_pct'] = 0
             stats['toss_decision_field_pct'] = 0

        venue_stats.append(stats)

    venue_df = pd.DataFrame(venue_stats)
    logging.info(f"Generated stats for {len(venue_df)} venues.")
    return venue_df

# --- Team Strength Statistics ---
def generate_team_strengths(df):
    """Calculates overall team strength statistics."""
    logging.info("Generating Team Strength Statistics...")
    if not all(col in df.columns for col in ['Team1', 'Team2', 'WinningTeam']):
        logging.error("Input CSV missing required columns: 'Team1', 'Team2', 'WinningTeam'.")
        return None

    # Standardize team names
    df['Team1_std'] = df['Team1'].apply(standardize_team_name)
    df['Team2_std'] = df['Team2'].apply(standardize_team_name)
    df['WinningTeam_std'] = df['WinningTeam'].apply(standardize_team_name)
    df['TossWinner_std'] = df['TossWinner'].apply(standardize_team_name)


    # Get unique team list
    all_teams = pd.concat([df['Team1_std'], df['Team2_std']]).unique()
    # Remove potential NaNs if WinningTeam could be NaN
    all_teams = [team for team in all_teams if pd.notna(team) and team != 'NA']

    team_stats = []

    for team in all_teams:
        stats = {'team_name': team}

        # Matches Played
        matches_as_team1 = df[df['Team1_std'] == team]
        matches_as_team2 = df[df['Team2_std'] == team]
        total_matches = len(matches_as_team1) + len(matches_as_team2)
        stats['total_matches'] = total_matches

        if total_matches == 0:
            stats['wins'] = 0
            stats['win_percentage'] = 0
            stats['toss_wins'] = 0
            stats['toss_win_percentage'] = 0
        else:
            # Wins
            wins = df[df['WinningTeam_std'] == team].shape[0]
            stats['wins'] = wins
            stats['win_percentage'] = (wins / total_matches) * 100

            # Toss Wins
            toss_wins = df[df['TossWinner_std'] == team].shape[0]
            stats['toss_wins'] = toss_wins
            stats['toss_win_percentage'] = (toss_wins / total_matches) * 100

        team_stats.append(stats)

    team_df = pd.DataFrame(team_stats)
    logging.info(f"Generated overall stats for {len(team_df)} teams.")
    return team_df


# --- Main Execution ---
if __name__ == "__main__":
    logging.info(f"Starting stats generation from: {INPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        logging.error(f"Input file not found: {INPUT_FILE}")
    else:
        try:
            match_df = pd.read_csv(INPUT_FILE)
            logging.info(f"Loaded {len(match_df)} matches.")

            # Generate and save venue stats
            venue_stats_df = generate_venue_stats(match_df.copy()) # Pass copy
            if venue_stats_df is not None:
                venue_stats_df.to_csv(VENUE_OUTPUT_FILE, index=False)
                logging.info(f"Venue statistics saved to: {VENUE_OUTPUT_FILE}")

            # Generate and save team strengths
            team_strengths_df = generate_team_strengths(match_df.copy()) # Pass copy
            if team_strengths_df is not None:
                team_strengths_df.to_csv(TEAM_OUTPUT_FILE, index=False)
                logging.info(f"Team strength statistics saved to: {TEAM_OUTPUT_FILE}")

            logging.info("Stats generation finished.")

        except FileNotFoundError:
            logging.error(f"Error: Input file not found at {INPUT_FILE}")
        except pd.errors.EmptyDataError:
            logging.error(f"Error: Input file {INPUT_FILE} is empty.")
        except KeyError as e:
             logging.error(f"Error: Input file {INPUT_FILE} is missing expected column: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}", exc_info=True) 