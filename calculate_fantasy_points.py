import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import traceback # Ensure traceback is imported

# --- Configuration ---
DELIVERY_FILES = [
    'dataset/ipl_2022_deliveries.csv',
    'dataset/ipl_2023_deliveries.csv',
    'dataset/ipl_2024_deliveries.csv',
    'dataset/ipl_2025_deliveries.csv' # Assuming this exists and has historical data
]
# MATCH_INFO_FILE = 'dataset/IPL_Matches_2008_2022.csv' # No longer needed for dates
FIELDING_DATA_FILE = 'dataset/Fielding_data.csv'
OUTPUT_FILE = 'dataset/player_history.csv'
DATASET_DIR = 'dataset'

# --- Fantasy Point Rules ---
POINTS = {
    'run': 1,
    'boundary_bonus': 1, # For 4s
    'six_bonus': 2,      # For 6s
    'half_century_bonus': 8,
    'century_bonus': 16,
    'duck_penalty': -2, # Applied to Batsmen, WK, All-Rounder (requires role info later)
    'wicket': 25,      # Excludes run-out
    'lbw_bowled_bonus': 8,
    '3_wicket_bonus': 4,
    '4_wicket_bonus': 8,
    '5_wicket_bonus': 16,
    'maiden_bonus': 12,
    # Fielding points below are reference for understanding,
    # but we will use the pre-calculated Fielding_FP from the CSV.
    # 'catch': 8,
    # '3_catch_bonus': 4,
    # 'stumping': 12,
    # 'run_out_direct': 12,
    # 'run_out_involved': 6
}

ECONOMY_RATE_POINTS = {
    (0, 4.99): 6,
    (5, 5.99): 4,
    (6, 7.00): 2,
    (10, 11.00): -2,
    (11.01, 12.00): -4,
    (12.01, float('inf')): -6
}

STRIKE_RATE_POINTS = {
    (60, 70): -2,
    (50, 59.99): -4,
    (0, 49.99): -6 # Assuming strike rate >= 0
}

MIN_OVERS_FOR_ECONOMY = 2
MIN_BALLS_FOR_STRIKE_RATE = 10

# --- Helper Functions ---

def clean_player_name(series):
    """ Standard function to clean player names. """
    return series.str.replace(r' \(sub\)', '', regex=True).str.strip()

def calculate_overs(balls):
    """ Calculates total overs from number of balls bowled. """
    return balls // 6 + (balls % 6) / 10.0

def get_economy_points(runs_conceded, balls_bowled):
    """ Calculates economy rate points. """
    if balls_bowled == 0:
        return 0
    overs = calculate_overs(balls_bowled)
    if overs < MIN_OVERS_FOR_ECONOMY:
        return 0
    economy_rate = runs_conceded / (balls_bowled / 6) # Use balls/6 for accurate rate calculation

    for (lower, upper), points in ECONOMY_RATE_POINTS.items():
        if lower <= economy_rate <= upper:
            return points
    return 0 # Default if rate falls outside defined ranges

def get_strike_rate_points(runs_scored, balls_faced):
    """ Calculates strike rate points. """
    if balls_faced < MIN_BALLS_FOR_STRIKE_RATE:
        return 0
    if balls_faced == 0: # Avoid division by zero
        return 0
    strike_rate = (runs_scored / balls_faced) * 100

    for (lower, upper), points in STRIKE_RATE_POINTS.items():
        if lower <= strike_rate <= upper:
            return points
    return 0 # Default if rate falls outside defined ranges


def calculate_batting_points(group):
    """ Calculates fantasy points for a batter in a match. """
    runs = group['runs_of_bat'].sum()
    balls_faced = len(group) # Assuming each row is a ball faced
    fours = (group['runs_of_bat'] == 4).sum()
    sixes = (group['runs_of_bat'] == 6).sum()

    points = 0
    points += runs * POINTS['run']
    points += fours * POINTS['boundary_bonus']
    points += sixes * POINTS['six_bonus']

    # Milestone bonuses
    if runs >= 100:
        points += POINTS['century_bonus']
    elif runs >= 50:
        points += POINTS['half_century_bonus']

    # Duck penalty - Cannot apply accurately without player role info
    # if runs == 0 and balls_faced > 0:
    #     # Need player role (BAT, WK, ALL) to apply penalty correctly
    #     # points += POINTS['duck_penalty'] # Placeholder
    #     pass

    # Strike rate penalty
    points += get_strike_rate_points(runs, balls_faced)

    return pd.Series({
        'batting_points': points,
        'runs_scored': runs,
        'balls_faced': balls_faced
    })

def calculate_bowling_points(group):
    """ Calculates fantasy points for a bowler in a match. """
    wickets = group[group['wicket_type'].notna() & (group['wicket_type'] != 'run out') & (group['wicket_type'] != 'retired hurt') & (group['wicket_type'] != 'obstructing the field')].shape[0]
    runs_conceded = group['runs_of_bat'].sum() + group[group['wide'].notna() | group['noballs'].notna()]['extras'].sum() # Conceded runs = bat runs + wides + noballs
    balls_bowled = len(group[group['wide'].isna() & group['noballs'].isna()]) # Count only legal deliveries
    overs = calculate_overs(balls_bowled)

    # Calculate maidens
    overs_bowled = group.groupby('over')
    maidens = 0
    for name, over_group in overs_bowled:
      # A maiden over concedes 0 runs from the bat, legbyes, or byes (wides/noballs don't count against maiden)
      runs_in_over = over_group['runs_of_bat'].sum() + over_group['legbyes'].sum() + over_group['byes'].sum()
      if runs_in_over == 0 and len(over_group[over_group['wide'].isna() & over_group['noballs'].isna()]) == 6: # Ensure full over was bowled
          maidens += 1


    points = 0
    points += wickets * POINTS['wicket']
    points += maidens * POINTS['maiden_bonus']

    # LBW/Bowled bonus
    lbw_bowled_wickets = group[group['wicket_type'].isin(['bowled', 'lbw'])].shape[0]
    points += lbw_bowled_wickets * POINTS['lbw_bowled_bonus']

    # Wicket haul bonuses
    if wickets >= 5:
        points += POINTS['5_wicket_bonus']
    elif wickets == 4:
        points += POINTS['4_wicket_bonus']
    elif wickets == 3:
        points += POINTS['3_wicket_bonus']

    # Economy rate points
    points += get_economy_points(runs_conceded, balls_bowled)

    return pd.Series({
        'bowling_points': points,
        'wickets_taken': wickets,
        'runs_conceded': runs_conceded,
        'overs_bowled': overs
    })

# --- Main Logic ---
print("Loading delivery data...")
all_deliveries = []
match_dates = {} # Dictionary to store match_id -> date mapping

for file in DELIVERY_FILES:
    if os.path.exists(file):
        try:
            df = pd.read_csv(file, low_memory=False)
            # Basic validation
            deliv_required_cols = ['match_id', 'date', 'innings', 'over', 'striker', 'bowler', 'runs_of_bat', 'extras', 'wicket_type', 'player_dismissed', 'wide', 'noballs', 'legbyes', 'byes']
            if all(col in df.columns for col in deliv_required_cols):
                 # Extract match dates
                 match_date_info = df[['match_id', 'date']].drop_duplicates()
                 for _, row in match_date_info.iterrows():
                     match_id = row['match_id']
                     date_str = row['date']
                     if match_id not in match_dates:
                          try:
                              # Attempt to parse the date, store as datetime object
                              match_dates[match_id] = pd.to_datetime(date_str, errors='coerce')
                          except Exception as e:
                              print(f"Warning: Could not parse date '{date_str}' for match_id {match_id} in {file}. Error: {e}")
                              match_dates[match_id] = pd.NaT # Store NaT if parsing fails
                              
                 all_deliveries.append(df)
                 print(f"Loaded {file} ({len(df)} rows)")
            else:
                missing = [col for col in deliv_required_cols if col not in df.columns]
                print(f"Warning: Skipping {file} due to missing required columns: {missing}")

        except Exception as e:
            print(f"Error loading {file}: {e}")
    else:
        print(f"Warning: File not found - {file}")

if not all_deliveries:
    print("Error: No valid delivery files found or loaded. Exiting.")
    exit()

deliveries_df = pd.concat(all_deliveries, ignore_index=True)
print(f"Total deliveries loaded: {len(deliveries_df)}")

# Clean player names
deliveries_df['striker'] = clean_player_name(deliveries_df['striker'])
deliveries_df['bowler'] = clean_player_name(deliveries_df['bowler'])
deliveries_df['player_dismissed'] = clean_player_name(deliveries_df['player_dismissed'])

# Convert numeric columns
numeric_cols = ['runs_of_bat', 'extras', 'wide', 'legbyes', 'byes', 'noballs']
for col in numeric_cols:
    deliveries_df[col] = pd.to_numeric(deliveries_df[col], errors='coerce').fillna(0)

print("Calculating batting points...")
batting_stats = deliveries_df.groupby(['match_id', 'striker']).apply(calculate_batting_points, include_groups=False).reset_index()
batting_stats.rename(columns={'striker': 'player_name'}, inplace=True)
print(f"Calculated batting points for {len(batting_stats)} player-match entries.")

print("Calculating bowling points...")
bowling_stats = deliveries_df.groupby(['match_id', 'bowler']).apply(calculate_bowling_points, include_groups=False).reset_index()
bowling_stats.rename(columns={'bowler': 'player_name'}, inplace=True)
print(f"Calculated bowling points for {len(bowling_stats)} player-match entries.")

# --- Fielding Points Calculation ---
print("Loading and processing fielding data...")
fielding_points_df = pd.DataFrame() # Initialize empty dataframe
if os.path.exists(FIELDING_DATA_FILE):
    try:
        fielding_df = pd.read_csv(FIELDING_DATA_FILE, low_memory=False)
        fielding_required_cols = ['match_id', 'fullName', 'Fielding_FP']
        if all(col in fielding_df.columns for col in fielding_required_cols):
            fielding_df = fielding_df[fielding_required_cols].copy()
            fielding_df.rename(columns={'fullName': 'player_name', 'Fielding_FP': 'fielding_points'}, inplace=True)
            fielding_df['player_name'] = clean_player_name(fielding_df['player_name'])
            fielding_df['match_id'] = pd.to_numeric(fielding_df['match_id'], errors='coerce')
            fielding_df['fielding_points'] = pd.to_numeric(fielding_df['fielding_points'], errors='coerce').fillna(0)
            fielding_df.dropna(subset=['match_id', 'player_name'], inplace=True)
            fielding_points_df = fielding_df.groupby(['match_id', 'player_name'])['fielding_points'].sum().reset_index()
            print(f"Processed fielding points for {len(fielding_points_df)} player-match entries.")
        else:
            print(f"Warning: Skipping {FIELDING_DATA_FILE} due to missing required columns (match_id, fullName, Fielding_FP).")
            fielding_points_df = pd.DataFrame(columns=['match_id', 'player_name', 'fielding_points'])
    except Exception as e:
        print(f"Error loading or processing {FIELDING_DATA_FILE}: {e}")
        fielding_points_df = pd.DataFrame(columns=['match_id', 'player_name', 'fielding_points'])
else:
    print(f"Warning: Fielding data file not found ({FIELDING_DATA_FILE}). Skipping fielding points.")
    fielding_points_df = pd.DataFrame(columns=['match_id', 'player_name', 'fielding_points'])

# --- Combine Points ---
print("Combining points...")
player_match_points = pd.merge(batting_stats[['match_id', 'player_name', 'batting_points']],
                               bowling_stats[['match_id', 'player_name', 'bowling_points']],
                               on=['match_id', 'player_name'],
                               how='outer')
if not fielding_points_df.empty:
    player_match_points['match_id'] = pd.to_numeric(player_match_points['match_id'], errors='coerce') # Ensure type before merge
    player_match_points = pd.merge(player_match_points,
                                   fielding_points_df,
                                   on=['match_id', 'player_name'],
                                   how='left')
else:
    player_match_points['fielding_points'] = 0

point_cols = ['batting_points', 'bowling_points', 'fielding_points']
for col in point_cols:
    if col not in player_match_points.columns:
         player_match_points[col] = 0
    player_match_points[col] = player_match_points[col].fillna(0)

player_match_points['fantasy_points'] = player_match_points[point_cols].sum(axis=1)

# --- Add Match Dates Directly ---
print("Adding match dates from delivery files...")

# Create a DataFrame from the extracted match_dates dictionary
match_dates_df = pd.DataFrame(list(match_dates.items()), columns=['match_id', 'match_date'])
match_dates_df['match_id'] = pd.to_numeric(match_dates_df['match_id'], errors='coerce')
match_dates_df.dropna(subset=['match_id', 'match_date'], inplace=True) # Drop rows where ID or Date is bad

# Ensure player_match_points match_id is numeric for merging
player_match_points['match_id'] = pd.to_numeric(player_match_points['match_id'], errors='coerce')
player_match_points.dropna(subset=['match_id'], inplace=True)

# Merge the dates
player_history = pd.merge(player_match_points[['match_id', 'player_name', 'fantasy_points']],
                          match_dates_df,
                          on='match_id',
                          how='left')

missing_dates = player_history['match_date'].isnull().sum()
if missing_dates > 0:
    print(f"Warning: Could not associate dates for {missing_dates} entries (check date parsing warnings during load)." )

# --- Final Output ---
# Ensure required columns exist
final_cols = ['player_name', 'match_id', 'match_date', 'fantasy_points']
for col in final_cols:
    if col not in player_history.columns:
        if col == 'match_date':
             player_history[col] = pd.NaT
        else:
             player_history[col] = 0

player_history = player_history[final_cols]

# Sort by player and date for consistency
player_history = player_history.sort_values(by=['player_name', 'match_date'], ascending=[True, False])

print(f"Saving player history to {OUTPUT_FILE}...")
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)
player_history.to_csv(OUTPUT_FILE, index=False)

print("Finished calculating fantasy points.")
print(f"Total historical records created: {len(player_history)}") 