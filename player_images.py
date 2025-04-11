# Player Images Module
import os
import streamlit as st
import base64
from pathlib import Path

def get_image_data_url(image_path):
    """Convert a local image to a data URL for use in Streamlit"""
    try:
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
            b64_img = base64.b64encode(img_bytes).decode()
            return f"data:image/jpeg;base64,{b64_img}"
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # Return a default online image if local path fails
        return "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/313100/313149.png"

def get_player_image_url(player_name, player_team=None):
    """Get URL for player image
    
    Args:
        player_name: Name of the player
        player_team: Team code of the player (e.g., 'MI', 'RCB', etc.)
    """
    # Team logo image paths
    team_logo_map = {
        'CSK': 'images/csk.jpeg',
        'CHE': 'images/csk.jpeg',
        'DC': 'images/dc.jpeg',
        'GT': 'images/gt.jpeg',
        'KKR': 'images/kkr.jpeg',
        'LSG': 'images/lsg.jpeg',
        'MI': 'images/mi.jpeg',
        'PBKS': 'images/pbks.jpeg',
        'RCB': 'images/rcb.jpeg',
        'RR': 'images/rr.jpeg',
        'SRH': 'images/srh.jpeg',
    }
    
    # Online image URLs for popular players (use only if team logo not available)
    player_image_map = {
        # Indian players
        "Virat Kohli": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316600/316605.png",
        "Rohit Sharma": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316500/316584.png",
        "MS Dhoni": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/319900/319946.png",
        "KL Rahul": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316600/316619.png",
        "Jasprit Bumrah": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316600/316624.png",
        "Rishabh Pant": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316600/316628.png",
        "Hardik Pandya": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316600/316636.png",
        "Ravindra Jadeja": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316600/316601.png",
        "Suryakumar Yadav": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316600/316631.png",
        "Shubman Gill": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/324700/324710.png",
        "Yashasvi Jaiswal": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/324700/324713.png",
        "Ravichandran Ashwin": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316600/316597.png",
        "Yuzvendra Chahal": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316600/316727.png",
        "Mohammed Shami": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316600/316691.png",
        "Mohammed Siraj": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316600/316746.png",
        
        # International players
        "Jos Buttler": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316500/316520.png",
        "David Warner": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316500/316546.png",
        "Quinton de Kock": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316500/316536.png",
        "Kagiso Rabada": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316500/316533.png",
        "Rashid Khan": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316500/316527.png",
        "Kane Williamson": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316500/316549.png",
        "Andre Russell": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316500/316510.png",
        "Glenn Maxwell": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316500/316547.png",
        "Mitchell Starc": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316500/316548.png",
        "Trent Boult": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316500/316554.png",
        "Kieron Pollard": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316500/316513.png",
        "Faf du Plessis": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316500/316532.png",
        "Liam Livingstone": "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/323300/323317.png",
    }
    
    # Clean the player name
    clean_name = player_name.strip() if player_name else "Unknown Player"
    
    # PRIORITY 1: Use the explicitly provided team if available
    if player_team and player_team in team_logo_map:
        return get_image_data_url(team_logo_map[player_team])
    
    # PRIORITY 2: Try to extract team from player name if not provided
    # Look for team code in player name (e.g., "Player Name (MI)")
    team_in_name = None
    
    # Check for team in parentheses
    if "(" in clean_name and ")" in clean_name:
        team_part = clean_name.split("(")[1].split(")")[0].strip().upper()
        if team_part in team_logo_map:
            return get_image_data_url(team_logo_map[team_part])
        
    # Check if team code is present in the name
    for team_code in team_logo_map:
        if f" {team_code}" in clean_name.upper() or f"({team_code})" in clean_name.upper():
            return get_image_data_url(team_logo_map[team_code])
    
    # PRIORITY 3: Use player's online image if it exists
    if clean_name in player_image_map:
        return player_image_map[clean_name]
    
    # PRIORITY 4: Try substring matching for known players
    for known_player, url in player_image_map.items():
        if clean_name in known_player or known_player in clean_name:
            return url
    
    # PRIORITY 5: Role-based default images if role is in the name
    if "(" in clean_name and ")" in clean_name:
        role_part = clean_name.split("(")[1].split(")")[0].strip().upper()
        role_default_images = {
            "WK": "images/csk.jpeg",  # WK gets CSK logo
            "BAT": "images/rcb.jpeg",  # BAT gets RCB logo
            "AR": "images/mi.jpeg",   # AR gets MI logo
            "BOWL": "images/dc.jpeg"  # BOWL gets DC logo
        }
        
        if role_part in role_default_images:
            return get_image_data_url(role_default_images[role_part])
    
    # PRIORITY 6: Default to a generic team logo (CSK in this case)
    return get_image_data_url('images/csk.jpeg')

def add_player_card_styles():
    """Add CSS styles for player cards"""
    
    st.markdown("""
    <style>
    .player-card {
        border: 1px solid rgba(151, 151, 151, 0.2);
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
        text-align: center;
        background-color: rgba(49, 51, 63, 0.7);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .player-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 7px 10px rgba(0, 0, 0, 0.15);
    }
    .player-card img {
        border-radius: 50%;
        width: 70px;
        height: 70px;
        object-fit: cover;
        margin-bottom: 12px;
        border: 3px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.15);
    }
    .player-name {
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 5px;
        color: #fff;
    }
    .player-points {
        color: #2ecc71;
        font-weight: bold;
        font-size: 0.95em;
        margin-top: 5px;
    }
    .player-role {
        color: #bdc3c7;
        font-size: 0.9em;
        margin-bottom: 5px;
    }
    .captain-marker {
        background-color: #f1c40f;
        color: #2c3e50;
        border-radius: 4px;
        padding: 1px 6px;
        margin-left: 5px;
        font-weight: bold;
        font-size: 0.75em;
        display: inline-block;
        vertical-align: middle;
    }
    .vice-captain-marker {
        background-color: #95a5a6;
        color: #2c3e50;
        border-radius: 4px;
        padding: 1px 6px;
        margin-left: 5px;
        font-weight: bold;
        font-size: 0.75em;
        display: inline-block;
        vertical-align: middle;
    }
    
    /* Impact player card styling */
    .impact-player-card {
        border: 1px solid rgba(41, 128, 185, 0.3);
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
        text-align: center;
        background-color: rgba(52, 73, 94, 0.7);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .impact-player-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 7px 10px rgba(0, 0, 0, 0.15);
    }
    .impact-player-card img {
        border-radius: 50%;
        width: 70px;
        height: 70px;
        object-fit: cover;
        margin-bottom: 12px;
        border: 3px solid rgba(52, 152, 219, 0.3);
    }
    
    /* Role-based colors */
    .role-WK .player-name, .role-WK .player-role {
        border-left: 3px solid #e74c3c;
        padding-left: 5px;
    }
    .role-BAT .player-name, .role-BAT .player-role {
        border-left: 3px solid #3498db;
        padding-left: 5px;
    }
    .role-AR .player-name, .role-AR .player-role {
        border-left: 3px solid #f39c12;
        padding-left: 5px;
    }
    .role-BOWL .player-name, .role-BOWL .player-role {
        border-left: 3px solid #2ecc71;
        padding-left: 5px;
    }
    </style>
    """, unsafe_allow_html=True)