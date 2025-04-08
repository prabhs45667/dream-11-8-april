# Dream11 Team Predictor

An advanced Dream11 Fantasy Cricket team prediction system that uses machine learning to predict the best possible team for any IPL match.

## Features

- Predicts optimal Dream11 team based on player performance, match conditions, and historical data
- Uses advanced machine learning models (XGBoost, LightGBM, Neural Networks)
- Implements linear programming for team optimization
- Provides backup players for each role
- Interactive Streamlit interface with visualizations
- Downloads predicted team as CSV

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dream11-predictor.git
cd dream11-predictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `data_preprocessor.py`: Handles data loading and preprocessing
- `model_trainer.py`: Implements model training and optimization
- `team_optimizer.py`: Optimizes team selection using linear programming
- `app.py`: Streamlit web application
- `dataset/`: Contains all required CSV files
- `models/`: Stores trained models

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Select the teams playing in the match from the sidebar
3. View the predicted team, backup players, and visualizations
4. Download the team as CSV if needed

## Data Sources

The model uses various data sources:
- Ball-by-ball data
- Player statistics
- Match results
- Fantasy points history
- Fielding statistics
- Performance metrics (runs, wickets, economy rate, etc.)

## Model Architecture

1. Data Preprocessing
   - Feature engineering
   - Data cleaning
   - Categorical encoding

2. Model Training
   - XGBoost
   - LightGBM
   - Neural Network
   - Model selection based on performance

3. Team Optimization
   - Linear programming
   - Role constraints
   - Credit constraints
   - Backup player selection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- IPL data providers
- Dream11 for the fantasy cricket platform
- Open source machine learning community 