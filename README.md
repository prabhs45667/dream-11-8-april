# Dream11 IPL Team Predictor

A machine learning-based application to predict the best Dream11 team for IPL matches using player performance data from 2022-2024.

## Features

- **Data-Driven Predictions**: Uses actual ball-by-ball data from IPL 2022-2024 to predict player performance
- **Multiple ML Models**: Combines Decision Trees, KNN, Random Forest, Gradient Boosting, XGBoost, and Neural Networks
- **Optimized Team Selection**: Uses Linear Programming to select the optimal team while respecting Dream11 constraints
- **Captain & Vice-Captain Selection**: Automatically selects the best captain and vice-captain
- **Interactive UI**: Easy-to-use Streamlit interface for team prediction

## Quick Start

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**:
   ```bash
   streamlit run improved_app.py
   ```

3. **First-time Setup**: The first time you run the app, it will:
   - Load and preprocess the IPL data
   - Train multiple prediction models
   - Save models for future use
   - Generate feature importance visualization

4. **Predict Teams**: 
   - Select two teams from the dropdown menus
   - Click "Predict Dream11 Team"
   - View and download your optimized Dream11 team

## Data Requirements

The following data files should be in the `dataset` folder:
- `ipl_2022_deliveries.csv`, `ipl_2023_deliveries.csv`, `ipl_2024_deliveries.csv`: Ball-by-ball data
- `SquadPlayerNames_IndianT20League - SquadData_AllTeams.csv`: Player information including roles and credits

## Implementation Details

### 1. Data Processing & Feature Engineering
- Extracts batting and bowling statistics from ball-by-ball data
- Creates comprehensive player features (strike rate, economy, etc.)
- Calculates fantasy points based on Dream11 scoring rules

### 2. Model Training
- Trains 6 different models and selects the best performing one:
  - Decision Tree
  - K-Nearest Neighbors
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Neural Network

### 3. Team Optimization
- Uses PuLP linear programming to optimize team selection
- Enforces Dream11 constraints:
  - 11 players total
  - 1-4 wicket-keepers
  - 3-6 batsmen
  - 1-4 all-rounders
  - 3-6 bowlers
  - Maximum 7 players from one team
  - Maximum 100 credits

### 4. Interactive Visualization
- Shows team composition breakdown
- Displays predicted points for each player
- Highlights captain and vice-captain selections

## Future Improvements

For future versions, consider implementing:
- Reinforcement learning for adaptive team selection
- Player form tracking over time
- Opposition-specific performance analysis
- Weather and pitch condition integration
- Integration with real-time data sources

## Requirements

```
pandas
numpy
scikit-learn
xgboost
tensorflow
pulp
streamlit
matplotlib
seaborn
joblib
tqdm
```

Create a `requirements.txt` file with the above dependencies.

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