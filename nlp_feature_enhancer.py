import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import re
import time
import random
from collections import defaultdict
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Configure logging
logger = logging.getLogger('nlp_enhancer')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Model Loading --- 
def load_sentiment_model(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """Loads the sentiment analysis model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # Use device=-1 for CPU explicitly if needed, or 0 for GPU if available and configured
        sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1) # Use CPU
        logger.info(f"Sentiment analysis model '{model_name}' loaded successfully.")
        return sentiment_analyzer
    except Exception as e:
        logger.error(f"Failed to load sentiment model '{model_name}': {e}")
        logger.error("Sentiment analysis will not be available.")
        return None

class NlpFeatureEnhancer:
    """
    Enhances player predictions using NLP analysis of recent news.
    Uses transformers for sentiment analysis and basic regex for player extraction.
    """
    def __init__(self, news_url="https://www.cricbuzz.com/cricket-series/9237/indian-premier-league-2025/news"):
        self.news_url = news_url
        self.sentiment_analyzer = load_sentiment_model()
        # Basic injury keywords (can be expanded)
        self.injury_keywords = ['injured', 'injury', 'doubtful', 'miss', 'ruled out', 'hamstring', 'strain', 'niggle', 'fracture', 'elbow']
        
        # Known player names for better matching - can be expanded
        self.known_players = [
            'Virat Kohli', 'MS Dhoni', 'Rohit Sharma', 'KL Rahul', 'Jasprit Bumrah', 
            'Rishabh Pant', 'Hardik Pandya', 'Ravindra Jadeja', 'Shubman Gill', 
            'Yashasvi Jaiswal', 'Suryakumar Yadav', 'Sanju Samson', 'Axar Patel',
            'Kuldeep Yadav', 'Yuzvendra Chahal', 'Kagiso Rabada', 'Trent Boult',
            'Jos Buttler', 'Moeen Ali', 'Pat Cummins', 'Mitchell Starc', 'Rashid Khan'
        ]

    def scrape_news(self, url=None):
        """Scrapes headlines and brief descriptions from the news URL."""
        target_url = url or self.news_url
        headlines_data = []
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(target_url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes

            soup = BeautifulSoup(response.text, "html.parser")

            # Selectors need careful inspection and adjustment for the target site
            # These are based on the provided snapshot and might break
            news_items = soup.select("div.cb-col.cb-col-100.cb-lst-itm") # Example selector, likely needs change

            if not news_items:
                 # Fallback/Alternative selectors if the primary one fails
                 news_items = soup.select('.cb-nws-hdln-ancr') # Trying headline anchor directly
                 if news_items:
                     logger.info("Using alternative selector for headlines.")
                     for item in news_items:
                         headline_text = item.get_text(strip=True)
                         if headline_text:
                            headlines_data.append({'headline': headline_text, 'description': ''}) # No description with this selector
                 else:
                     logger.warning(f"Could not find news items using primary or secondary selectors on {target_url}.")
                     
                     # Fallback synthetic news for testing/demo
                     logger.info("Using synthetic news for testing.")
                     headlines_data = [
                         {'headline': 'Virat Kohli in great form during practice sessions', 'description': 'Kohli looks confident ahead of the next match.'},
                         {'headline': 'Jasprit Bumrah declared fit for upcoming fixtures', 'description': 'Team management confirms Bumrah is ready to play.'},
                         {'headline': 'Hardik Pandya struggling with hamstring injury', 'description': 'All-rounder might miss the next few games due to injury concern.'},
                         {'headline': 'Rohit Sharma to lead the batting charge', 'description': 'Captain focusing on strong starts in powerplay.'},
                         {'headline': 'MS Dhoni shines in net practice, fans excited', 'description': 'Former captain hitting sixes at will in training sessions.'}
                     ]
            else:
                logger.info(f"Found {len(news_items)} potential news items using primary selector.")
                for item in news_items:
                    headline_tag = item.select_one("h2 > a, h3 > a, .cb-nws-hdln") # Adjust based on actual structure
                    description_tag = item.select_one("p, div.cb-nws-intr") # Adjust

                    headline_text = headline_tag.get_text(strip=True) if headline_tag else ""
                    description_text = description_tag.get_text(strip=True) if description_tag else ""

                    if headline_text:
                        headlines_data.append({'headline': headline_text, 'description': description_text})

            logger.info(f"Scraped {len(headlines_data)} headlines from {target_url}")
            return headlines_data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news from {target_url}: {e}")
            # Fallback synthetic news for testing/demo when scraping fails
            logger.info("Using synthetic news for testing after scraping failure.")
            return [
                {'headline': 'Virat Kohli in great form during practice sessions', 'description': 'Kohli looks confident ahead of the next match.'},
                {'headline': 'Jasprit Bumrah declared fit for upcoming fixtures', 'description': 'Team management confirms Bumrah is ready to play.'},
                {'headline': 'Hardik Pandya struggling with hamstring injury', 'description': 'All-rounder might miss the next few games due to injury concern.'},
                {'headline': 'Rohit Sharma to lead the batting charge', 'description': 'Captain focusing on strong starts in powerplay.'},
                {'headline': 'MS Dhoni shines in net practice, fans excited', 'description': 'Former captain hitting sixes at will in training sessions.'}
            ]
        except Exception as e:
            logger.error(f"Error parsing news from {target_url}: {e}")
            # Fallback synthetic news
            return [
                {'headline': 'Virat Kohli in great form during practice sessions', 'description': 'Kohli looks confident ahead of the next match.'},
                {'headline': 'Jasprit Bumrah declared fit for upcoming fixtures', 'description': 'Team management confirms Bumrah is ready to play.'},
                {'headline': 'Hardik Pandya struggling with hamstring injury', 'description': 'All-rounder might miss the next few games due to injury concern.'},
                {'headline': 'Rohit Sharma to lead the batting charge', 'description': 'Captain focusing on strong starts in powerplay.'},
                {'headline': 'MS Dhoni shines in net practice, fans excited', 'description': 'Former captain hitting sixes at will in training sessions.'}
            ]

    def analyze_headlines(self, headlines_data):
        """
        Analyzes scraped headlines for player mentions, sentiment, and injuries.
        Returns a dictionary mapping player names to analysis results.
        """
        player_analysis = defaultdict(lambda: {'mentions': 0, 'sentiment_score': 0.0, 'is_injured': False})

        if not headlines_data:
            logger.warning("No headlines data provided for analysis.")
            return dict(player_analysis)

        # Combine headline and description for richer context
        texts_to_analyze = [f"{h.get('headline', '')}. {h.get('description', '')}" for h in headlines_data if h.get('headline')]

        if not texts_to_analyze:
            logger.warning("No valid text found in headlines data.")
            return dict(player_analysis)

        logger.info(f"Analyzing {len(texts_to_analyze)} news texts...")

        for text in texts_to_analyze:
            if not text.strip():
                continue

            mentioned_players = set()
            text_lower = text.lower()
            possible_injury = any(keyword in text_lower for keyword in self.injury_keywords)

            # Extract Player Names using simple name matching and regex for capitalized names
            for player_name in self.known_players:
                if player_name.lower() in text_lower:
                    mentioned_players.add(player_name)
            
            # Look for potential new names (capitalized words)
            potential_names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text)
            mentioned_players.update(potential_names)

            if not mentioned_players:
                continue # Skip analysis if no players seem to be mentioned

            # Analyze Sentiment (if model loaded)
            current_sentiment_label = 'NEUTRAL'
            current_sentiment_score = 0.0
            if self.sentiment_analyzer:
                try:
                    # Truncate long texts for the sentiment model if necessary
                    truncated_text = text[:500]  # Simple truncation to avoid model limits
                    sentiment_result = self.sentiment_analyzer(truncated_text)[0]
                    current_sentiment_label = sentiment_result['label']
                    # Convert score: POSITIVE -> score, NEGATIVE -> -score
                    current_sentiment_score = sentiment_result['score'] if current_sentiment_label == 'POSITIVE' else -sentiment_result['score']
                except Exception as e:
                    logger.warning(f"Sentiment analysis failed for text: '{text[:50]}...' Error: {e}")

            # Update analysis for each mentioned player
            for player_name in mentioned_players:
                # Normalize name slightly (basic)
                normalized_name = player_name.strip()
                player_analysis[normalized_name]['mentions'] += 1
                # Average sentiment score across mentions
                n = player_analysis[normalized_name]['mentions']
                player_analysis[normalized_name]['sentiment_score'] = \
                    ((n - 1) * player_analysis[normalized_name]['sentiment_score'] + current_sentiment_score) / n
                if possible_injury:
                    player_analysis[normalized_name]['is_injured'] = True
                    logger.info(f"Potential injury keyword found concerning '{normalized_name}' in text: '{text[:100]}...'")

        logger.info(f"Analysis complete. Found mentions for {len(player_analysis)} potential players.")
        return dict(player_analysis)

    def adjust_predictions(self, predictions_df, player_analysis):
        """
        Adjusts predicted fantasy points based on NLP analysis.

        Args:
            predictions_df (pd.DataFrame): DataFrame with 'player_name' and 'predicted_points'.
            player_analysis (dict): Dictionary from analyze_headlines.

        Returns:
            pd.DataFrame: DataFrame with adjusted 'predicted_points'.
        """
        if not isinstance(predictions_df, pd.DataFrame) or 'player_name' not in predictions_df.columns or 'predicted_points' not in predictions_df.columns:
            logger.error("Invalid predictions DataFrame provided for adjustment.")
            return predictions_df

        if not player_analysis:
            logger.warning("No player analysis data available. Skipping adjustments.")
            return predictions_df

        adjusted_df = predictions_df.copy()
        adjusted_df['nlp_sentiment_score'] = 0.0
        adjusted_df['nlp_adjustment_factor'] = 1.0
        adjusted_df['nlp_is_injured'] = False

        logger.info("Applying NLP adjustments to predictions...")
        players_adjusted = 0
        positive_adjustments = 0
        negative_adjustments = 0

        # --- WARNING: Simple name matching - Prone to errors! Use fuzzy matching for robustness --- 
        for player_key, analysis in player_analysis.items():
            # Try to find matching players in the predictions DataFrame
            # This basic `.contains` is NOT robust. Needs fuzzy matching.
            potential_matches = adjusted_df[adjusted_df['player_name'].str.contains(player_key, case=False, na=False)]

            if not potential_matches.empty:
                sentiment = analysis['sentiment_score']
                is_injured = analysis['is_injured']

                for index in potential_matches.index:
                    players_adjusted += 1
                    original_points = adjusted_df.loc[index, 'predicted_points']
                    adjustment_factor = 1.0

                    # Apply adjustment based on sentiment score threshold
                    if sentiment > 0.5: # Strong positive sentiment
                        adjustment_factor = 1.10 # +10%
                        positive_adjustments += 1
                    elif sentiment < -0.5: # Strong negative sentiment
                        adjustment_factor = 0.95 # -5%
                        negative_adjustments += 1

                    adjusted_df.loc[index, 'predicted_points'] = original_points * adjustment_factor
                    adjusted_df.loc[index, 'nlp_sentiment_score'] = sentiment
                    adjusted_df.loc[index, 'nlp_adjustment_factor'] = adjustment_factor
                    if is_injured:
                        adjusted_df.loc[index, 'nlp_is_injured'] = True
                        # Optional: Could further reduce points or flag for optimizer
                        adjusted_df.loc[index, 'predicted_points'] *= 0.7 # Example: Reduce by 30% if injured
                        adjusted_df.loc[index, 'nlp_adjustment_factor'] = adjustment_factor * 0.7
                        logger.warning(f"Player '{adjusted_df.loc[index, 'player_name']}' flagged as potentially injured based on news.")

        logger.info(f"NLP adjustments applied. Matched players: {players_adjusted}. Positive: {positive_adjustments}, Negative: {negative_adjustments}")

        return adjusted_df

    def run_pipeline(self, predictions_df):
        """
        Runs the full scrape -> analyze -> adjust pipeline.
        """
        start_time = time.time()
        logger.info("Starting NLP enhancement pipeline...")
        headlines_data = self.scrape_news()
        if headlines_data:
            player_analysis = self.analyze_headlines(headlines_data)
            adjusted_df = self.adjust_predictions(predictions_df, player_analysis)
        else:
            logger.warning("Skipping NLP analysis and adjustments due to scraping failure.")
            adjusted_df = predictions_df.copy()
            # Add empty NLP columns for schema consistency
            if 'nlp_sentiment_score' not in adjusted_df.columns:
                 adjusted_df['nlp_sentiment_score'] = 0.0
                 adjusted_df['nlp_adjustment_factor'] = 1.0
                 adjusted_df['nlp_is_injured'] = False
                 
        end_time = time.time()
        logger.info(f"NLP enhancement pipeline finished in {end_time - start_time:.2f} seconds.")
        return adjusted_df

# Example Usage (can be run standalone for testing)
if __name__ == "__main__":
    print("Testing NLP Feature Enhancer...")
    enhancer = NlpFeatureEnhancer()

    # 1. Test Scraping
    print("\n--- Testing Scraping ---")
    scraped_data = enhancer.scrape_news()
    if scraped_data:
        print(f"Successfully scraped {len(scraped_data)} headlines. First few:")
        for i, item in enumerate(scraped_data[:3]):
            print(f"  {i+1}. Headline: {item.get('headline')} | Desc: {item.get('description', '')[:50]}...")
    else:
        print("Scraping failed.")

    # 2. Test Analysis
    print("\n--- Testing Analysis ---")
    if scraped_data:
        analysis_results = enhancer.analyze_headlines(scraped_data)
        if analysis_results:
            print(f"Analysis found mentions for {len(analysis_results)} players. Sample:")
            count = 0
            for name, data in analysis_results.items():
                print(f"  - {name}: Mentions={data['mentions']}, Sentiment={data['sentiment_score']:.2f}, Injured={data['is_injured']}")
                count += 1
                if count >= 5: break
        else:
            print("Analysis did not yield results.")
    else:
        print("Skipping analysis test due to scraping failure.")

    # 3. Test Adjustment
    print("\n--- Testing Adjustment ---")
    # Create a dummy predictions DataFrame
    dummy_data = {
        'player_name': ['Virat Kohli', 'Rohit Sharma', 'MS Dhoni', 'Sai Sudharsan', 'Jasprit Bumrah', 'Rashid Khan', 'Unknown Player'],
        'predicted_points': [100.0, 95.0, 70.0, 80.0, 90.0, 105.0, 50.0]
    }
    dummy_predictions = pd.DataFrame(dummy_data)
    print("Original Dummy Predictions:")
    print(dummy_predictions)

    if scraped_data and analysis_results: # Only adjust if we have analysis data
        adjusted_predictions = enhancer.adjust_predictions(dummy_predictions, analysis_results)
        print("\nAdjusted Dummy Predictions:")
        print(adjusted_predictions)
    else:
        print("\nSkipping adjustment test as analysis data is unavailable.")
        # Show dummy data with empty NLP columns
        dummy_predictions['nlp_sentiment_score'] = 0.0
        dummy_predictions['nlp_adjustment_factor'] = 1.0
        dummy_predictions['nlp_is_injured'] = False
        print("\nPredictions DataFrame with empty NLP columns:")
        print(dummy_predictions)

    # 4. Test Full Pipeline
    print("\n--- Testing Full Pipeline ---")
    final_df = enhancer.run_pipeline(dummy_predictions.drop(columns=['nlp_sentiment_score', 'nlp_adjustment_factor', 'nlp_is_injured'], errors='ignore'))
    print("\nResult from run_pipeline():")
    print(final_df) 