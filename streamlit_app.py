import logging
from app import Dream11App

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Streamlit app...")
    try:
        # Initialize the Dream11App
        app = Dream11App()
        
        # Run the Streamlit interface
        app.run_app()
        
    except Exception as e:
        logger.error(f"Error starting Streamlit app: {str(e)}")
        import traceback
        traceback.print_exc() 