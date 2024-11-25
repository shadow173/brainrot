from typing import Optional, Tuple
import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
import logging
from dotenv import load_dotenv
import os
import time
from pathlib import Path
from utils import create_explanation_video, verify_video_playability, get_video_download_link

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set page config - Must be the first Streamlit command
st.set_page_config(page_title="PDF to Brainrot", page_icon="📚")


def main() -> None:
  """
    Main function to run the Streamlit application.
    """
  # Cache the initial setup
  st.title("PDF to Brainrot 📚")
  st.write("Upload a PDF file and get a brainrot video explaining it!")
  st.write(
      "Made with ❤️ by [Your Name](https://github.com/your-github-username)")

  # Your code will go below here


if __name__ == "__main__":
  main()
