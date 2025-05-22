# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file into environment

API_BASE = os.getenv("API_BASE")
API_KEY = os.getenv("API_KEY")

# Check if the variables are loaded
if API_KEY is None:
    print("Warning: API_KEY not found in environment variables. Please check your .env file.")
if API_BASE is None:
    print("Warning: API_BASE not found in environment variables. Please check your .env file.")
