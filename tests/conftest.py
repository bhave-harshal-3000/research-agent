import os
import sys

# Make the project root importable and make sure tests never use real API keys.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["GROQ_API_KEY"] = "test-key-not-real"
os.environ["TAVILY_API_KEY"] = "test-key-not-real"
