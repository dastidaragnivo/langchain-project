import sys
from pathlib import Path

from dotenv import load_dotenv

# Make the repo root importable when this file is run from another working directory.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG!\n")
    question = input("Enter your question: \n").strip()
    print("\n")
    print(app.invoke(input={"question": question}))