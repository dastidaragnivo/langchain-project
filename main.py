import asyncio
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import os
print(os.getenv("LANGSMITH_PROJECT"))  # Print the value of LANGSMITH_PROJECT

async def main():
    print("Hello from mcp-crash-course!")


if __name__ == "__main__":
    asyncio.run(main())
