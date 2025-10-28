# AI Chatbot with Long-Term Memory



This project implements a personalized AI chatbot that uses the DeepSeek API for language generation and the Cognio server for persistent, long-term memory. This allows the chatbot to remember user-specific information (like your name or preferences) across separate conversations.

This system consists of two main components:

1. **The Cognio Server (`src/server.py`)**: A backend service that stores and retrieves memories from a database.
2. **The Chatbot Client (`main.py`)**: An interactive command-line interface that connects to DeepSeek and the Cognio Server.



## System Requirements



- Python 3.7+
- A valid DeepSeek API Key



## Installation



1. **Install Dependencies**: Install all necessary Python packages for both the server and the client.

   Bash

   ```
   pip install -r requirements.txt
   ```

   *(Note: This assumes you are using a combined `requirements.txt` file that includes `fastapi`, `uvicorn`, `sentence-transformers`, `openai`, and `requests`).*

2. **Set API Key**: Open `main.py` in a text editor and set your DeepSeek API key on line 11:

   Python

   ```
   os.environ["DEEPSEEK_API_KEY"] = "YOUR_DEEPSEEK_API_KEY_HERE"
   ```



## How to Run



This project requires **two separate terminals** to run simultaneously.



### Terminal 1: Start the Cognio Memory Server



In your first terminal, navigate to the project's root directory and start the Cognio server using `uvicorn`. This application listens for requests to save or retrieve memories.

(Note: This command assumes you have renamed `src/main.py` to `src/server.py`).

Bash

```
uvicorn src.server:app --host 0.0.0.0 --port 8080
```

Leave this terminal running. You should see "Application startup complete."



### Terminal 2: Start the Chatbot Client



In your second terminal, run the interactive chatbot script (`main.py`). This script will connect to the server you just started.

Bash

```
python main.py
```



## How to Use



1. Once the chatbot starts, it will first connect to the server and API, then ask for your `User ID`:

   ```
   Please enter your User ID (e.g., 'liutianming'):
   ```

2. Enter a unique ID. This ID is used to create a separate "memory box" for you. All facts will be saved under this ID.

3. Start chatting! The AI will automatically try to save new facts (like your name or preferences) and retrieve old ones.

4. **To test the memory**:

   - Tell the AI a new fact (e.g., "My name is...").
   - Stop the chatbot script (type `quit`).
   - Restart the chatbot script (`python main.py`).
   - Log in with the **exact same User ID**.
   - Ask the AI about the fact you told it (e.g., "What is my name?").

5. Type `quit` or `exit` to stop the chatbot.



## Troubleshooting



- **AI can't remember facts (like my name)**: This is likely due to the semantic search threshold. The query ("What is my name?") might not be "similar" enough to the stored fact ("My name is...").
  - **Fix**: Open `main.py`, find the `search_memory` function, and lower the `threshold` value (e.g., from `0.3` to `0.2`) to make the search "less strict".
- **How to clear all memory**: Stop the Cognio server (Terminal 1) and delete the `memory.db` file from the `data/` directory. Restart the server to create a fresh, empty database.