import os
import requests
from openai import OpenAI, AuthenticationError, APITimeoutError

# --- 1. Configuration ---

#  Please enter your DeepSeek API key here
os.environ["DEEPSEEK_API_KEY"] = "sk-60b1be137abf445c9d23f5ddb899b363"  # Use your key

# Address of your Memory server (make sure it's running)
MEMORY_BASE_URL = "http://localhost:8080"


# --- 2. Memory Client  ---

class MemoryClient:
    """Python client for interacting with the local Memory server"""

    def __init__(self, base_url):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}

    def set_api_key(self, api_key):
        self.headers["X-API-Key"] = api_key

    def check_health(self):
        """Check if the Memory server is running"""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            print(f"Memory server connected successfully (at {self.base_url})")
            return True
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to Memory server at {self.base_url}")
            print("Please ensure you are running 'uvicorn src.server:app --port 8080' in another terminal")
            return False
        except requests.exceptions.RequestException as e:
            print(f"Memory server error: {e}")
            return False

    def save_memory(self, text, project, tags=None):
        """ (Store) Save a new memory """
        #
        url = f"{self.base_url}/memory/save"
        payload = {"text": text, "project": project, "tags": tags or []}
        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"\n[!!!] Error saving memory (HTTP Error): {e}\n")
            return None

    def search_memory(self, query, project, limit=3, threshold=0.2):
        """
        (Retrieve) Search for relevant memories
        """
        #
        url = f"{self.base_url}/memory/search"
        params = {"q": query, "project": project, "limit": limit, "threshold": threshold}
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            return response.json().get("results", [])
        except requests.exceptions.RequestException as e:
            print(f"\n[!!!] Error searching memory (HTTP Error): {e}\n")
            return []


# --- 3. DeepSeek LLM Logic ---

def get_deepseek_response(client, messages, model="deepseek-chat"):
    """ (Inject) Call the DeepSeek API """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content
    except AuthenticationError as e:
        print(f"DeepSeek API key error: {e}")
        return "Sorry, my API key is configured incorrectly."
    except APITimeoutError:
        print("DeepSeek API request timed out.")
        return "Sorry, the request timed out. Please try again later."
    except Exception as e:
        print(f"Unknown error calling DeepSeek: {e}")
        return "Sorry, I am unable to process your request."


def summarize_facts_for_memory(client, user_message, ai_response):
    """ (Store) Use DeepSeek to decide what to remember """
    messages = [
        {"role": "system", "content": (
            "You are a fact-summarizing assistant."
            "Based on the user's message and the AI's response, summarize the core facts to be remembered long-term in one or two sentences."
            "Example: 'The user's name is John' or 'The user likes the color blue'.\n"
            "If there are no new facts to remember, reply only with 'None'."
        )},
        {"role": "user", "content": (
            f"Please summarize the core facts from the following conversation:\n"
            f"User: {user_message}\n"
            f"AI: {ai_response}"
        )}
    ]
    try:
        facts = get_deepseek_response(client, messages)
        facts_stripped = facts.strip()
        if facts_stripped.lower() == "none" or "no new facts" in facts_stripped.lower() or not facts_stripped:
            return None
        return facts_stripped
    except Exception as e:
        print(f"Error during DeepSeek summarization: {e}")
        return None


# --- 4. Core Chat Loop ---

def chat_with_memory(memory_client, deepseek_client, user_id, user_message, chat_history):
    """
    Execute the full Retrieve-Inject-Store loop
    """
    # 1. "Retrieve"
    print("... Retrieving memories from Memory Server ...")
    relevant_memories = memory_client.search_memory(
        query=user_message,
        project=user_id
    )

    # 2. "Inject"
    system_instruction = (
        "You are an AI assistant."
        "Please provide a personalized response to the user's latest question based on the following long-term memories and short-term conversation history."
    )
    if relevant_memories:
        system_instruction += "\n\n--- Long-term Memory (from Memory Server) ---\n"
        for i, mem in enumerate(relevant_memories):
            # We print the score for debugging
            print(f"    -> Retrieved memory (Score: {mem['score']:.2f}): {mem['text']}")
            system_instruction += f"- {mem['text']}\n"
    else:
        system_instruction += "\n\n(No relevant long-term memory)"

    messages = [{"role": "system", "content": system_instruction}]
    for turn in chat_history[-5:]:
        messages.append({"role": turn["role"], "content": turn["parts"]})
    messages.append({"role": "user", "content": user_message})

    print(f"--- Building prompt for DeepSeek ({len(messages)} messages total) ---")

    ai_response = get_deepseek_response(deepseek_client, messages)
    print(f"AI: {ai_response}")

    # 3. "Store"
    print("\n... Requesting DeepSeek to summarize new facts ...")
    new_fact = summarize_facts_for_memory(deepseek_client, user_message, ai_response)

    if new_fact:
        print(f"... Saving new fact to Memory Server: {new_fact} ...")
        #
        save_result = memory_client.save_memory(
            text=new_fact,
            project=user_id,
            tags=["auto-summary", "deepseek"]
        )
        if save_result:
            print(f"    -> Save successful (ID: {save_result.get('id')})")
        else:
            print("    -> (!!!) Save FAILED (Check Memory server logs) (!!!)")
    else:
        print("... DeepSeek found no new facts to store ...")

    return ai_response


# --- 5. Main Program ---

def main():
    if "DEEPSEEK_API_KEY" not in os.environ or os.environ["DEEPSEEK_API_KEY"] == "YOUR_DEEPSEEK_API_KEY_HERE":
        print("Error: Please set your DEEPSEEK_API_KEY on line 11 of the script.")
        return

    memory = MemoryClient(MEMORY_BASE_URL)

    if not memory.check_health():
        return

    try:
        print("Configuring DeepSeek API client...")
        client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1"
        )
        client.models.list()
        print("DeepSeek API client configured successfully.")
    except AuthenticationError:
        print("DeepSeek API key is invalid or has expired. Please check.")
        return
    except Exception as e:
        print(f"DeepSeek API client initialization failed: {e}")
        return

    print("\n" + "=" * 50)
    print(" Welcome to the Memory Chatbot (DeepSeek Edition)")
    print("=" * 50 + "\n")

    USER_ID = ""
    while not USER_ID:
        USER_ID = input("Please enter your User ID: ")
        if not USER_ID:
            print("User ID cannot be empty. Please try again.")

    print(f"\nLogin successful! Your memories will be saved and retrieved for user '{USER_ID}'.")
    print("Type 'quit' or 'exit' to end the program.")
    print("=" * 50 + "\n")

    chat_history = []

    while True:
        try:
            user_input = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExit command detected.")
            break

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye! Thanks for chatting.")
            break

        if not user_input:
            continue

        ai_reply = chat_with_memory(
            memory_client=memory,
            deepseek_client=client,
            user_id=USER_ID,
            user_message=user_input,
            chat_history=chat_history
        )

        chat_history.append({'role': 'user', 'parts': user_input})
        chat_history.append({'role': 'assistant', 'parts': ai_reply})


if __name__ == "__main__":
    main()