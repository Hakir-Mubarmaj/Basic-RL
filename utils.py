import re

def clean_text(text):
    """Cleans input text by removing unnecessary characters."""
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text

def map_action(action_index):
    """Maps action index to a descriptive action."""
    actions = {0: "ask_clarification", 1: "store_fact"}
    return actions.get(action_index, "unknown_action")
