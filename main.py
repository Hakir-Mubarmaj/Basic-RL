from agent import DQNAgent
from environment import InteractiveEnv
from knowledge_graph import KnowledgeGraph
from utils import clean_text
import numpy as np

def main():
    state_size = 5
    action_size = 3  # Now includes "Retrieve Answer"
    agent = DQNAgent(state_size, action_size)
    env = InteractiveEnv()
    knowledge_graph = KnowledgeGraph()
    
    print("\n🚀 Welcome to the Interactive Learning AI!")
    print("💡 Teach me about the world, and I will learn step by step.\n")

    for e in range(100):
        state = env.reset()
        
        user_input = input("🗣️ Enter a fact or ask a question: ")
        user_input = clean_text(user_input)
        
        action_index = agent.act(state)
        
        if action_index == 0:
            print("🤖 AI: Can you clarify what you mean?")
            user_response = input("🗣️ You: ")
            next_state, reward, done = env.step(action_index, clean_text(user_response))
        
        elif action_index == 1:
            print("🤖 AI: Got it! Storing this fact.")
            knowledge_graph.add_fact(user_input)
        
        elif action_index == 2:
            answer = knowledge_graph.query_fact(user_input)
            if answer:
                print(f"🤖 AI: I remember! {answer}")
            else:
                print("🤖 AI: I don’t know this yet.")

if __name__ == "__main__":
    main()
