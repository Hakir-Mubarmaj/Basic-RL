class InteractiveEnv:
    def __init__(self):
        self.state = []
        self.knowledge_graph = {}
        self.done = False

    def step(self, action, user_response):
        # Simulate the action taken by the agent
        reward, new_state = self.process_interaction(action, user_response)
        self.state.append(new_state)
        return new_state, reward, self.done

    def process_interaction(self, action, user_response):
        # Example: simple logic to interpret and store facts
        if action == "ask_clarification":
            # Ask user a clarifying question based on current state
            return 1, user_response
        elif action == "store_fact":
            # Store fact in knowledge base
            self.knowledge_graph[len(self.knowledge_graph)] = user_response
            return 1, "Stored fact: " + user_response
        else:
            return -1, "Invalid action"

    def reset(self):
        self.state = []
        self.knowledge_graph = {}
        self.done = False
        return self.state
