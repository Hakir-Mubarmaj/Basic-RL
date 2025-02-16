import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size=3, model_path="model.pth"):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.model_path = model_path
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.load_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size),
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.tensor(next_state, dtype=torch.float32)
                target = reward + self.gamma * torch.max(self.model(next_state)).item()

            state = torch.tensor(state, dtype=torch.float32)
            target_f = self.model(state).detach().clone()
            target_f[action] = target

            optimizer.zero_grad()
            loss = criterion(self.model(state)[action], torch.tensor(target, dtype=torch.float32))
            loss.backward()
            optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.save_model()

    def save_model(self):
        """Save model weights."""
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        """Load model weights if available."""
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print("✅ Model loaded successfully!")
