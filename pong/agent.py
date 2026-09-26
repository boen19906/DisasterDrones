import random
from collections import deque
import numpy as np
import torch

from game import PongGameAI
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self, model_file=None):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.99 # Increased from 0.9 to allow long-term planning
        self.memory = deque(maxlen=MAX_MEMORY)
        self.last_q_values = [0.0, 0.0, 0.0]

        # 7 Inputs: player_y, ball_x, ball_y, ball_vx, ball_vy, opponent_y, y_distance
        self.model = Linear_QNet(input_size=7, hidden_size=256, output_size=3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        if model_file:
            self.load_model(model_file)

    def get_state(self, game):
        # Normalize states for better neural network performance
        paddle_center_y = game.player_y + 40 # PADDLE_HEIGHT // 2 is 40
        y_dist = (paddle_center_y - game.ball_y) / game.h

        state = [
            game.player_y / game.h,
            game.ball_x / game.w,
            game.ball_y / game.h,
            game.ball_vx / 15.0, # Approximate max speed
            game.ball_vy / 15.0,
            game.opponent_y / game.h,
            y_dist
        ]
        return np.array(state, dtype=np.float32)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        if not mini_sample:
            return 0.0

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        return self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        return self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, test_mode=False):
        final_move = [0, 0, 0]

        state0 = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            prediction = self.model(state0)
            self.last_q_values = prediction.cpu().numpy().tolist()

        if test_mode:
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            return final_move

        # Slowed down epsilon decay so it explores for the first 8000 games
        self.epsilon = max(0.10, 1.0 - (self.n_games / 8000.0))

        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def save_model(self, file_name='model.pth'):
        self.model.save(file_name)

    def load_model(self, file_path):
        return self.model.load(file_path)
