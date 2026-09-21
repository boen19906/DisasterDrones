import random
from collections import deque
import numpy as np
import torch

from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    """
    DQN Agent that observes the SnakeGameAI environment, chooses actions
    via an epsilon-greedy policy, and learns from short-term and long-term experience replay.
    """

    def __init__(self, model_file=None):
        self.n_games = 0
        self.epsilon = 0  # Randomness factor
        self.gamma = 0.9   # Discount factor for future rewards
        self.memory = deque(maxlen=MAX_MEMORY)  # Experience replay memory
        self.last_q_values = [0.0, 0.0, 0.0]

        # Neural Network & Trainer
        self.model = Linear_QNet(input_size=11, hidden_size=256, output_size=3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        # Optional pre-trained model loading
        if model_file:
            self.load_model(model_file)

    def get_state(self, game):
        """
        Extracts an 11-dimensional binary feature vector representing the state:
        - Danger Straight / Right / Left (3 bits)
        - Current Direction (4 bits: Left, Right, Up, Down)
        - Food Relative Location (4 bits: Left, Right, Up, Down)
        """
        head = game.snake[0]

        # Points adjacent to head in 4 cardinal directions
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # Current direction booleans
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # 1. Danger Straight ahead
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # 2. Danger Right (clockwise from current direction)
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # 3. Danger Left (counter-clockwise from current direction)
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # 4. Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # 5. Food location relative to head
            game.food.x < game.head.x,  # Food is to the left
            game.food.x > game.head.x,  # Food is to the right
            game.food.y < game.head.y,  # Food is above (up)
            game.food.y > game.head.y   # Food is below (down)
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """Stores a transition experience into replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """
        Samples a mini-batch from experience replay memory and trains the model.
        This breaks temporal correlations between consecutive steps and stabilizes learning.
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        if not mini_sample:
            return 0.0

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        return self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Trains the model immediately on the single most recent step."""
        return self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, test_mode=False):
        """
        Epsilon-greedy action selection:
        - In early games: Explores by taking random actions.
        - As games increase: Exploits learned Q-values predicted by the neural network.
        - If test_mode=True: Randomness is turned OFF (pure policy evaluation).
        """
        final_move = [0, 0, 0]

        # Calculate network predicted Q-values
        state0 = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            prediction = self.model(state0)
            self.last_q_values = prediction.cpu().numpy().tolist()

        if test_mode:
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            return final_move

        # Epsilon decay: Starts at 1.0 (100% exploration), smoothly decays to 0.02 (2% exploration)
        self.epsilon = max(0.02, 1.0 - (self.n_games / 100.0))

        if random.random() < self.epsilon:
            # Explore: Choose a random relative turn
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploit: Choose the action with highest predicted Q-value
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def save_model(self, file_name='model.pth'):
        """Saves current network weights."""
        self.model.save(file_name)

    def load_model(self, file_path):
        """Loads network weights from disk."""
        return self.model.load(file_path)


# -------------------------------------------------------------
# Unit test for Agent
# -------------------------------------------------------------
if __name__ == '__main__':
    print("Testing Agent initialization and state generation...")
    agent = Agent()
    game = SnakeGameAI(render_mode=False)
    state = agent.get_state(game)

    print(f"State shape: {state.shape} (Expected 11)")
    print(f"Initial state vector: {state}")
    assert len(state) == 11, "State vector length must be 11"

    # Test action selection
    action = agent.get_action(state)
    print(f"Sample action generated: {action} (sum={sum(action)})")
    assert sum(action) == 1, "Action must be a valid one-hot vector"

    # Test short and long memory training
    agent.remember(state, action, 0, state, False)
    agent.train_short_memory(state, action, 0, state, False)
    agent.train_long_memory()

    print("Agent unit test passed successfully!")
