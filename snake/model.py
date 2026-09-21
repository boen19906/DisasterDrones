import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class Linear_QNet(nn.Module):
    """
    Deep Q-Network (Function Approximator).
    Maps the 11-dimensional state vector to 3 Q-values corresponding
    to the actions: [Straight, Turn Right, Turn Left].
    """

    def __init__(self, input_size=11, hidden_size=256, output_size=3):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the neural network."""
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)  # Raw Q-values (no activation at output)
        return x

    def save(self, file_name='model.pth', model_folder_path='./models'):
        """Saves network weights to disk."""
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to: {file_path}")

    def load(self, file_path):
        """Loads network weights from disk."""
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path, weights_only=True))
            self.eval()
            print(f"Successfully loaded model from: {file_path}")
            return True
        else:
            print(f"Error: Model file not found at: {file_path}")
            return False


class QTrainer:
    """
    Optimizes the Deep Q-Network using the Bellman Equation and Mean Squared Error.
    Supports both single-step transitions and random mini-batches from replay memory.
    """

    def __init__(self, model, lr=0.001, gamma=0.9):
        self.model = model
        self.lr = lr
        self.gamma = gamma  # Discount factor for future rewards (0.9 means future matters)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        Executes one optimization step using the Bellman Equation:
            Q_target = R + gamma * max(Q(next_state))   (if not done)
            Q_target = R                                (if done)
        """
        # Convert numpy arrays / lists to PyTorch Tensors
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        # If training on a single step (short-term memory), add batch dimension: shape (1, X)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1. Current predicted Q values: Q(s) -> shape (batch_size, 3)
        pred = self.model(state)

        # 2. Clone predicted values to create the target tensor
        target = pred.clone()

        # 3. Update target for the action taken using the Bellman equation
        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                # Bellman Equation: Q_new = reward + gamma * max(Q(s'))
                next_q_values = self.model(next_state[idx])
                q_new = reward[idx] + self.gamma * torch.max(next_q_values)

            # Locate which action was taken: [1,0,0]=0, [0,1,0]=1, [0,0,1]=2
            action_idx = torch.argmax(action[idx]).item()
            target[idx][action_idx] = q_new

        # 4. Compute Loss and Backpropagate
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

        return loss.item()


# -------------------------------------------------------------
# Unit test for Linear_QNet and QTrainer
# -------------------------------------------------------------
if __name__ == '__main__':
    print("Testing Linear_QNet and QTrainer...")

    model = Linear_QNet(input_size=11, hidden_size=256, output_size=3)
    trainer = QTrainer(model, lr=0.001, gamma=0.9)

    # Fake transition (state, action, reward, next_state, done)
    dummy_state = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    dummy_action = [0, 1, 0]
    dummy_reward = 10
    dummy_next_state = [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]
    dummy_done = False

    loss = trainer.train_step(dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done)
    print(f"Single step training test passed! Loss: {loss:.5f}")

    # Test saving & loading
    model.save("test_model.pth")
    test_load = Linear_QNet()
    assert test_load.load("./models/test_model.pth")
    print("Model save & load test passed!")
    
    # Cleanup test artifact
    if os.path.exists("./models/test_model.pth"):
        os.remove("./models/test_model.pth")
    print("All DQN tests passed successfully!")
