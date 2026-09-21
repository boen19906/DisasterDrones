import pygame
import os
from game import PongGameAI
from agent import Agent

def play():
    # Initialize the game in human_playing mode
    game = PongGameAI(speed=60, show_dashboard=True, human_playing=True)
    
    agent = Agent()
    model_path = 'models/model_latest.pth'
    
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"Loaded trained model from {model_path}.")
    else:
        print("No trained model found! The agent will act randomly.")

    while True:
        # Get old state
        state_old = agent.get_state(game)
        
        # Get move (test_mode=True disables exploration/epsilon)
        final_move = agent.get_action(state_old, test_mode=True)
        
        # Perform move and get new state
        reward, done, score = game.play_step(final_move, q_values=agent.last_q_values)
        
        if game.user_exit:
            break
            
        if done:
            game.reset()
            
    pygame.quit()

if __name__ == '__main__':
    play()
