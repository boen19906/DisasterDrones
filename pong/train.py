import sys
import argparse
import os
import pygame
from agent import Agent
from game import PongGameAI
from plotter import SafePlotter

def train(target_episodes=200, speed=60, headless=False):
    print("=" * 68)
    print("  Deep Q-Learning Pong AI - Autonomous Training Pipeline")
    print("=" * 68)

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    mean_score = 0.0

    agent = Agent()
    game = PongGameAI(render_mode=not headless, speed=speed, show_dashboard=not headless)

    if not os.path.exists("./models"):
        os.makedirs("./models")

    try:
        while agent.n_games < target_episodes:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)

            reward, done, score = game.play_step(
                action=final_move,
                episode=agent.n_games + 1,
                high_score=record,
                epsilon=agent.epsilon,
                mean_score=mean_score,
                q_values=agent.last_q_values
            )
            state_new = agent.get_state(game)

            if game.user_exit:
                print("\nExit requested by user.")
                break

            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                is_new_record = False
                if score > record:
                    record = score
                    agent.save_model("model_best.pth")
                    is_new_record = True

                if agent.n_games in (1, 10, 25, 50, 75, 100, 150, 200):
                    agent.save_model(f"model_ep{agent.n_games}.pth")
                    print(f">>> [Checkpoint Saved] models/model_ep{agent.n_games}.pth")

                agent.save_model("model_latest.pth")

                total_score += score
                mean_score = total_score / agent.n_games
                plot_scores.append(score)
                plot_mean_scores.append(mean_score)

                rec_flag = " *** NEW HIGH SCORE! ***" if is_new_record else ""
                print(
                    f"Game {agent.n_games:3d} | "
                    f"Score: {score:2d} | "
                    f"High: {record:2d} | "
                    f"Mean: {mean_score:4.1f} | "
                    f"Epsilon: {int(agent.epsilon * 100):2d}%"
                    f"{rec_flag}",
                    flush=True
                )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")

    finally:
        print("\n" + "=" * 68)
        print("  Training Session Completed / Stopped")
        print("=" * 68)
        print(f"Total Games Played: {agent.n_games}")
        print(f"All-Time Record:    {record}")

        if plot_scores:
            import json
            with open("training_history.json", "w") as f:
                json.dump({"scores": plot_scores, "means": plot_mean_scores, "record": record}, f)
            plotter = SafePlotter()
            plotter.save_plot(plot_scores, plot_mean_scores, "training_curve.png")

        if game.render_mode and pygame.get_init():
            pygame.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--speed', type=int, default=60)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    train(args.episodes, args.speed, args.headless)
