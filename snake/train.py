import sys
import argparse
import os
import pygame
from agent import Agent
from game import SnakeGameAI
from plotter import SafePlotter


def train(target_episodes=200, speed=60, headless=False):
    """
    Main Reinforcement Learning training loop:
    - Runs the unified Snake environment and Live Learning Dashboard.
    - Observes states, trains short-term memory and long-term experience replay.
    - Saves milestone models (Episode 1: Dumb, Episode 50: Novice, Best: Master).
    - Saves high-res training_curve.png on completion.
    """
    print("=" * 68)
    print("  Deep Q-Learning Snake AI - Autonomous Training Pipeline")
    print("=" * 68)
    print(f"Target Episodes: {target_episodes}")
    print(f"Initial Speed:   {speed} FPS (Press SPACE in game window to toggle TURBO)")
    print(f"Display Mode:    {'Headless (Background)' if headless else 'Interactive Unified Dashboard'}")
    print("=" * 68)

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    mean_score = 0.0

    agent = Agent()
    game = SnakeGameAI(render_mode=not headless, speed=speed, show_dashboard=not headless)

    # Ensure models directory exists
    if not os.path.exists("./models"):
        os.makedirs("./models")

    # Clean existing models if starting fresh? We can keep them or let them overwrite cleanly
    try:
        while agent.n_games < target_episodes:
            # 1. Get current state
            state_old = agent.get_state(game)

            # 2. Get action from agent (also records agent.last_q_values)
            final_move = agent.get_action(state_old)

            # 3. Step environment with full live dashboard telemetry
            reward, done, score = game.play_step(
                action=final_move,
                episode=agent.n_games + 1,
                high_score=record,
                epsilon=agent.epsilon,
                mean_score=mean_score,
                scores_history=plot_scores,
                mean_scores_history=plot_mean_scores,
                q_values=agent.last_q_values
            )
            state_new = agent.get_state(game)

            # Check if user closed window or pressed ESC during play_step
            if game.user_exit:
                print("\nExit requested by user.")
                break

            # 4. Train short-term memory on this single transition
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # 5. Store transition in experience replay memory
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # Reset game environment for next episode
                game.reset()
                agent.n_games += 1

                # Train long-term memory on batch sampled from replay buffer
                agent.train_long_memory()

                # Update High Score and Milestone Checkpoints
                is_new_record = False
                if score > record:
                    record = score
                    agent.save_model("model_best.pth")
                    is_new_record = True

                # Save milestone checkpoints across training timeline
                if agent.n_games in (1, 10, 25, 50, 75, 100, 150, 200):
                    agent.save_model(f"model_ep{agent.n_games}.pth")
                    print(f">>> [Checkpoint Saved] models/model_ep{agent.n_games}.pth")

                # Always save latest
                agent.save_model("model_latest.pth")

                # Track statistics
                total_score += score
                mean_score = total_score / agent.n_games
                plot_scores.append(score)
                plot_mean_scores.append(mean_score)

                # Log progress
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
        if agent.n_games > 0:
            print(f"Final Mean Score:   {total_score / agent.n_games:.2f}")

        # Generate and save final high-resolution training curve and data
        if plot_scores:
            import json
            with open("training_history.json", "w") as f:
                json.dump({"scores": plot_scores, "means": plot_mean_scores, "record": record}, f)
            plotter = SafePlotter()
            plotter.save_plot(plot_scores, plot_mean_scores, "training_curve.png")

        if game.render_mode and pygame.get_init():
            pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Deep Q-Network on Snake")
    parser.add_argument('--episodes', type=int, default=200, help="Number of training episodes (default: 200)")
    parser.add_argument('--speed', type=int, default=60, help="FPS speed during visualization (default: 60)")
    parser.add_argument('--headless', action='store_true', help="Run without UI window for maximum speed")

    args = parser.parse_args()
    train(
        target_episodes=args.episodes,
        speed=args.speed,
        headless=args.headless
    )
