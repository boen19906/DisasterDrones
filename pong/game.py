import pygame
import random
import numpy as np

# Initialize Pygame
pygame.init()
pygame.font.init()

# Constants
PADDLE_WIDTH = 15
PADDLE_HEIGHT = 80
BALL_SIZE = 15

COLOR_BG = (15, 20, 30)
COLOR_PADDLE = (0, 240, 160)
COLOR_OPPONENT = (255, 60, 80)
COLOR_BALL = (255, 200, 40)
COLOR_TEXT = (240, 245, 255)
COLOR_DASHBOARD_BG = (17, 23, 38)
COLOR_ACCENT = (0, 215, 255)

class PongGameAI:
    def __init__(self, w=640, h=480, render_mode=True, speed=60, show_dashboard=True, human_playing=False):
        self.w = w
        self.h = h
        self.render_mode = render_mode
        self.speed = speed
        self.show_dashboard = show_dashboard
        self.human_playing = human_playing
        self.max_score = 5
        self.dashboard_w = 360 if show_dashboard else 0
        self.total_w = self.w + self.dashboard_w
        self.total_h = self.h
        self.is_turbo = False
        self.user_exit = False
        
        self.font = pygame.font.SysFont('Arial', 20, bold=True)
        self.font_small = pygame.font.SysFont('Arial', 12)

        if self.render_mode:
            self.display = pygame.display.set_mode((self.total_w, self.total_h))
            pygame.display.set_caption("RL Pong AI")
            self.clock = pygame.time.Clock()
        
        self.reset()

    def reset(self):
        self.player_y = self.h // 2 - PADDLE_HEIGHT // 2
        self.opponent_y = self.h // 2 - PADDLE_HEIGHT // 2
        
        self.player_score = 0
        self.opponent_score = 0
        
        self._reset_ball()

        self.volleys = 0
        self.frame_iteration = 0
        return self.get_state_snapshot()

    def get_state_snapshot(self):
        return {
            'player_y': self.player_y,
            'opponent_y': self.opponent_y,
            'ball': (self.ball_x, self.ball_y),
            'score': self.player_score
        }

    def play_step(self, action, episode=None, high_score=None, epsilon=None,
                  mean_score=None, scores_history=None, mean_scores_history=None,
                  q_values=None):
        self.frame_iteration += 1

        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.user_exit = True
                    return -10, True, self.player_score
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.is_turbo = not self.is_turbo
                    elif event.key == pygame.K_ESCAPE:
                        self.user_exit = True
                        return -10, True, self.player_score

        # Measure distance before moving
        old_dist = abs((self.player_y + PADDLE_HEIGHT // 2) - self.ball_y)

        # Action: [Up, Stay, Down]
        if np.array_equal(action, [1, 0, 0]):
            self.player_y -= 8
        elif np.array_equal(action, [0, 0, 1]):
            self.player_y += 8
        
        # Clamp player paddle
        self.player_y = max(0, min(self.h - PADDLE_HEIGHT, self.player_y))

        # Measure distance after moving
        new_dist = abs((self.player_y + PADDLE_HEIGHT // 2) - self.ball_y)

        if self.human_playing:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.opponent_y -= 8
            if keys[pygame.K_DOWN]:
                self.opponent_y += 8
        else:
            # Opponent AI (simple tracking with limited speed)
            target_y = self.ball_y - PADDLE_HEIGHT // 2
            if self.opponent_y < target_y:
                self.opponent_y += 4
            elif self.opponent_y > target_y:
                self.opponent_y -= 4
        self.opponent_y = max(0, min(self.h - PADDLE_HEIGHT, self.opponent_y))

        # Update ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Ball collision with top/bottom walls
        if self.ball_y <= 0 or self.ball_y >= self.h - BALL_SIZE:
            self.ball_vy *= -1

        reward = 0
        game_over = False

        # Dense Reward Shaping: removed, reward based solely on volleys/score

        # Ball collision with player paddle
        if (self.ball_x <= 20 + PADDLE_WIDTH and
            self.player_y < self.ball_y + BALL_SIZE and
            self.player_y + PADDLE_HEIGHT > self.ball_y):
            self.ball_vx *= -1
            self.ball_x = 20 + PADDLE_WIDTH  # Fix sticking
            reward = 5  # Strong reward for hitting the ball
            self.volleys += 1
            # Increase speed slightly to make it harder over time
            self.ball_vx += 0.5 if self.ball_vx > 0 else -0.5

        # Ball collision with opponent paddle
        elif (self.ball_x >= self.w - 20 - PADDLE_WIDTH - BALL_SIZE and
              self.opponent_y < self.ball_y + BALL_SIZE and
              self.opponent_y + PADDLE_HEIGHT > self.ball_y):
            self.ball_vx *= -1
            self.ball_x = self.w - 20 - PADDLE_WIDTH - BALL_SIZE

        # Ball out of bounds (Player misses)
        if self.ball_x < 0:
            reward = -10
            self.opponent_score += 1
            if self.opponent_score >= self.max_score:
                game_over = True
                return reward, game_over, self.player_score
            self._reset_ball()

        # Ball out of bounds (Opponent misses)
        elif self.ball_x > self.w:
            reward = 10
            self.player_score += 1
            if self.player_score >= self.max_score:
                game_over = True
                return reward, game_over, self.player_score
            self._reset_ball()

        # Render
        if self.render_mode:
            self._update_ui(episode, high_score, epsilon, mean_score, q_values)
            fps = 0 if self.is_turbo else self.speed
            self.clock.tick(fps)

        return reward, game_over, self.player_score

    def _reset_ball(self):
        self.ball_x = self.w // 2
        self.ball_y = self.h // 2
        self.ball_vx = random.choice([-5, 5])
        self.ball_vy = random.choice([-5, 5])

    def _update_ui(self, episode, high_score, epsilon, mean_score, q_values):
        self.display.fill(COLOR_BG)
        
        # Center line
        pygame.draw.line(self.display, (50, 60, 80), (self.w // 2, 0), (self.w // 2, self.h), 2)
        
        # Paddles
        pygame.draw.rect(self.display, COLOR_PADDLE, (20, self.player_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(self.display, COLOR_OPPONENT, (self.w - 20 - PADDLE_WIDTH, self.opponent_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        
        # Ball
        pygame.draw.ellipse(self.display, COLOR_BALL, (self.ball_x, self.ball_y, BALL_SIZE, BALL_SIZE))
        
        # Score
        score_text = self.font.render(f"Score: Player {self.player_score} - {self.opponent_score} Opponent  |  Volleys: {self.volleys}", True, COLOR_TEXT)
        self.display.blit(score_text, (20, 20))

        if self.show_dashboard:
            self._draw_dashboard(episode, high_score, epsilon, mean_score, q_values)

        pygame.display.flip()

    def _draw_dashboard(self, episode, high_score, epsilon, mean_score, q_values):
        dx = self.w
        dw = self.dashboard_w
        dh = self.h
        
        pygame.draw.rect(self.display, COLOR_DASHBOARD_BG, pygame.Rect(dx, 0, dw, dh))
        pygame.draw.line(self.display, (35, 48, 75), (dx, 0), (dx, dh), 2)
        
        # Basic dashboard elements
        self.display.blit(self.font.render("PONG AI TRAINING", True, COLOR_ACCENT), (dx + 20, 20))
        
        ep = episode if episode is not None else 0
        eps = int(epsilon * 100) if epsilon is not None else 0
        hi = high_score if high_score is not None else 0
        mn = f"{mean_score:.2f}" if mean_score is not None else "0.00"
        
        self.display.blit(self.font_small.render(f"Episode: {ep}", True, COLOR_TEXT), (dx + 20, 60))
        self.display.blit(self.font_small.render(f"Epsilon: {eps}%", True, COLOR_TEXT), (dx + 20, 80))
        self.display.blit(self.font_small.render(f"High Score: {hi}", True, COLOR_TEXT), (dx + 20, 100))
        self.display.blit(self.font_small.render(f"Mean Score: {mn}", True, COLOR_TEXT), (dx + 20, 120))
        
        # Q-Values
        if q_values:
            self.display.blit(self.font_small.render("Q-Values [Up, Stay, Down]:", True, COLOR_ACCENT), (dx + 20, 160))
            self.display.blit(self.font_small.render(f"Up:   {q_values[0]:.2f}", True, COLOR_TEXT), (dx + 20, 180))
            self.display.blit(self.font_small.render(f"Stay: {q_values[1]:.2f}", True, COLOR_TEXT), (dx + 20, 200))
            self.display.blit(self.font_small.render(f"Down: {q_values[2]:.2f}", True, COLOR_TEXT), (dx + 20, 220))
