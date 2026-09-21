import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Initialize Pygame
pygame.init()
pygame.font.init()

# Type definitions
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Constants & Visual Palette
BLOCK_SIZE = 20
COLOR_BG_DARK = (15, 20, 30)
COLOR_BG_GRID = (22, 30, 46)
COLOR_DASHBOARD_BG = (17, 23, 38)
COLOR_CARD_BG = (25, 34, 54)
COLOR_CARD_BORDER = (40, 54, 82)
COLOR_SNAKE_HEAD = (0, 240, 160)
COLOR_SNAKE_BODY_1 = (0, 200, 140)
COLOR_SNAKE_BODY_2 = (0, 160, 120)
COLOR_FOOD = (255, 60, 80)
COLOR_FOOD_GLOW = (255, 120, 140)
COLOR_TEXT = (240, 245, 255)
COLOR_TEXT_DIM = (140, 155, 180)
COLOR_ACCENT = (0, 215, 255)
COLOR_GOLD = (255, 200, 40)
COLOR_BAR_BG = (35, 45, 65)
COLOR_BAR_FILL = (0, 200, 255)
COLOR_BAR_WARN = (255, 160, 0)
COLOR_BAR_DANGER = (255, 50, 70)


class SnakeGameAI:
    """
    Snake Game Environment tailored for Reinforcement Learning (DQN).
    Features a unified widescreen interface with an integrated Real-Time Learning Dashboard:
    - Game Grid (640x480)
    - Live Analytics & Dynamic Performance Curve (360x480)
    - Live Q-Value Brain Activity visualization
    """

    def __init__(self, w=640, h=480, render_mode=True, speed=60, show_dashboard=True):
        self.w = w
        self.h = h
        self.render_mode = render_mode
        self.speed = speed
        self.is_turbo = False
        self.user_exit = False
        self.show_dashboard = show_dashboard
        self.dashboard_w = 360 if show_dashboard else 0
        self.total_w = self.w + self.dashboard_w
        self.total_h = self.h

        # Fonts
        self.font_title = pygame.font.SysFont('Arial', 16, bold=True)
        self.font_large = pygame.font.SysFont('Arial', 20, bold=True)
        self.font_mid = pygame.font.SysFont('Arial', 13, bold=True)
        self.font_small = pygame.font.SysFont('Arial', 11)

        if self.render_mode:
            self.display = pygame.display.set_mode((self.total_w, self.total_h))
            pygame.display.set_caption("RL Snake AI - Autonomous Training Environment & Live Dashboard")
            self.clock = pygame.time.Clock()
        else:
            self.display = None
            self.clock = None

        self.reset()

    def reset(self):
        """Resets the game state to start a new episode."""
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        return self.get_state_snapshot()

    def _place_food(self):
        """Randomly generates food in a grid cell not occupied by the snake."""
        max_x = (self.w - BLOCK_SIZE) // BLOCK_SIZE
        max_y = (self.h - BLOCK_SIZE) // BLOCK_SIZE
        while True:
            x = random.randint(0, max_x) * BLOCK_SIZE
            y = random.randint(0, max_y) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def get_state_snapshot(self):
        """Returns minimal state info useful for logging or debugging."""
        return {
            'head': self.head,
            'direction': self.direction,
            'score': self.score,
            'snake_len': len(self.snake),
            'food': self.food
        }

    def play_step(self, action, episode=None, high_score=None, epsilon=None,
                  mean_score=None, scores_history=None, mean_scores_history=None,
                  q_values=None, stage_banner=None):
        """
        Executes one step in the environment.
        :param action: [straight, right, left] (e.g. [1, 0, 0])
        :return: (reward, game_over, score)
        """
        self.frame_iteration += 1

        # 1. Handle user inputs and keyboard shortcuts (Space to toggle Turbo, ESC to quit)
        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.user_exit = True
                    return -10, True, self.score
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.is_turbo = not self.is_turbo
                    elif event.key == pygame.K_UP:
                        self.speed = min(300, self.speed + 10)
                    elif event.key == pygame.K_DOWN:
                        self.speed = max(10, self.speed - 10)
                    elif event.key == pygame.K_ESCAPE:
                        self.user_exit = True
                        return -10, True, self.score

        # Measure Manhattan distance to food before moving
        old_dist = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

        # 2. Update snake direction based on relative action
        self._move(action)
        self.snake.insert(0, self.head)

        # Measure Manhattan distance after moving
        new_dist = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

        # 3. Check collision or starvation (taking too long without food)
        reward = 0
        game_over = False
        starvation_limit = max(80, 50 * len(self.snake))

        if self.is_collision() or self.frame_iteration > starvation_limit:
            game_over = True
            reward = -15
            return reward, game_over, self.score

        # 4. Check if food eaten
        if self.head == self.food:
            self.score += 1
            reward = 15
            self._place_food()
            self.frame_iteration = 0  # reset starvation timer
        else:
            self.snake.pop()
            # Reward shaping: encourage moving closer, penalize moving away or looping
            if new_dist < old_dist:
                reward = 1.0   # Closer to apple
            else:
                reward = -1.5  # Moving away or circling

        # 5. Visual Render
        if self.render_mode:
            self._update_ui(
                episode=episode,
                high_score=high_score,
                epsilon=epsilon,
                mean_score=mean_score,
                scores_history=scores_history,
                mean_scores_history=mean_scores_history,
                q_values=q_values,
                action=action,
                starvation_limit=starvation_limit
            )
            fps = 0 if self.is_turbo else self.speed
            self.clock.tick(fps)

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """Checks if a point collides with walls or the snake's own body."""
        if pt is None:
            pt = self.head

        # Hits wall
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # Hits body (excluding the head itself)
        if pt in self.snake[1:]:
            return True

        return False

    def _move(self, action):
        """
        Translates a relative action [straight, right, left] into cardinal direction.
        Order: [RIGHT, DOWN, LEFT, UP] (Clockwise)
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # No change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Turn Right
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Turn Left

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def _draw_grid(self):
        """Draws subtle background grid lines for clean spatial visual reference."""
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, COLOR_BG_GRID, (x, 0), (x, self.h), 1)
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, COLOR_BG_GRID, (0, y), (self.w, y), 1)

    def _draw_dashboard(self, episode, high_score, epsilon, mean_score,
                        scores_history, mean_scores_history, q_values, action):
        """Renders the right-hand Real-Time Analytics Dashboard and Learning Curve."""
        dx = self.w
        dw = self.dashboard_w
        dh = self.h

        # Panel Background & Separator Line
        pygame.draw.rect(self.display, COLOR_DASHBOARD_BG, pygame.Rect(dx, 0, dw, dh))
        pygame.draw.line(self.display, (35, 48, 75), (dx, 0), (dx, dh), 2)

        # Header Title
        title_surf = self.font_title.render("AI TRAINING DASHBOARD", True, COLOR_ACCENT)
        self.display.blit(title_surf, (dx + 18, 12))
        sub_surf = self.font_small.render("Deep Q-Learning Autonomous Agent", True, COLOR_TEXT_DIM)
        self.display.blit(sub_surf, (dx + 18, 32))

        # 1. Top Metrics Grid (3 mini cards)
        card1 = pygame.Rect(dx + 15, 54, 100, 52)
        card2 = pygame.Rect(dx + 125, 54, 105, 52)
        card3 = pygame.Rect(dx + 240, 54, 105, 52)

        for card in [card1, card2, card3]:
            pygame.draw.rect(self.display, COLOR_CARD_BG, card, border_radius=6)
            pygame.draw.rect(self.display, COLOR_CARD_BORDER, card, width=1, border_radius=6)

        # Card 1: Episode & Epsilon
        ep_val = episode if episode is not None else 0
        eps_val = max(0, int(epsilon)) if epsilon is not None else 0
        self.display.blit(self.font_small.render("EPISODE", True, COLOR_TEXT_DIM), (dx + 23, 60))
        self.display.blit(self.font_large.render(str(ep_val), True, COLOR_TEXT), (dx + 23, 76))

        # Card 2: Score & Record
        hi_val = high_score if high_score is not None else 0
        self.display.blit(self.font_small.render("SCORE / HIGH", True, COLOR_TEXT_DIM), (dx + 133, 60))
        score_str = f"{self.score} / {hi_val}"
        self.display.blit(self.font_large.render(score_str, True, COLOR_GOLD), (dx + 133, 76))

        # Card 3: Rolling Average & Epsilon
        avg_val = f"{mean_score:.1f}" if mean_score is not None else "0.0"
        self.display.blit(self.font_small.render("MEAN SCORE", True, COLOR_TEXT_DIM), (dx + 248, 60))
        self.display.blit(self.font_large.render(avg_val, True, (0, 230, 160)), (dx + 248, 76))

        # Epsilon (Exploration) Banner
        if epsilon is not None:
            eps_pct = int(epsilon * 100) if epsilon <= 1.0 else int(epsilon)
        else:
            eps_pct = 0
        eps_ratio = min(1.0, max(0.0, eps_pct / 100.0))
        eps_banner = pygame.Rect(dx + 15, 114, dw - 30, 22)
        pygame.draw.rect(self.display, (25, 35, 55), eps_banner, border_radius=4)
        pygame.draw.rect(self.display, (60, 100, 180), pygame.Rect(dx + 15, 114, int((dw - 30) * eps_ratio), 22), border_radius=4)
        eps_label = self.font_small.render(f"Exploration Rate (Randomness): {eps_pct}%", True, COLOR_TEXT)
        self.display.blit(eps_label, (dx + 24, 118))

        # 2. Neural Network Brain Activity (Live Q-Values)
        self.display.blit(self.font_mid.render("MODEL BRAIN ACTIVITY (Q-Values)", True, COLOR_TEXT), (dx + 18, 146))
        actions_info = [("Straight", 0), ("Turn Right", 1), ("Turn Left", 2)]

        vals = q_values if (q_values and len(q_values) == 3) else [0.0, 0.0, 0.0]
        max_q = max(abs(min(vals)), abs(max(vals)), 5.0)

        for i, (name, idx) in enumerate(actions_info):
            row_y = 168 + (i * 28)
            q_val = vals[idx]
            is_chosen = (action is not None and len(action) == 3 and action[idx] == 1)

            # Label
            lbl_color = COLOR_ACCENT if is_chosen else COLOR_TEXT_DIM
            self.display.blit(self.font_small.render(name, True, lbl_color), (dx + 18, row_y + 4))

            # Bar background
            bar_rect = pygame.Rect(dx + 90, row_y + 3, 175, 16)
            pygame.draw.rect(self.display, (25, 34, 52), bar_rect, border_radius=3)

            # Bar fill based on normalized Q value
            fill_pct = max(0.05, min(1.0, (q_val + max_q) / (2.0 * max_q)))
            fill_w = int(175 * fill_pct)
            fill_color = (0, 230, 160) if is_chosen else (70, 95, 140)
            pygame.draw.rect(self.display, fill_color, pygame.Rect(dx + 90, row_y + 3, fill_w, 16), border_radius=3)

            if is_chosen:
                pygame.draw.rect(self.display, (0, 255, 180), bar_rect, width=1, border_radius=3)

            # Number text
            val_text = self.font_small.render(f"{q_val:+.2f}", True, COLOR_TEXT)
            self.display.blit(val_text, (dx + 275, row_y + 4))

        # 3. Real-Time Learning Curve Plot
        self.display.blit(self.font_mid.render("REAL-TIME LEARNING CURVE", True, COLOR_TEXT), (dx + 18, 258))

        chart_box = pygame.Rect(dx + 15, 278, dw - 30, 145)
        pygame.draw.rect(self.display, (18, 26, 44), chart_box, border_radius=6)
        pygame.draw.rect(self.display, (35, 50, 78), chart_box, width=1, border_radius=6)

        # Plot axes & grid lines
        for gy in [0.25, 0.5, 0.75]:
            line_y = chart_box.bottom - int(chart_box.height * gy)
            pygame.draw.line(self.display, (28, 38, 60), (chart_box.left + 5, line_y), (chart_box.right - 5, line_y), 1)

        # Draw curves if history available
        if scores_history and len(scores_history) > 1:
            max_y = max(max(scores_history), 5)
            n_pts = len(scores_history)

            # Plot raw scores
            score_pts = []
            for i, sc in enumerate(scores_history):
                px = chart_box.left + 8 + int((i / max(1, n_pts - 1)) * (chart_box.width - 16))
                py = chart_box.bottom - 8 - int((sc / max_y) * (chart_box.height - 20))
                score_pts.append((px, py))
            if len(score_pts) > 1:
                pygame.draw.lines(self.display, (0, 230, 160), False, score_pts, 1)

            # Plot rolling mean
            if mean_scores_history and len(mean_scores_history) > 1:
                mean_pts = []
                for i, mn in enumerate(mean_scores_history):
                    px = chart_box.left + 8 + int((i / max(1, n_pts - 1)) * (chart_box.width - 16))
                    py = chart_box.bottom - 8 - int((mn / max_y) * (chart_box.height - 20))
                    mean_pts.append((px, py))
                if len(mean_pts) > 1:
                    pygame.draw.lines(self.display, (255, 175, 40), False, mean_pts, 2)

        # Chart Legend
        pygame.draw.circle(self.display, (0, 230, 160), (dx + 25, 410), 3)
        self.display.blit(self.font_small.render("Score", True, COLOR_TEXT_DIM), (dx + 33, 403))
        pygame.draw.circle(self.display, (255, 175, 40), (dx + 90, 410), 3)
        self.display.blit(self.font_small.render("Rolling Mean", True, COLOR_TEXT_DIM), (dx + 98, 403))

        # Bottom Instructions & Speed Mode
        speed_str = "TURBO (SPACE to slow)" if self.is_turbo else f"{self.speed} FPS (SPACE: Turbo)"
        speed_col = (255, 120, 100) if self.is_turbo else (120, 220, 140)
        bot_text = self.font_small.render(speed_str, True, speed_col)
        self.display.blit(bot_text, (dx + 18, dh - 36))

        ctrl_text = self.font_small.render("UP/DN: Speed | ESC: Stop & Save", True, (130, 145, 170))
        self.display.blit(ctrl_text, (dx + 18, dh - 20))

    def _update_ui(self, episode=None, high_score=None, epsilon=None,
                   mean_score=None, scores_history=None, mean_scores_history=None,
                   q_values=None, action=None, starvation_limit=100, stage_banner=None):
        """Renders the game board, food, snake, and integrated dashboard."""
        # 1. Fill game board area
        pygame.draw.rect(self.display, COLOR_BG_DARK, pygame.Rect(0, 0, self.w, self.h))
        self._draw_grid()

        # 2. Draw Food with glowing effect and inner shine
        fx, fy = self.food.x, self.food.y
        pygame.draw.rect(self.display, COLOR_FOOD, pygame.Rect(fx + 2, fy + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4), border_radius=6)
        pygame.draw.circle(self.display, COLOR_FOOD_GLOW, (fx + 7, fy + 7), 3)

        # 3. Draw Snake (Head has eyes indicating direction, body has alternating gradient)
        for i, pt in enumerate(self.snake):
            rect = pygame.Rect(pt.x + 1, pt.y + 1, BLOCK_SIZE - 2, BLOCK_SIZE - 2)
            if i == 0:
                # Snake Head
                pygame.draw.rect(self.display, COLOR_SNAKE_HEAD, rect, border_radius=6)

                # Directional eyes
                eye_color = (20, 30, 45)
                eye_pupil = (255, 255, 255)
                if self.direction == Direction.RIGHT:
                    e1, e2 = (pt.x + 14, pt.y + 5), (pt.x + 14, pt.y + 13)
                elif self.direction == Direction.LEFT:
                    e1, e2 = (pt.x + 5, pt.y + 5), (pt.x + 5, pt.y + 13)
                elif self.direction == Direction.UP:
                    e1, e2 = (pt.x + 5, pt.y + 5), (pt.x + 13, pt.y + 5)
                else:  # DOWN
                    e1, e2 = (pt.x + 5, pt.y + 14), (pt.x + 13, pt.y + 14)

                pygame.draw.circle(self.display, eye_pupil, e1, 3)
                pygame.draw.circle(self.display, eye_pupil, e2, 3)
                pygame.draw.circle(self.display, eye_color, e1, 1)
                pygame.draw.circle(self.display, eye_color, e2, 1)
            else:
                # Body segments with alternating color
                color = COLOR_SNAKE_BODY_1 if (i % 2 == 0) else COLOR_SNAKE_BODY_2
                pygame.draw.rect(self.display, color, rect, border_radius=4)

        # 4. In-Game Top HUD (Score & Mode)
        score_badge = self.font_large.render(f"Score: {self.score}", True, COLOR_TEXT)
        self.display.blit(score_badge, (15, 10))

        # 5. Starvation / Hunger meter (Bottom Bar of game grid)
        hunger_ratio = min(1.0, self.frame_iteration / max(1, starvation_limit))
        bar_w = self.w - 30
        bar_h = 4
        bar_x = 15
        bar_y = self.h - 10
        pygame.draw.rect(self.display, COLOR_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=2)
        fill_color = COLOR_BAR_FILL if hunger_ratio < 0.6 else (COLOR_BAR_WARN if hunger_ratio < 0.85 else COLOR_BAR_DANGER)
        pygame.draw.rect(self.display, fill_color, (bar_x, bar_y, int(bar_w * (1.0 - hunger_ratio)), bar_h), border_radius=2)

        # 6. Stage Banner Overlay (used during visual showcase demos)
        if stage_banner:
            title, subtitle, banner_color = stage_banner
            banner_surf = pygame.Surface((self.w, 42), pygame.SRCALPHA)
            banner_surf.fill((10, 15, 25, 230))
            self.display.blit(banner_surf, (0, self.h - 48))
            self.display.blit(self.font_mid.render(f"{title} - {subtitle}", True, banner_color), (15, self.h - 44))
            self.display.blit(self.font_small.render("Keys: [1] Dumb  [2] Novice  [3] Master  [Space] Turbo  [ESC] Quit", True, (160, 175, 200)), (15, self.h - 22))

        # 7. Draw Dashboard on the right panel
        if self.show_dashboard:
            self._draw_dashboard(
                episode, high_score, epsilon, mean_score,
                scores_history, mean_scores_history, q_values, action
            )

        pygame.display.flip()


# -------------------------------------------------------------
# Manual Human Play Mode
# -------------------------------------------------------------
if __name__ == '__main__':
    print("Starting Snake Game in Human Play Mode...")
    print("Controls: Arrow Keys or WASD to steer. Space to toggle turbo. ESC to quit.")
    game = SnakeGameAI(render_mode=True, speed=12, show_dashboard=False)

    while True:
        action = [1, 0, 0]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    if game.direction == Direction.UP:
                        action = [0, 0, 1]
                    elif game.direction == Direction.DOWN:
                        action = [0, 1, 0]
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    if game.direction == Direction.UP:
                        action = [0, 1, 0]
                    elif game.direction == Direction.DOWN:
                        action = [0, 0, 1]
                elif event.key in (pygame.K_UP, pygame.K_w):
                    if game.direction == Direction.LEFT:
                        action = [0, 1, 0]
                    elif game.direction == Direction.RIGHT:
                        action = [0, 0, 1]
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    if game.direction == Direction.LEFT:
                        action = [0, 0, 1]
                    elif game.direction == Direction.RIGHT:
                        action = [0, 1, 0]
                elif event.key == pygame.K_SPACE:
                    game.is_turbo = not game.is_turbo
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()

        reward, done, score = game.play_step(action)
        if done:
            print(f"Game Over! Final Score: {score}")
            game.reset()
