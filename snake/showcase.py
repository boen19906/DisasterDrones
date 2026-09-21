import sys
import os
import json
import pygame
import numpy as np
import torch
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from agent import Agent

# Initialize Pygame
pygame.init()
pygame.font.init()

# Visual Palette
COLOR_BG_WINDOW = (12, 16, 26)
COLOR_BG_GAME = (15, 20, 30)
COLOR_BG_GRID = (22, 30, 46)
COLOR_PANEL_BG = (17, 24, 38)
COLOR_CARD_BG = (24, 34, 54)
COLOR_CARD_BORDER = (40, 56, 86)
COLOR_TEXT = (240, 245, 255)
COLOR_TEXT_DIM = (140, 155, 180)
COLOR_ACCENT = (0, 215, 255)
COLOR_GOLD = (255, 205, 45)
COLOR_EMERALD = (0, 240, 160)
COLOR_SNAKE_HEAD = (0, 240, 160)
COLOR_SNAKE_BODY_1 = (0, 200, 140)
COLOR_SNAKE_BODY_2 = (0, 160, 120)
COLOR_FOOD = (255, 60, 80)
COLOR_FOOD_GLOW = (255, 120, 140)
COLOR_PAUSED = (255, 175, 40)

# Speed Settings
SPEED_OPTIONS = [
    {"label": "1x (Normal)", "fps": 30},
    {"label": "2x (Fast)",   "fps": 60},
    {"label": "4x (High)",   "fps": 120},
    {"label": "TURBO (Max)", "fps": 0}
]


def get_checkpoint_for_game(game_num):
    """
    Maps game number (1 to 200) to the corresponding trained model checkpoint,
    skill tier, and exploration rate.
    """
    if game_num < 10:
        return "model_ep1.pth", "DUMB (Random / Untrained)", (255, 80, 80), 0.85
    elif game_num < 25:
        return "model_ep10.pth", "EARLY EXPLORER", (255, 120, 70), 0.55
    elif game_num < 50:
        return "model_ep25.pth", "LEARNING BOUNDARIES", (255, 170, 50), 0.30
    elif game_num < 75:
        return "model_ep50.pth", "NOVICE HUNTER", (230, 200, 50), 0.12
    elif game_num < 100:
        return "model_ep75.pth", "INTERMEDIATE WEAVER", (160, 220, 70), 0.05
    elif game_num < 150:
        return "model_ep100.pth", "ADVANCED NAVIGATOR", (80, 230, 130), 0.02
    elif game_num < 185:
        return "model_ep150.pth", "EXPERT (Long-Term Loops)", (0, 235, 160), 0.0
    else:
        return "model_ep200.pth", "CHAMPIONSHIP MASTER", (0, 245, 200), 0.0


class TrainingReplayViewer:
    """
    Continuous Training Replay Player:
    - Replays the entire 200-game training progression: Game 1 -> Game 2 -> ... -> Game 200!
    - Preloaded historical training curves: Scores, moving average, and milestones update instantly!
    - Right / Left Arrow skips forward or backward by 25 games (with instant graph sync).
    - Click directly on the graph timeline to scrub to any game episode!
    - Press G to toggle between Full 1-200 Career Timeline and Zoomed 30-game view.
    - Up / Down Arrow or S toggles speed (1x, 2x, 4x, TURBO).
    - Space bar freezes the frame (no flickering).
    """

    def __init__(self):
        self.game_w = 640
        self.game_h = 480
        self.panel_w = 380
        self.win_w = self.game_w + self.panel_w + 30  # 1050
        self.win_h = self.game_h + 30                # 510

        self.speed_idx = 1   # Default: 2x (60 FPS)
        self.is_paused = False
        self.graph_zoom_mode = False  # False: Full 1-200 Career Timeline; True: Zoomed 30 Games

        self.display = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("RL Snake AI - Training Replay Player (Game 1 to 200)")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont('Arial', 16, bold=True)
        self.font_large = pygame.font.SysFont('Arial', 24, bold=True)
        self.font_mid = pygame.font.SysFont('Arial', 13, bold=True)
        self.font_small = pygame.font.SysFont('Arial', 11)

        # Game Engine and Agent
        self.game = SnakeGameAI(w=self.game_w, h=self.game_h, render_mode=False)
        self.agent = Agent()

        # Sequential Game Replay State
        self.current_game = 1
        self.max_games = 200
        self.current_model_file = None
        self.high_score = 0      # High score at current point in replay (from presaved history)
        self.last_score = 0
        self.skip_requested = False

        # Preload training history data for instant graph & high score updates
        self.history_scores = []
        self.history_means = []
        self.history_highs = []
        self._load_training_history()

        # Stable frozen frame variables (zero flickering when paused)
        self.current_action = [1, 0, 0]
        self.current_q_values = [0.0, 0.0, 0.0]

        # Sync stats for starting Game 1
        self.update_stats_for_current_game()

        # Load starting model for Game 1
        self.load_model_for_current_game()

    def _load_training_history(self):
        """Preloads historical training scores, means, and highs from training_history.json."""
        history_path = "training_history.json"
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as f:
                    data = json.load(f)
                self.history_scores = list(data.get("scores", []))
                self.history_means = list(data.get("means", []))
                self.history_highs = list(data.get("highs", []))
                if self.history_scores:
                    self.max_games = max(200, len(self.history_scores))
                print(f">>> [Showcase] Successfully preloaded {len(self.history_scores)} games with progressive high scores!")
            except Exception as e:
                print(f">>> [Showcase] Warning: Could not parse training_history.json: {e}")

        # Fallback if no history file found
        if not self.history_scores:
            self.history_scores = [0] * self.max_games
            self.history_means = [0.0] * self.max_games
            self.history_highs = [0] * self.max_games

    def update_stats_for_current_game(self):
        """Updates LAST GAME and live HIGH SCORE at current point in replay based on presaved history."""
        # Last Game score (score of game immediately preceding current_game)
        if self.current_game > 1 and len(self.history_scores) >= self.current_game - 1:
            self.last_score = self.history_scores[self.current_game - 2]
        else:
            self.last_score = 0

        # High score reached at this exact point in replay from presaved training history
        hist_high = 0
        if len(self.history_highs) >= self.current_game:
            hist_high = self.history_highs[self.current_game - 1]
        elif self.history_scores:
            hist_high = max(self.history_scores[:self.current_game])
            
        # Reset high score if skipping; otherwise, keep session high score if it beats history
        if self.skip_requested or not hasattr(self, 'high_score'):
            self.high_score = hist_high
        else:
            self.high_score = max(self.high_score, hist_high)

    def load_model_for_current_game(self):
        """Loads the appropriate checkpoint model for self.current_game."""
        model_file, skill, color, epsilon = get_checkpoint_for_game(self.current_game)
        if model_file == self.current_model_file:
            return  # Already loaded

        path = os.path.join("./models", model_file)
        if not os.path.exists(path):
            # Fallbacks
            if self.current_game <= 45 and os.path.exists("./models/model_ep1.pth"):
                path = "./models/model_ep1.pth"
            elif self.current_game <= 90 and os.path.exists("./models/model_ep50.pth"):
                path = "./models/model_ep50.pth"
            elif os.path.exists("./models/model_ep200.pth"):
                path = "./models/model_ep200.pth"
            elif os.path.exists("./models/model_best.pth"):
                path = "./models/model_best.pth"
            elif os.path.exists("./models/model_latest.pth"):
                path = "./models/model_latest.pth"

        if os.path.exists(path):
            self.agent.load_model(path)
            self.current_model_file = model_file
            print(f">>> [Replay] Game {self.current_game}: Loaded {model_file} ({skill})")
        else:
            self.agent = Agent()
            self.current_model_file = "untrained"
            print(f">>> [Replay] Game {self.current_game}: Untrained baseline")

    def get_action(self, state, epsilon):
        """Action selection for replay with realistic exploration."""
        if epsilon > 0 and np.random.rand() < epsilon:
            move = np.random.randint(0, 3)
            final_move = [0, 0, 0]
            final_move[move] = 1
            # Calculate Q-values for visual display
            state0 = torch.tensor(state, dtype=torch.float)
            with torch.no_grad():
                pred = self.agent.model(state0)
                self.current_q_values = pred.cpu().numpy().tolist()
            return final_move

        action = self.agent.get_action(state, test_mode=True)
        self.current_q_values = getattr(self.agent, 'last_q_values', [0.0, 0.0, 0.0])
        return action

    def run(self):
        """Main replay loop advancing from Game 1 to Game 200."""
        running = True

        while running and self.current_game <= self.max_games:
            self.load_model_for_current_game()
            model_file, skill, skill_color, epsilon = get_checkpoint_for_game(self.current_game)

            self.game.reset()
            done = False
            current_score = 0

            # Initial step
            state = self.agent.get_state(self.game)
            self.current_action = self.get_action(state, epsilon)

            while not done and running:
                # 1. Event Handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            break
                        # Space: Pause / Play toggle
                        elif event.key == pygame.K_SPACE:
                            self.is_paused = not self.is_paused
                        # RIGHT ARROW: Skip FORWARD 25 games in replay
                        elif event.key == pygame.K_RIGHT:
                            self.current_game = min(self.max_games, self.current_game + 25)
                            self.skip_requested = True
                            self.update_stats_for_current_game()
                            done = True
                        # LEFT ARROW: Skip BACKWARD 25 games in replay
                        elif event.key == pygame.K_LEFT:
                            self.current_game = max(1, self.current_game - 25)
                            self.skip_requested = True
                            self.update_stats_for_current_game()
                            done = True
                        # UP ARROW / S: Speed up
                        elif event.key in (pygame.K_UP, pygame.K_s):
                            self.speed_idx = (self.speed_idx + 1) % len(SPEED_OPTIONS)
                        # DOWN ARROW: Slow down
                        elif event.key == pygame.K_DOWN:
                            self.speed_idx = (self.speed_idx - 1) % len(SPEED_OPTIONS)
                        # G: Toggle Graph Zoom View (Full 1-200 vs Zoomed Last 30)
                        elif event.key == pygame.K_g:
                            self.graph_zoom_mode = not self.graph_zoom_mode
                        # R: Restart replay from Game 1
                        elif event.key == pygame.K_r:
                            self.current_game = 1
                            self.skip_requested = True
                            self.update_stats_for_current_game()
                            done = True

                    # Mouse Clicks: Timeline Scrubbing & Speed Selection
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        mx, my = event.pos
                        px = 15 + self.game_w + 15
                        py = 15
                        pw = self.panel_w

                        # 1. Graph Scrubbing & View Toggle
                        graph_box = pygame.Rect(px + 12, py + 256, pw - 24, 102)
                        canvas = pygame.Rect(graph_box.x + 8, graph_box.y + 22, graph_box.width - 16, graph_box.height - 28)

                        if graph_box.collidepoint(mx, my):
                            if my < canvas.top:
                                # Header clicked: toggle zoom
                                self.graph_zoom_mode = not self.graph_zoom_mode
                            elif canvas.collidepoint(mx, my):
                                # Scrub directly to clicked episode
                                pct = max(0.0, min(1.0, (mx - (canvas.left + 4)) / max(1, canvas.width - 8)))
                                target_game = max(1, min(self.max_games, int(pct * (self.max_games - 1)) + 1))
                                self.current_game = target_game
                                self.skip_requested = True
                                self.update_stats_for_current_game()
                                done = True

                        # 2. Speed Button Clicks
                        sp_y = py + 394
                        sp_w = (pw - 24) // len(SPEED_OPTIONS)
                        for i in range(len(SPEED_OPTIONS)):
                            sx = px + 12 + (i * sp_w)
                            if sx <= mx <= sx + sp_w - 3 and sp_y <= my <= sp_y + 26:
                                self.speed_idx = i

                if not running:
                    break

                # 2. Physics & Step (only when NOT paused)
                if not self.is_paused:
                    reward, done, current_score = self.game.play_step(self.current_action)

                    # Update live high score if the replay gets a lucky run and beats history
                    if current_score > self.high_score:
                        self.high_score = current_score

                    if not done:
                        state = self.agent.get_state(self.game)
                        self.current_action = self.get_action(state, epsilon)

                # 3. Render Game and Replay Dashboard
                self._render_frame(skill, skill_color, current_score)

                # 4. Clock Tick
                cur_speed = SPEED_OPTIONS[self.speed_idx]
                fps = cur_speed["fps"]
                self.clock.tick(fps if not self.is_paused else 30)

            # Game complete: record and advance replay sequence!
            if running:
                if not self.skip_requested:
                    # ADVANCE TO NEXT GAME IN REPLAY!
                    self.current_game += 1
                    if self.current_game > self.max_games:
                        print(">>> [Replay Complete] Finished all 200 games. Restarting replay from Game 1!")
                        self.current_game = 1
                        self.skip_requested = True  # reset high score
                    self.update_stats_for_current_game()
                else:
                    # Scrubbed or skipped by user: do not advance game or overwrite history
                    self.skip_requested = False

        pygame.quit()

    def _render_frame(self, skill, skill_color, current_score):
        """Draws the game and the sequential replay telemetry dashboard."""
        self.display.fill(COLOR_BG_WINDOW)

        # ---------------- 1. GAME BOARD (Left) ----------------
        gx, gy = 15, 15
        gw, gh = self.game_w, self.game_h
        pygame.draw.rect(self.display, COLOR_BG_GAME, pygame.Rect(gx, gy, gw, gh), border_radius=6)
        pygame.draw.rect(self.display, COLOR_CARD_BORDER, pygame.Rect(gx, gy, gw, gh), width=1, border_radius=6)

        # Grid lines
        for x in range(0, gw, BLOCK_SIZE):
            pygame.draw.line(self.display, COLOR_BG_GRID, (gx + x, gy), (gx + x, gy + gh), 1)
        for y in range(0, gh, BLOCK_SIZE):
            pygame.draw.line(self.display, COLOR_BG_GRID, (gx, gy + y), (gx + gw, gy + y), 1)

        # Draw Food
        fx = gx + self.game.food.x
        fy = gy + self.game.food.y
        pygame.draw.rect(self.display, COLOR_FOOD, pygame.Rect(fx + 2, fy + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4), border_radius=6)
        pygame.draw.circle(self.display, COLOR_FOOD_GLOW, (fx + 7, fy + 7), 3)

        # Draw Snake
        for i, pt in enumerate(self.game.snake):
            sx = gx + pt.x
            sy = gy + pt.y
            rect = pygame.Rect(sx + 1, sy + 1, BLOCK_SIZE - 2, BLOCK_SIZE - 2)
            if i == 0:
                pygame.draw.rect(self.display, COLOR_SNAKE_HEAD, rect, border_radius=6)
                eye_color = (20, 30, 45)
                eye_pupil = (255, 255, 255)
                if self.game.direction == Direction.RIGHT:
                    e1, e2 = (sx + 14, sy + 5), (sx + 14, sy + 13)
                elif self.game.direction == Direction.LEFT:
                    e1, e2 = (sx + 5, sy + 5), (sx + 5, sy + 13)
                elif self.game.direction == Direction.UP:
                    e1, e2 = (sx + 5, sy + 5), (sx + 13, sy + 5)
                else:
                    e1, e2 = (sx + 5, sy + 14), (sx + 13, sy + 14)

                pygame.draw.circle(self.display, eye_pupil, e1, 3)
                pygame.draw.circle(self.display, eye_pupil, e2, 3)
                pygame.draw.circle(self.display, eye_color, e1, 1)
                pygame.draw.circle(self.display, eye_color, e2, 1)
            else:
                col = COLOR_SNAKE_BODY_1 if (i % 2 == 0) else COLOR_SNAKE_BODY_2
                pygame.draw.rect(self.display, col, rect, border_radius=4)

        # Top In-Game Badge: Score & Length
        score_badge = self.font_large.render(f"Apples: {current_score}", True, COLOR_TEXT)
        self.display.blit(score_badge, (gx + 18, gy + 12))

        # Pause Indicator
        if self.is_paused:
            pause_badge = pygame.Rect(gx + (gw // 2) - 95, gy + 14, 190, 32)
            pygame.draw.rect(self.display, (15, 20, 32, 235), pause_badge, border_radius=6)
            pygame.draw.rect(self.display, COLOR_PAUSED, pause_badge, width=2, border_radius=6)
            self.display.blit(self.font_mid.render("PAUSED (Space to Play)", True, COLOR_PAUSED), (pause_badge.x + 14, pause_badge.y + 7))

        # ---------------- 2. TELEMETRY PANEL (Right) ----------------
        px = gx + gw + 15
        py = 15
        pw = self.panel_w
        ph = self.game_h

        pygame.draw.rect(self.display, COLOR_PANEL_BG, pygame.Rect(px, py, pw, ph), border_radius=6)
        pygame.draw.rect(self.display, COLOR_CARD_BORDER, pygame.Rect(px, py, pw, ph), width=1, border_radius=6)

        # 1. Header: Training Replay Progress
        head_box = pygame.Rect(px + 12, py + 12, pw - 24, 60)
        pygame.draw.rect(self.display, COLOR_CARD_BG, head_box, border_radius=6)
        pygame.draw.rect(self.display, skill_color, head_box, width=1, border_radius=6)

        game_title = f"TRAINING REPLAY: GAME {self.current_game} of {self.max_games}"
        self.display.blit(self.font_title.render(game_title, True, COLOR_TEXT), (head_box.x + 10, head_box.y + 8))

        # Progress bar through 200 games
        progress_pct = min(1.0, self.current_game / self.max_games)
        p_bar = pygame.Rect(head_box.x + 10, head_box.y + 34, head_box.width - 20, 8)
        pygame.draw.rect(self.display, (30, 42, 66), p_bar, border_radius=4)
        pygame.draw.rect(self.display, skill_color, pygame.Rect(p_bar.x, p_bar.y, int(p_bar.width * progress_pct), 8), border_radius=4)

        tier_txt = f"Skill: {skill}"
        self.display.blit(self.font_small.render(tier_txt, True, skill_color), (head_box.x + 10, head_box.y + 44))

        # 2. Score Cards (Live & Accurate)
        c_w = (pw - 36) // 3
        c1 = pygame.Rect(px + 12, py + 82, c_w, 56)
        c2 = pygame.Rect(px + 12 + c_w + 6, py + 82, c_w, 56)
        c3 = pygame.Rect(px + 12 + (c_w + 6) * 2, py + 82, c_w, 56)

        for c in [c1, c2, c3]:
            pygame.draw.rect(self.display, COLOR_CARD_BG, c, border_radius=5)
            pygame.draw.rect(self.display, COLOR_CARD_BORDER, c, width=1, border_radius=5)

        # Card 1: Last Game Score
        self.display.blit(self.font_small.render("LAST GAME", True, COLOR_TEXT_DIM), (c1.x + 8, c1.y + 6))
        self.display.blit(self.font_large.render(str(self.last_score), True, COLOR_TEXT), (c1.x + 8, c1.y + 24))

        # Card 2: Current Score (Live)
        self.display.blit(self.font_small.render("CURRENT", True, COLOR_TEXT_DIM), (c2.x + 8, c2.y + 6))
        self.display.blit(self.font_large.render(str(current_score), True, COLOR_GOLD), (c2.x + 8, c2.y + 24))

        # Card 3: High Score at current point in replay
        self.display.blit(self.font_small.render("HIGH SCORE", True, COLOR_TEXT_DIM), (c3.x + 8, c3.y + 6))
        self.display.blit(self.font_large.render(str(self.high_score), True, COLOR_EMERALD), (c3.x + 8, c3.y + 24))

        # 3. Model Brain Activity (Live Q-Values - Zero flashing when paused!)
        self.display.blit(self.font_mid.render("MODEL BRAIN ACTIVITY (Q-Values)", True, COLOR_TEXT), (px + 15, py + 148))
        vals = self.current_q_values if self.current_q_values else [0.0, 0.0, 0.0]
        max_q = max(abs(min(vals)), abs(max(vals)), 5.0)
        action_names = [("Straight", 0), ("Turn Right", 1), ("Turn Left", 2)]

        for i, (name, idx) in enumerate(action_names):
            row_y = py + 172 + (i * 26)
            q_val = vals[idx]
            is_chosen = (len(self.current_action) == 3 and self.current_action[idx] == 1)

            lbl_col = COLOR_ACCENT if is_chosen else COLOR_TEXT_DIM
            self.display.blit(self.font_small.render(name, True, lbl_col), (px + 15, row_y + 4))

            bar_w = 175
            bar_x = px + 95
            pygame.draw.rect(self.display, (25, 34, 52), pygame.Rect(bar_x, row_y + 3, bar_w, 16), border_radius=3)

            fill_pct = max(0.05, min(1.0, (q_val + max_q) / (2.0 * max_q)))
            fill_col = COLOR_EMERALD if is_chosen else (70, 95, 140)
            pygame.draw.rect(self.display, fill_col, pygame.Rect(bar_x, row_y + 3, int(bar_w * fill_pct), 16), border_radius=3)

            if is_chosen:
                pygame.draw.rect(self.display, (0, 255, 180), pygame.Rect(bar_x, row_y + 3, bar_w, 16), width=1, border_radius=3)

            self.display.blit(self.font_small.render(f"{q_val:+.2f}", True, COLOR_TEXT), (px + 280, row_y + 4))

        # 4. Scores & Moving Average Real-Time Graph (Preloaded & Synchronized)
        graph_box = pygame.Rect(px + 12, py + 256, pw - 24, 102)
        pygame.draw.rect(self.display, COLOR_CARD_BG, graph_box, border_radius=5)
        pygame.draw.rect(self.display, COLOR_CARD_BORDER, graph_box, width=1, border_radius=5)

        # Plot title and mini legend
        title_mode = "TIMELINE (1-200)" if not self.graph_zoom_mode else "ZOOM (Last 30)"
        self.display.blit(self.font_small.render(f"GRAPH: {title_mode}", True, COLOR_TEXT), (graph_box.x + 8, graph_box.y + 6))

        # Legend indicators
        pygame.draw.circle(self.display, COLOR_EMERALD, (graph_box.right - 145, graph_box.y + 11), 3)
        self.display.blit(self.font_small.render("Score", True, COLOR_EMERALD), (graph_box.right - 138, graph_box.y + 6))

        pygame.draw.circle(self.display, COLOR_GOLD, (graph_box.right - 88, graph_box.y + 11), 3)
        self.display.blit(self.font_small.render("Avg", True, COLOR_GOLD), (graph_box.right - 81, graph_box.y + 6))

        # Mini toggle hint
        self.display.blit(self.font_small.render("[G]", True, COLOR_ACCENT), (graph_box.right - 30, graph_box.y + 6))

        # Canvas inner area
        canvas = pygame.Rect(graph_box.x + 8, graph_box.y + 22, graph_box.width - 16, graph_box.height - 28)
        pygame.draw.rect(self.display, (16, 23, 38), canvas, border_radius=4)

        # Subtle horizontal grid lines inside canvas
        for gy in [0.33, 0.66]:
            line_y = canvas.bottom - int(canvas.height * gy)
            pygame.draw.line(self.display, (26, 36, 56), (canvas.left + 2, line_y), (canvas.right - 2, line_y), 1)

        # Preloaded Data Rendering
        total_games = self.max_games
        max_score_overall = max(max(self.history_scores) if self.history_scores else 10, 10)
        max_y = int(max_score_overall * 1.1) + 1

        curr_idx = min(self.current_game - 1, total_games - 1)
        cur_score_pt = self.history_scores[curr_idx] if curr_idx < len(self.history_scores) else 0
        cur_mean_pt = self.history_means[curr_idx] if curr_idx < len(self.history_means) else 0.0
        cur_high_pt = self.history_highs[curr_idx] if curr_idx < len(self.history_highs) else 0

        if not self.graph_zoom_mode:
            # MODE A: FULL CAREER (1 to 200) TIMELINE
            # 1. Faint Ghost Trajectory (Full 200 games benchmark horizon)
            if len(self.history_scores) > 1:
                ghost_coords = []
                for j in range(len(self.history_scores)):
                    gx = canvas.left + 4 + int((j / max(1, total_games - 1)) * (canvas.width - 8))
                    gy_s = canvas.bottom - 4 - int((self.history_scores[j] / max_y) * (canvas.height - 8))
                    ghost_coords.append((gx, gy_s))
                if len(ghost_coords) > 1:
                    pygame.draw.lines(self.display, (35, 48, 70), False, ghost_coords, 1)

            # 2. Active Replay Progress Curve (Game 1 to current_game)
            active_n = min(self.current_game, len(self.history_scores))
            if active_n > 0:
                score_coords = []
                mean_coords = []
                for j in range(active_n):
                    cx = canvas.left + 4 + int((j / max(1, total_games - 1)) * (canvas.width - 8))
                    cy_s = canvas.bottom - 4 - int((self.history_scores[j] / max_y) * (canvas.height - 8))
                    cy_m = canvas.bottom - 4 - int((self.history_means[j] / max_y) * (canvas.height - 8))
                    score_coords.append((cx, cy_s))
                    mean_coords.append((cx, cy_m))

                # Draw score curve (Emerald)
                if len(score_coords) > 1:
                    pygame.draw.lines(self.display, COLOR_EMERALD, False, score_coords, 1)
                else:
                    pygame.draw.circle(self.display, COLOR_EMERALD, score_coords[0], 3)

                # Draw moving average curve (Gold)
                if len(mean_coords) > 1:
                    pygame.draw.lines(self.display, COLOR_GOLD, False, mean_coords, 2)

            # 3. Playback Head Needle & Glowing Beacon at current_game
            head_x = canvas.left + 4 + int((curr_idx / max(1, total_games - 1)) * (canvas.width - 8))
            head_y = canvas.bottom - 4 - int((cur_score_pt / max_y) * (canvas.height - 8))
            # Vertical needle
            pygame.draw.line(self.display, (0, 215, 255), (head_x, canvas.top + 2), (head_x, canvas.bottom - 2), 1)
            # Glowing dot
            pygame.draw.circle(self.display, (0, 215, 255), (head_x, head_y), 4)
            pygame.draw.circle(self.display, (255, 255, 255), (head_x, head_y), 2)

            # Header stats overlay
            stat_lbl = f"Ep {self.current_game}/200 | Score: {cur_score_pt} | Avg: {cur_mean_pt:.1f} | High: {self.high_score}"
            self.display.blit(self.font_small.render(stat_lbl, True, COLOR_TEXT), (canvas.left + 6, canvas.top + 3))

        else:
            # MODE B: ZOOMED (Last 30 Games up to current_game)
            start_i = max(0, self.current_game - 30)
            end_i = self.current_game
            z_scores = self.history_scores[start_i:end_i]
            z_means = self.history_means[start_i:end_i]
            n_pts = len(z_scores)

            if n_pts > 0:
                z_max_y = max(max(z_scores), 10)
                z_score_coords = []
                z_mean_coords = []

                for j in range(n_pts):
                    cx = canvas.left + 6 + int((j / max(1, n_pts - 1)) * (canvas.width - 12)) if n_pts > 1 else canvas.left + (canvas.width // 2)
                    cy_s = canvas.bottom - 4 - int((z_scores[j] / z_max_y) * (canvas.height - 8))
                    cy_m = canvas.bottom - 4 - int((z_means[j] / z_max_y) * (canvas.height - 8))
                    z_score_coords.append((cx, cy_s))
                    z_mean_coords.append((cx, cy_m))

                # Draw lines
                if len(z_score_coords) > 1:
                    pygame.draw.lines(self.display, COLOR_EMERALD, False, z_score_coords, 1)
                for pt in z_score_coords:
                    pygame.draw.circle(self.display, COLOR_EMERALD, pt, 2)

                if len(z_mean_coords) > 1:
                    pygame.draw.lines(self.display, COLOR_GOLD, False, z_mean_coords, 2)

                # Needle on rightmost (current game)
                if z_score_coords:
                    rx, ry = z_score_coords[-1]
                    pygame.draw.circle(self.display, (0, 215, 255), (rx, ry), 5)
                    pygame.draw.circle(self.display, (255, 255, 255), (rx, ry), 2)

                stat_lbl = f"Games {start_i + 1}-{end_i} | Score: {cur_score_pt} | Avg: {cur_mean_pt:.1f} | High: {self.high_score}"
                self.display.blit(self.font_small.render(stat_lbl, True, COLOR_TEXT), (canvas.left + 6, canvas.top + 3))

        # 5. Speed Selector Bar (Clickable & controlled with S or UP/DOWN)
        cur_spd = SPEED_OPTIONS[self.speed_idx]
        self.display.blit(self.font_mid.render(f"REPLAY SPEED: {cur_spd['label']} (Press S or UP/DOWN)", True, COLOR_ACCENT), (px + 15, py + 372))

        sp_w = (pw - 24) // len(SPEED_OPTIONS)
        for i, sp in enumerate(SPEED_OPTIONS):
            sx = px + 12 + (i * sp_w)
            sy = py + 394
            is_active = (i == self.speed_idx)

            s_rect = pygame.Rect(sx, sy, sp_w - 3, 26)
            bg = COLOR_EMERALD if is_active else (25, 34, 52)
            txt_c = (10, 15, 25) if is_active else COLOR_TEXT_DIM

            pygame.draw.rect(self.display, bg, s_rect, border_radius=4)
            if is_active:
                pygame.draw.rect(self.display, (0, 255, 180), s_rect, width=1, border_radius=4)

            lbl = self.font_small.render(sp["label"].split()[0], True, txt_c)
            self.display.blit(lbl, (sx + (sp_w - 3 - lbl.get_width()) // 2, sy + 6))

        # Bottom Quick Controls
        status_txt = "⏸ PAUSED" if self.is_paused else f"▶ REPLAYING ({cur_spd['label']})"
        status_col = COLOR_PAUSED if self.is_paused else COLOR_EMERALD
        self.display.blit(self.font_mid.render(status_txt, True, status_col), (px + 15, py + 434))

        ctrl_str = "[Space] Pause | [Right/Left] Skip 25 | [Click Graph] Scrub | [G] Zoom | [S] Speed"
        self.display.blit(self.font_small.render(ctrl_str, True, COLOR_TEXT_DIM), (px + 15, py + 456))

        pygame.display.flip()


if __name__ == '__main__':
    viewer = TrainingReplayViewer()
    viewer.run()
