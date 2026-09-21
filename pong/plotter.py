import matplotlib
matplotlib.use('Agg')  # Headless backend to prevent Python 3.14 Tkinter/SDL GIL thread conflicts
import matplotlib.pyplot as plt

plt.style.use('dark_background')


class SafePlotter:
    """
    Thread-safe plotter using Matplotlib's 'Agg' (raster-only) backend.
    Renders high-resolution training summary plots without opening Tkinter GUI windows,
    completely preventing Python 3.14 GIL restoration crashes.
    """

    def __init__(self):
        pass

    def save_plot(self, scores, mean_scores, file_path="training_curve.png"):
        """Generates and saves the publication-quality training curve image."""
        if not scores:
            return

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('#0f141e')
        ax.set_facecolor('#161f30')
        ax.grid(True, linestyle='--', alpha=0.25, color='#4a5d78')

        # Title & Labels
        ax.set_title("Deep Q-Learning Snake AI - Performance Curve", fontsize=14, fontweight='bold', color='#ffffff', pad=12)
        ax.set_xlabel("Episode (Game Number)", fontsize=11, color='#a0b0c8')
        ax.set_ylabel("Score (Apples Eaten)", fontsize=11, color='#a0b0c8')

        # Plot raw scores and rolling mean
        ax.plot(scores, label='Episode Score', color='#00e5a3', alpha=0.55, linewidth=1.2)
        ax.plot(mean_scores, label='Mean Score (Rolling)', color='#ff9800', linewidth=2.5)

        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=max(len(scores) + 5, 20))

        best_score = max(scores)
        latest_mean = mean_scores[-1]

        stat_text = f"Total Episodes: {len(scores)} | Peak Score: {best_score} | Final Mean: {latest_mean:.2f}"
        ax.text(
            0.03, 0.93, stat_text,
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            color='#ffffff',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#1e293b', edgecolor='#334155', alpha=0.9)
        )

        ax.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', fontsize=9)
        plt.tight_layout()
        fig.savefig(file_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        print(f">>> [Plot Saved] High-res training curve saved to: {file_path}")
