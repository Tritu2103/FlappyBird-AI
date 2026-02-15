import os
import sys
import argparse
import numpy as np
import asyncio

# Add root directory to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from agent.dqn_agent import DQNAgent
from FlapPyBird.src.flappy import Flappy
from FlapPyBird.src.entities import Background, Floor, Pipes, Player, Score, PlayerMode


async def evaluate_dqn(checkpoint_path: str, num_episodes: int = 10, no_render: bool = False):
    """
    Đánh giá DQN agent đã train.
    
    Args:
        checkpoint_path: Đường dẫn đến model checkpoint
        num_episodes: Số episodes để evaluate
        no_render: Tắt rendering để chạy nhanh hơn
    """
    # Load agent
    agent = DQNAgent()
    agent.load(checkpoint_path)
    
    print("=" * 50)
    print(f"Evaluating DQN Agent")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Render: {'No' if no_render else 'Yes'}")
    print("=" * 50)
    print()
    
    scores = []
    
    for episode in range(num_episodes):
        # Initialize game
        game = Flappy()
        
        # Initialize entities
        game.background = Background(game.config)
        game.floor = Floor(game.config)
        game.player = Player(game.config)
        game.pipes = Pipes(game.config)
        game.score = Score(game.config)
        
        # Chạy game với/không render
        if no_render:
            score = await game.play_no_render(agent=agent)  # ← Không render
        else:
            score = await game.play(agent=agent)  # ← Có render
        
        scores.append(score)
        print(f"Episode {episode + 1}: Score = {score}")
    
    # Statistics
    print()
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Std Dev: {np.std(scores):.2f}")
    print(f"Min Score: {np.min(scores)}")
    print(f"Max Score: {np.max(scores)}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DQN agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering for faster evaluation")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    asyncio.run(evaluate_dqn(args.checkpoint, args.episodes, args.no_render))


if __name__ == "__main__":
    main()
