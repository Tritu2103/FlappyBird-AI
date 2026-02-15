import os
import sys
import asyncio
import numpy as np
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from agent.dqn_agent import DQNAgent
from FlapPyBird.src.flappy import Flappy
from FlapPyBird.src.entities import Background, Floor, Pipes, Player, Score


async def quick_eval(checkpoint_path: str, num_episodes: int = 100):
    """
    Đánh giá siêu nhanh (không render).
    """
    agent = DQNAgent()
    agent.load(checkpoint_path)
    
    print(f"Quick Evaluation ({num_episodes} episodes)")
    print(f"Checkpoint: {checkpoint_path}")
    print()
    
    scores = []
    start_time = time.time()
    
    for episode in range(num_episodes):
        game = Flappy(headless=True)  # Headless mode
        
        # Initialize
        game.background = Background(game.config)
        game.floor = Floor(game.config)
        game.player = Player(game.config)
        game.pipes = Pipes(game.config)
        game.score = Score(game.config)
        
        # Play fast
        score = await game.play_fast(agent=agent)
        scores.append(score)
        
        if (episode + 1) % 10 == 0:
            print(f"Progress: {episode + 1}/{num_episodes}", end='\r')
    
    elapsed = time.time() - start_time
    
    print()
    print("=" * 50)
    print("Results:")
    print(f"Episodes: {num_episodes}")
    print(f"Time: {elapsed:.2f}s ({elapsed/num_episodes*1000:.1f}ms/episode)")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Std Dev: {np.std(scores):.2f}")
    print(f"Min: {np.min(scores)}")
    print(f"Max: {np.max(scores)}")
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    
    asyncio.run(quick_eval(args.checkpoint, args.episodes))
