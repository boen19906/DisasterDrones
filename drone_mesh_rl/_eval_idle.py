import numpy as np
import torch

from envs.search_env import SurvivorSearchEnv
from models.actor_critic import MAPPOModel

env = SurvivorSearchEnv(num_drones=3, env_size=200.0, max_steps=1200, render_mode=None, seed=9000)
model = MAPPOModel(3, env.obs_dim, env.act_dim, discrete=True, state_dim=env.state_dim)
model.load("models/mappo_search.pt", map_location="cpu")
model.eval()
names = env.possible_agents
for ep in range(5):
    obs, _ = env.reset(seed=9000 + ep)
    idle = 0
    steps = 0
    while True:
        arr = np.array([obs[a] for a in names], np.float32)
        st = env.global_state()
        with torch.no_grad():
            actions, _, _, _ = model.get_action_and_value(
                torch.tensor(arr), torch.tensor(st), deterministic=True
            )
        obs, _, terms, truncs, infos = env.step(
            {names[i]: actions.numpy()[i] for i in range(3)}
        )
        steps += 1
        speeds = [
            float(np.hypot(env.drone_velocities[i, 0], env.drone_velocities[i, 1]))
            for i in range(3)
        ]
        if min(speeds) < 0.5:
            idle += 1
        info = infos[names[0]]
        if any(terms.values()) or any(truncs.values()):
            break
    found = info["discovered_survivors"]
    total = info["total_survivors"]
    print(
        "ep", ep,
        "found", found, "/", total,
        "cover", round(info["coverage_frac"] * 100, 1),
        "steps", steps,
        "all_found", found == total,
        "idle_steps", idle,
    )
env.close()
