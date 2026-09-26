"""
visualize_env.py — Viewer entry point.

--task search (default) and --task mesh play the learned drone policy.
--task town opens the USGS / metro spectator (no drones).

Usage:
  cd drone_mesh_rl && python visualize_env.py
  cd drone_mesh_rl && python visualize_env.py --task town
  cd drone_mesh_rl && python visualize_env.py --task town --procedural
"""

import sys


def _split_task(argv):
    """Pull --task out of argv. Default is the drone search playback."""
    task = "search"
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--task" and i + 1 < len(argv):
            task = argv[i + 1]
            i += 2
            continue
        rest.append(argv[i])
        i += 1
    return task, rest


def main():
    task, rest = _split_task(sys.argv[1:])
    if task == "town":
        sys.argv = [sys.argv[0], *rest]
        from visualize_town import main as town_main

        town_main()
        return
    if task not in ("search", "mesh"):
        raise SystemExit(f"Unknown --task {task!r}. Use search, mesh, or town.")
    sys.argv = [sys.argv[0], "--task", task, *rest]
    from visualize_search import main as search_main

    search_main()


if __name__ == "__main__":
    main()
