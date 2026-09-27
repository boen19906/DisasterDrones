"""
visualize_env.py — Viewer entry point.

--task search (default) plays MAPPO over the rubble city (5 m AGL).
--task mesh plays the LEO mesh demo.
--task town opens the USGS / metro spectator (no drones).
--task rubble opens the isolated rubble-city spectator (no drones).

Usage:
  cd drone_mesh_rl && python visualize_env.py
  cd drone_mesh_rl && python visualize_env.py --task search --policy model
  cd drone_mesh_rl && python visualize_env.py --task town
  cd drone_mesh_rl && python visualize_env.py --task rubble
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
    if task == "rubble":
        sys.argv = [sys.argv[0], *rest]
        from visualize_rubble import main as rubble_main

        rubble_main()
        return
    if task not in ("search", "mesh"):
        raise SystemExit(f"Unknown --task {task!r}. Use search, mesh, town, or rubble.")
    sys.argv = [sys.argv[0], "--task", task, *rest]
    from visualize_search import main as search_main

    search_main()


if __name__ == "__main__":
    main()
