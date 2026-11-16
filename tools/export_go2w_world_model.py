"""Export a standalone conditional WM asset; excludes Actor, Critic and Replay."""

import argparse
import json
from pathlib import Path

from unisupermimic.tasks.velocity.go2w.world_model import export_world_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_world_model(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
