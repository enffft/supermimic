"""Read effective physical COM through public APIs and verify partial reset/exploration."""

import json
from pathlib import Path

import numpy as np
import torch

from unisupermimic.tasks.velocity.go2.config import enable_randomization, load_config
from unisupermimic.tasks.velocity.go2.env import make_go2_env


def main():
    results = []
    for backend in ["mujoco", "mjwarp"]:
        env = make_go2_env(enable_randomization(load_config(backend)), 4, backend)
        env.init_state()
        try:
            command = env.command_manager.get_term("velocity")
            body = env.backend.get_body_id("robot/base")
            default = env.backend.get_body_ipos()[body]
            errors = []
            for _ in range(3):
                physical = env.backend.get_body_ipos(env_ids=np.arange(4))[:, body]
                expected = command.xi[:, 13:].cpu().numpy()
                errors.append(float(np.abs(physical - default - expected).max()))
                np.testing.assert_allclose(physical - default, expected, atol=1e-6, rtol=1e-5)
                before = command.xi.clone()
                env.reset(torch.tensor([0], device=env.device))
                assert torch.equal(before[1:], command.xi[1:])
            env.set_collection_exploration(0.1)
            state = env.step(torch.zeros(4, 12, device=env.device))
            assert state.info["supermimic"]["a"].abs().max() > 0
            assert env.current_info()["a_prev"].eq(0).all()
            results.append(
                {
                    "backend": backend,
                    "envs": 4,
                    "partial_resets": 3,
                    "com_max_abs_errors": errors,
                    "context": command.xi.cpu().tolist(),
                    "clean_executed_checked": True,
                }
            )
        finally:
            env.close()
            env.backend.close()
    out = Path(__file__).resolve().parents[1] / "results/domain_randomization_fields.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
