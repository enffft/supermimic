# Go2 / UniLab 独立迁移实验

工作区 `/home/yifeiy/ws_go2_unilab`，conda 环境 `go2_unilab`，分支 `experiment/go2-velocity-unilab`。

[完整中文技术报告](docs/go2_unilab_migration_report.md) 包含架构矩阵、变更文件、复现命令、等价性结果、完整网络 profiler、峰值显存、地形能力和未解决问题。

```bash
cd /home/yifeiy/ws_go2_unilab/Uni-SuperMimic
conda activate go2_unilab
OMP_NUM_THREADS=4 python -m pytest tests/tasks/test_go2_velocity.py tests/tasks/test_go2_terrain.py tests/envs/test_go2_velocity_env.py tests/rl/algos -q
```

任务名 `Go2VelocitySuperMimic`，Hydra 选择 `task=go2_velocity/mujoco` 或 `task=go2_velocity/mjwarp`。默认32环境、8迭代，仅作平地部署检查。

**平地基线尚未验收。复杂地形训练在代码中被阻止。** 当前完成可运行迁移和受控短测；瞬时加速度 Reward、完整课程和 WM 收敛仍需处理。原仓库、旧 Checkpoint、原 Python 环境和 UniLab 上游源码保持原状。
