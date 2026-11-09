# 随代码发布的验证证据

这些文件来自已完成的受控验证，并非长训结果。

- `actor_equivalence.json`、`physics_equivalence.json`：旧Actor逐层前向和物理参数检查。
- `domain_randomization_fields.json`：两后端有效COM字段及部分Reset检查。
- `physics_final_{mujoco,mjwarp}/`：固定7组速度命令的物理测试。
- `profile_serial_{mujoco,mjwarp}/`：相同完整网络、32环境、batch1024的串行短训及分阶段耗时。
- `terrain_{rough,slope,stairs,mixed}_force/`：地形生成、Reset、scanner和公开快照接触力检查。

目录内保留配置、命令、JSON结果、训练CSV与profiler operator表。权重、TensorBoard事件、大profiler trace、早期失败运行和本机完整环境快照不进入Git；原件保存在 `/home/yifeiy/ws_go2_unilab`。复现入口和已知限制见[中文技术报告](../docs/go2_unilab_migration_report.md)。
