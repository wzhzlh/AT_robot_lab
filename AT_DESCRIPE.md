# Robot Lab训练机器人基本流程命令示例（GO2机器人）


ATDOG2
任务名称：
RobotLab-Isaac-Velocity-Flat-ATDog-Dog2-v0      --平地
RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-v0     --崎岖
RobotLab-Isaac-Velocity-Stairs-ATDog-Dog2-v0    --台阶
RobotLab-Isaac-Velocity-Sand-ATDog-Dog2-v0      --沙地
RobotLab-Isaac-Velocity-Slope-ATDog-Dog2-v0     --斜坡

ATDOG2ARM
任务名称：
RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-Arm-v0   --任务赛（减速带）

ATDOG3
任务名称：
RobotLab-Isaac-Velocity-Flat-ATDog-Dog3-v0      --平地
RobotLab-Isaac-Velocity-Rough-ATDog-Dog3-v0     --崎岖
RobotLab-Isaac-Velocity-Stairs-ATDog-Dog3-v0    --台阶
RobotLab-Isaac-Velocity-Sand-ATDog-Dog3-v0      --沙地
RobotLab-Isaac-Velocity-Slope-ATDog-Dog3-v0     --斜坡

ATDOG3ARM
任务名称：
RobotLab-Isaac-Velocity-Rough-ATDog-Dog3-Arm-v0   --任务赛（减速带）




注意：ATDOG已经废弃，请使用ATDOG3


export http_proxy=192.168.2.180:7890
export https_proxy=192.168.2.180:7890
export socket_proxy=192.168.2.180:7890

1.  ssh -p 3022  sw@shenweitechnology.com

2.  cd /home/sw/code/AT_rl_sar

3.  
export http_proxy=192.168.2.180:7890
export https_proxy=192.168.2.180:7890

4.  docker/container.sh start

## 创建环境：

```bash
docker/container.sh build
docker/container.sh start
docker/container.sh enter
```

## ATDOG2 步态相位、奖励与策略观测改进方案

更新日期：2026-11-11。本轮已实现固定频率步态状态、接触约束、摆动间隙、持续站立奖励及导出修复；暂不扩展观测。下方保留方案说明及后续观测升级方向。修改前的检查结果见 [ATDOG2 步态相位检查报告](docs/ATDOG2_GAIT_PHASE_REVIEW.md)。

当前 ATDOG2 的 actor 使用相位 sin/cos、机身角速度、重力投影、速度命令、关节位置、关节速度和上次动作，单帧共 47 维；HIM 使用 6 帧历史，共 282 维，按从旧到新排列，当前帧位于最后。Actor 不输入真实机身线速度及地形扫描，critic 保留真实机身线速度，但不再输入地形扫描。默认控制频率为 50 Hz，步态周期为 0.7 s（约 1.43 Hz）。

平地 `Flat` 和继承它的 `FlatDomainRand` 单回合上限设为 5 s（250 个控制步），其他地形保持原配置。单回合时长控制环境超时重置，不等同于训练日志中的 `Iteration time`；运行中的训练需要重新启动后才会加载此修改，也可在启动命令末尾使用 `env.episode_length_s=5.0` 覆盖。

1. **统一步态状态。**

   由命令模块维护相位、运动状态及启停规则，观测、奖励和部署端共同读取同一配置。平移速度与偏航角速度分别设置进入/退出阈值，通过迟滞避免低速命令附近反复切换。相位只在控制循环中推进一次，观测和奖励函数只读取状态。导出信息需包含周期、相位编码、初始状态、运动阈值、停止规则及更新顺序，确保训练与部署使用同一套时钟逻辑。

2. **重构接触奖励。**

   统一使用显式足序 FL、FR、RL、RR，对角 trot 的相位偏移为 `[0, 0.5, 0.5, 0]`，通过支撑占空比区分支撑与摆动。主要惩罚“摆动阶段仍接触”和“支撑阶段未接触”，在相位边界设置平滑宽容区，允许低速和台阶上的短暂多脚支撑。新的相位接触项稳定后，关闭或降低重复的对角同步约束。

   修改前的周期项中，四脚全着地、全腾空都能获得 0.5 的基础分，理想交替 trot 的周期平均分约为 0.8183。本轮已替换为接触违约惩罚，不再提供该项正基础分。

3. **单独设计摆动间隙。**

   摆动脚跟踪平滑的离地高度包络，例如按摆动进度生成 `h_des = h_min + A * sin²(πu)`；支撑脚主要约束滑移和异常接触。高度应相对局部地形、按实际足底位置计算，核对固定脚合并后的几何偏移，避免用 calf 原点的世界高度代替足底间隙。逐步替换不区分支撑/摆动的全腿固定高度目标，并处理台阶侧壁、射线未命中等情况。

4. **完善站立与停止。**

   停止时让摆动腿完成落脚后进入稳定支撑，避免运动中强制相位置零，并设置过渡超时和恢复规则。静止奖励采用持续支撑、低机身/脚部速度、低滑移及低动作变化；当前使用“首次接触”的静止奖励可能鼓励反复抬脚再落地，应改为持续状态指标。平移、原地转向及混合命令分别处理，避免纯偏航命令被平移进度门控屏蔽。

5. **分版本升级观测。**

   第一版保留 47 维单帧、6 帧共 282 维的输入布局，优先修复奖励、统一门控和导出契约，采用简单、确定性的启停规则。后续若引入可变频率、占空比和渐变启停，再追加频率、占空比、活动强度等状态；仅增加这三项时为 50 维单帧、共 300 维。若启停仍依赖额外 mode 或计时器，还需提供足够观测，避免隐藏状态歧义。

   观测升级须同步更新 HIM 配置、历史缓存、导出字段映射及部署端输入契约。282 维旧模型不能直接按 300 维输入恢复；即使维度不变，相位定义或停止规则改变后也需要验证模型兼容性。

实施顺序：先修复确定性缺陷，再完成平地固定频率接触奖励、摆动间隙与启停逻辑，随后迁移到随机化及台阶地形，最后开展可变步态观测升级。本轮已修复 HIM 导出读取不存在的 `phase.params["cycle_time"]`，并将 stairs 的 `feet_stumble` 改为负权重。

验收需覆盖低速、侧移、后退、原地转向、启停及台阶场景，同时比较速度误差、相位接触匹配率、实际摆动间隙、滑移、落地冲击、跌倒率与台阶通过率。部署验证应使用相同命令序列，逐步比较训练端和部署端相位输出，不能只检查模型是否成功导出。

### 本轮实现参数与兼容约束

Flat 零速偏航调整：`Flat` 和 `FlatDomainRand` 新增 `standing_yaw_drift`，权重 -4，误差尺度 0.15 rad/s，惩罚为 `min((yaw_rate/0.15)^2, 9)`；仅在步态时钟停止、完成收脚后生效，保留正常转向。两任务的 `standing_support` 权重提高到 +6，其他地形保持 +4。观测和运动时偏航跟踪参数保持不变。这些奖励参数需要重新训练或微调才能影响策略，不能直接消除旧模型的偏航漂移。

2026-11-12 调参：周期从 0.5 s 调到 0.7 s，降低步频；持续站立奖励从 +2 提高到 +4，增加权重 -2 的零速关节运动惩罚。该惩罚在时钟停止后生效，不干扰收脚过渡。关节速度按 0.5 rad/s、动作差按 0.05 归一化，并限制单分量误差。参数变化需要重新训练或继续微调后再评估，旧模型只修改时钟不保证形成长步幅。

47 维单帧顺序及缩放保持不变：

| 切片 | 观测 | 缩放 |
|---|---|---|
| `[0:2]` | 相位 sin、cos | 1 |
| `[2:5]` | 机身角速度 | 0.25 |
| `[5:8]` | 重力投影 | 1 |
| `[8:11]` | vx、vy、yaw 命令 | `(1, 1, 0.25)` |
| `[11:23]` | 相对默认关节位置 | 1 |
| `[23:35]` | 关节速度 | 0.05 |
| `[35:47]` | 上次动作 | 1 |

关节与动作仍按 FR、FL、RR、RL 排列，每腿依次 hip、thigh、calf；奖励通过名称单独解析 FL、FR、RL、RR，避免与观测索引混用。

时钟参考实现位于 `source/robot_lab/robot_lab/gait.py`，配置入口为 `commands.base_velocity.gait`。默认固定周期 0.7 s，支撑占空比 0.6，相位边界宽容区宽度为周期的 0.05（Stairs 的覆盖值见后文「观测契约与逐任务 policy / critic 对照」）。平移进入/退出阈值为 0.08/0.04 m/s，偏航为 0.10/0.05 rad/s。停止时推进到下一段计划双支撑并冻结，最长不超过半个周期；这是按相位结束摆动，不根据接触反馈等待实际落地。该版本没有额外停止计时器、可变频率或活动强度观测。

各 dog2 环境统一通过 `dog2/periodic_gait.py:configure_gait_rewards()` 配置：

| 奖励项 | 权重 | 定义 |
|---|---:|---|
| `gait_contact` | -5 | 支撑未接触、摆动仍接触的相位违约 |
| `swing_clearance` | -10（stairs 覆盖为 -15） | 摆动足底相对脚下地形的归一化间隙误差 |
| `feet_slide` | -1 | 接触脚中心的水平滑移，包含刚体转动引起的速度 |
| `standing_support` | +4 | 持续接触、低机身/脚速、低动作变化、直立姿态 |
| `standing_joint_motion` | -2 | 停止过渡结束后的关节速度与动作变化惩罚 |

摆动包络峰值为 flat/slope/sand 0.07 m、rough 0.05 m、flat_domain_rand 0.08 m、bar/tires 0.10 m、stairs 0.12 m；间隙误差尺度默认 0.10 m，stairs 为 0.06 m 并额外启用 `deficit_only`、`lookahead_distance=0.08`、`terrain_sample_radius=0.0384`。足底位置使用 URDF 固定足关节偏移，并按足部网格外接圆柱计算随姿态变化的竖向支撑范围；它是足底几何近似，不是 calf 原点高度。训练奖励复用特权地形扫描器加载的网格，在脚中心竖直向下射线查询地面；无效射线和过陡侧面命中不计入间隙项。Actor 没有新增地形输入。

本轮停用旧周期接触奖励、旧对角同步/接触模式、全腿固定高度、腾空时间/方差及首次落地式静止奖励。原有速度跟踪和其他稳定性约束继续参与训练。权重与间隙高度是训练起点，尚未通过多种子训练评估效果。

导出目录新增 `gait_clock.py`，`policy_metadata.json` 记录完整时钟配置。部署端应使用 `TrotClock.from_metadata(num_envs, device, metadata["gait_clock"])` 或等价实现；旧的按回合时间生成正弦的方式不兼容新启停规则。HIM 仍使用 282 维、从旧到新的历史，导出的历史索引为 `[5, 4, 3, 2, 1, 0]`，当前帧在最后。外部部署控制器仍需接入这些规则。

维度保持相同不代表旧策略行为自动兼容：本轮改变了相位支撑定义、奖励及启停规则，建议重新训练基线，或在验证后微调旧权重。本轮没有启动训练。

验证命令（在现有容器内、项目目录下执行）：

```bash
/workspace/isaaclab/isaaclab.sh -p tests/test_gait.py
/workspace/isaaclab/isaaclab.sh -p tests/check_atdog2_gait_sim.py --headless
/workspace/isaaclab/isaaclab.sh -p tests/check_atdog2_gait_sim.py --headless --terrain stairs
```

已完成全部 dog2 环境配置检查、步态数值及现有回归测试；平地与台阶各 4 个环境运行 50 个控制步，检查了 47/282 维顺序、有限奖励、平移/转向/停止，以及 ONNX/JIT、元数据和部署时钟一致性。这些检查证明接线与计算路径可运行，步态质量和台阶通过率仍需训练评估。

## ATDOG2 各训练任务的配置与观测

更新日期：2026-11-14。本节按当前代码（`source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/atdog/dog2/`）整理各任务的注册入口、地形、指令、奖励权重、训练超参与观测契约，作为上一节步态/奖励方案的配置索引。两节数值冲突时以本节为准。本轮按 `dog2/periodic_gait.py::configure_periodic_gait()` 与各任务 `__post_init__()` 重新核对了 8 个任务的 policy / critic 观测，并补充逐任务对照表。

### 1. 任务注册与入口

| 任务 ID | 环境配置类（文件） | HIM runner | PPO runner |
|---|---|---|---|
| `RobotLab-Isaac-Velocity-Flat-ATDog-Dog2-v0` | `ATDogDog2FlatEnvCfg`（`flat_env_cfg.py`） | `ATDogDog2FlatHIMRunnerCfg` | `ATDogDog2FlatPPORunnerCfg` |
| `RobotLab-Isaac-Velocity-Flat-DomainRand-ATDog-Dog2-v0` | `ATDogDog2FlatDomainRandEnvCfg`（`flat_env_cfg.py`） | `ATDogDog2FlatDomainRandHIMRunnerCfg` | `ATDogDog2FlatPPORunnerCfg` |
| `RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-v0` | `ATDogDog2RoughEnvCfg`（`rough_env_cfg.py`） | `ATDogDog2RoughHIMRunnerCfg` | `ATDogDog2RoughPPORunnerCfg` |
| `RobotLab-Isaac-Velocity-Stairs-ATDog-Dog2-v0` | `ATDogDog2StairsEnvCfg`（`stairs_env_cfg.py`） | `ATDogDog2StairsHIMRunnerCfg` | `ATDogDog2StairsPPORunnerCfg` |
| `RobotLab-Isaac-Velocity-Sand-ATDog-Dog2-v0` | `ATDogDog2SandEnvCfg`（`sand_env_cfg.py`） | `ATDogDog2SandHIMRunnerCfg` | `ATDogDog2SandPPORunnerCfg` |
| `RobotLab-Isaac-Velocity-Bar-ATDog-Dog2-v0` | `ATDogDog2BarEnvCfg`（`bar_env_cfg.py`） | `ATDogDog2BarHIMRunnerCfg` | `ATDogDog2BarPPORunnerCfg` |
| `RobotLab-Isaac-Velocity-Tires-ATDog-Dog2-v0` | `ATDogDog2TiresEnvCfg`（`tires_env_cfg.py`） | `ATDogDog2TiresHIMRunnerCfg` | `ATDogDog2TiresPPORunnerCfg` |
| `RobotLab-Isaac-Velocity-Slope-ATDog-Dog2-v0` | `ATDogDog2SlopeEnvCfg`（`slope_env_cfg.py`） | `ATDogDog2SlopeHIMRunnerCfg` | `ATDogDog2SlopePPORunnerCfg` |

每个任务同时注册了 `cusrl_cfg_entry_point`（`cusrl_ppo_cfg.py`，与 rsl_rl 并行的一套训练器）。注册表见 `dog2/__init__.py`，训练入口为 `scripts/reinforcement_learning/rsl_rl/train.py --agent=rsl_rl_him_cfg_entry_point`。

### 2. 公共环境配置

| 项目 | 取值 |
|---|---|
| 机器人资产 | `AT_DOG2_CFG`（URDF 固定足关节合并到 `.*_calf`，接触项按 calf 匹配；机身 `base`） |
| 关节与动作顺序 | `FR, FL, RR, RL` × `hip, thigh, calf`，共 12 维，`preserve_order=True` |
| 控制频率 | `sim.dt=0.005`，`decimation=4` → 50 Hz（控制周期 0.02 s） |
| 单回合时长 | 20 s；`Flat` / `Flat-DomainRand` 为 5 s（250 控制步） |
| 并行环境数 | 配置默认 4096，用 `--num_envs` 覆盖（近期 Flat 训练用 4000） |
| 动作 | `JointPositionActionCfg(scale=0.25, use_default_offset, clip=-100~100)`；除 Flat/Flat-DomainRand 外，`hip` 缩放改为 0.125 |
| 接触传感器 | `contact_forces`：`{ENV}/Robot/.*`，`history_length=3`，`track_air_time=True`，`update_period=0.005` |
| 地形传感器 | `height_scanner`：机身 base，0.1 m 网格 ×1.6×1.0 m → 187 条射线，`update_period=0.02`，不再进入 critic；保留给奖励/地形几何查询使用 |
| 测高传感器 | `height_scanner_base`：0.05 m 网格 ×0.1×0.1 m → 9 条射线，供 `base_height_l2`；`Bar` 关闭、`Flat` 未使用 |
| 终止条件 | `time_out`、`terrain_out_of_bounds`（3 m 缓冲）；`illegal_contact`（base/hip）在 Stairs/Sand/Slope/Bar/Tires 启用，Rough/Flat/Flat-DomainRand 关闭 |
| 地形课程 | `Rough` 保留 `terrain_levels`（初始等级 5）；`Bar`/`Flat-DomainRand` 显式置空；Stairs 按实际台阶进度与高度变化升降级；`Flat` 为 plane；`Sand`/`Slope`/`Tires` 未显式关闭，但只使用单一子地形，等级变化基本不改变几何 |

### 3. 观测契约与逐任务 policy / critic 对照

Policy（actor）单帧 47 维，HIM 取 6 帧历史 → 282 维（`oldest_to_latest`，当前帧在最后，导出索引 `[5,4,3,2,1,0]`）：

| 切片 | 观测 | 缩放 | 噪声 |
|---|---|---|---|
| `[0:2]` | 步态时钟 `sin`、`cos`（`stairs_phase` 读取命令模块时钟） | 1 | 无 |
| `[2:5]` | 机身角速度 | 0.25 | ±0.2 |
| `[5:8]` | 重力投影 | 1 | ±0.05 |
| `[8:11]` | vx、vy、yaw 命令 | `(1, 1, 0.25)` | 无 |
| `[11:23]` | 相对默认关节位置 | 1 | ±0.01 |
| `[23:35]` | 关节速度 | 0.05 | ±1.5 |
| `[35:47]` | 上次动作 | 1 | 无 |

Policy **不含**真实机身线速度与地形扫描（部分可观测）。Critic 共 50 维：与 policy 相同的 7 个项（同样定义、不带噪声），加上 3 维特权机身线速度；**顺序不同**（线速度在最前、相位在最后）。`configure_periodic_gait()` 对 8 个任务统一把 `critic.height_scan` 置 `None`、给 critic 增加 `phase` 项，Stairs 仅再次显式置空，不再重新添加扫描：

| Critic 切片 | 分量 | 维度 | 说明 |
|---|---|---:|---|
| `[0:3]` | `base_lin_vel` | 3 | 特权量，缩放 2.0；HIM 训练时 `align_him_critic_observations` 清除全部 critic 噪声并关闭 critic 加噪（`him_clean_critic=True`） |
| `[3:6]` / `[6:9]` / `[9:12]` | `base_ang_vel` / `projected_gravity` / `velocity_commands` | 3 / 3 / 3 | 与 policy 同名项对齐（含裁剪、缩放、参数） |
| `[12:24]` / `[24:36]` / `[36:48]` | `joint_pos` / `joint_vel` / `actions` | 12 / 12 / 12 | 与 policy 同名项对齐 |
| `[48:50]` | `phase` | 2 | 与 policy 同一时钟的 sin/cos |
| — | `height_scan` | 0 | 已置 `None`，不输入 Critic |

**逐任务 policy / critic 观测对照（2026-11-14 核对当前代码）**：记 `P = [phase(2), base_ang_vel(3), projected_gravity(3), velocity_commands(3), joint_pos(12), joint_vel(12), actions(12)]`（47 维），`C = [base_lin_vel(3), base_ang_vel(3), projected_gravity(3), velocity_commands(3), joint_pos(12), joint_vel(12), actions(12), phase(2)]`（50 维）。8 个任务的项集合、顺序、缩放与维度完全一致，差异只在相位时钟、命令分布和传感器挂载：

| 任务 | policy 单帧 | HIM actor / 标准 PPO、cusrl actor | critic | 相位时钟与命令来源 | 该任务观测相关差异 |
|---|---|---|---|---|---|
| Flat | `P`（47） | `P×6=282` / 47 | `C`（50） | `PhaseVelocityCommandCfg`（继承 Rough 的 `heading_command=False`）+ 默认 `TrotClockCfg` | yaw 命令直接采样；`base_height_l2.params["sensor_cfg"]=None`；两个射线传感器仍在场景中但不进入任何观测组 |
| Flat-DomainRand | `P`（47） | `P×6=282` / 47 | `C`（50） | 同上，命令范围改为 vx −0.8~1.2、vy ±0.6、站立比例 0.05 | 域随机化改变本体状态的数值分布，不改项定义；`base_height_l2` 重新指向 `height_scanner_base`（权重 0，被清理） |
| Rough | `P`（47） | `P×6=282` / 47 | `C`（50） | 同上；Rough 在 `configure_periodic_gait()` 之后把 `heading_command` 关掉、`rel_heading_envs=0` | 唯一显式给 `critic.joint_pos/joint_vel.asset_cfg.joint_names` 指定 `self.joint_names` 的任务；保留 `terrain_levels` 课程（不输入观测） |
| Stairs | `P`（47） | `P×6=282` / 47 | `C`（50） | `StairsVelocityCommandCfg`（`heading_command=False`），时钟 `cycle_time=0.5`、`duty_factor=0.55`、`transition_width=0.07`、平移/偏航门槛均 0.05/0.025 | `critic.height_scan=None` 再次显式置空；`phase` 的频率与占空比和其他任务不同，跨任务迁移时相位语义不一致；`terrain_levels_stairs` 只调地形等级，不进入观测 |
| Sand | `P`（47） | `P×6=282` / 47 | `C`（50） | `PhaseVelocityCommandCfg` + 默认时钟；未显式关闭 `heading_command`（父类默认 `True`，由 `configure_periodic_gait()` 复制），yaw 由 heading 误差生成 | 无扫描观测；`height_scanner_base` 的 9 条射线只供 `base_height_l2` 奖励 |
| Slope | `P`（47） | `P×6=282` / 47 | `C`（50） | 同 Sand 的默认时钟与 heading 命令语义 | 坡度变化不进入观测 |
| Bar | `P`（47） | `P×6=282` / 47 | `C`（50） | 同 Sand 的默认时钟与 heading 命令语义 | `height_scanner_base=None`；`height_scanner` 由 `MySceneCfg()` 重新挂到 plane 上，仅用于奖励/几何查询 |
| Tires | `P`（47） | `P×6=282` / 47 | `C`（50） | 同 Sand 的默认时钟与 heading 命令语义 | 两个射线传感器都重新挂到 `base`；均不进入观测组 |

- 运行期变换：只有 HIM 入口（`rsl_rl_him_cfg_entry_point`）调用 `apply_him_observation_history()`，把 `P` 设为 `history_length=6`、`flatten_history_dim=True`，再由 `HIMVecEnvWrapper._reorder_policy_history()` 排成 282 维（`oldest_to_latest`，当前帧 `[235:282]`）；标准 PPO / cusrl 的 actor 仍是 47 维单帧，critic 都是 50 维。
- `align_him_critic_observations()` 仅在 HIM 入口（train/play）把 critic 中与 policy 同名的 7 个项替换为 policy 项的深拷贝（含 scale、clip、noise、`asset_cfg`），随后按 `him_clean_critic=True` 清空噪声并设 `critic.enable_corruption=False`。因此 HIM 下所有任务的 `critic.joint_pos/joint_vel` 都使用 policy 的 12 关节显式顺序；标准 PPO / cusrl 不调用该函数，critic 保留父类 `joint_names=".*"`（Rough 例外）。
- `critic.height_scan=None` 关闭的是观测项，不是射线计算：场景仍保留 `height_scanner`；`lazy_sensor_update=True` 下 `configure_gait_rewards()` 的摆腿净空奖励每步访问 `scanner.data`，187 条网格射线仍会每步计算，速度收益需要实测。
- HIM 相关维度由 `HIMVecEnvWrapper` 在启动时校验并在日志打印 `policy_total / critic_total / policy_one_step`；导出时写入 `policy_metadata.json`（当前配置预期 282 / 50 / 47；历史有扫描 checkpoint 为 282 / 237 / 47）。

### 4. 各任务地形

| 任务 | 地形配置 | 难度/几何 | 课程 |
|---|---|---|---|
| Flat | `terrain_type="plane"`（仓库内 `default_environment.usd` 地面） | 平地；`height_scanner` 保留给奖励使用，不进入 critic | 无 |
| Flat-DomainRand | `ATDOG2_LOW_OBSTACLE_TERRAIN_CFG`（generator，8×8 m，10×20 tile） | 85% 平地 + 15% 低矮方格（网格 0.45 m，高 0.01–0.025 m，平台 2.0 m） | 无，`max_init_terrain_level=None` |
| Rough | `DOG2_ROUGH_TERRAIN_CFG`（generator，8×8 m，10×20 tile） | 倒金字塔 + 正金字塔台阶，阶高 0.09–0.11 m、阶宽 0.30 m、平台 3.0 m | `terrain_levels` 保留 |
| Stairs | `ATDog2StairsTerrainGeneratorCfg`（自定义生成器） | 30% 平地 + 65% 上台阶 + 5% 下台阶，阶高按课程从 0.04 m 到 0.12 m、阶宽 0.30 m | 实际台阶跨越课程 |
| Sand | `DOG2_OBSTACLE_TERRAIN_CFG`（generator） | 每 tile 16 条长条凸起，高 0.3 m、截面 0.03×1.6 m，`horizontal_scale=0.01`；注意这不是可变形沙地 | 未显式关闭（`object_params_start=end`，难度固定） |
| Bar | 实际 `terrain_type="plane"` | 文件内减速带地形 `DOG2_ARM_ROUGH_TERRAIN_CFG`（3 条 0.05 m 高、0.12 m 厚）已定义但未挂载 | 无 |
| Tires | `DOG2_TIRE_ARRAY_TERRAIN_CFG`（generator，5×5 tile，border 0） | 每 tile 2×2 轮胎阵列，外径 0.275 / 内径 0.165 / 宽 0.165 m、12 段，网格间距 1.18 m，中心留空作 reset 区 | 未显式关闭（单一子地形，等级不改变几何） |
| Slope | `DOG2_SLOPE_TERRAIN_CFG` | `HfPyramidSlopedTerrainCfg`，坡度 0–0.3（约 0–16.7°），平台 1.0 m | 未显式关闭（单一子地形，等级不改变几何） |

### 5. 各任务指令范围

| 任务 | 重采样 | 站立比例 | vx (m/s) | vy (m/s) | yaw (rad/s) | 备注 |
|---|---|---|---|---|---|---|
| Flat | 5 s | 0.15 | ±1.0 | ±1.0 | ±1.0 | `debug_vis` 默认开启 |
| Flat-DomainRand | 5 s | 0.05 | −0.8 ~ 1.2 | ±0.6 | ±1.0 | 推力扰动 15–20 s，±0.5 m/s |
| Rough | 5 s | 0.15 | ±1.0 | ±1.0 | ±1.0 | `debug_vis` 默认开启 |
| Stairs | 20 s | 0.15 | −0.8 ~ 1.2 | ±0.6 | ±1.0 | `StairsVelocityCommand`：沿台阶向外、速度 0.25–0.8 m/s、35% 侧向；成功判定 2.1 m / 抬升 0.06 m |
| Sand | 5 s | 0.15 | ±1.0 | ±0.5 | ±1.0 | |
| Slope | 5 s | 0.15 | ±1.0 | ±1.0 | ±1.0 | 使用父类默认范围 |
| Bar | 5 s | 0.15 | −1.0 ~ 1.5 | ±1.0 | ±1.5 | |
| Tires | 5 s | 0.15 | ±1.0 | ±1.0 | ±1.0 | 使用父类默认范围 |

### 6. 各任务奖励差异（权重）

公共项（`periodic_gait.py:configure_gait_rewards()`）：`gait_contact=-5`、`swing_clearance=-10`、`standing_support=+4`、`standing_joint_motion=-2`、`feet_slide=-1`；旧周期/腾空时间/固定抬腿项统一置空。

| 任务 | track_lin_xy | track_ang_z | base_height_l2 | undesired_contacts | action_rate | joint_pos_limits | stand_still | upward | gait_contact / swing | 摆动高度 | 其他 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---|
| Flat | 45 | 40 | 0 @0.32（传感器 None，零权重被清理） | −1 | −0.2 | −10 | −2 | +1 | −5 / −10 | 0.07 | `is_terminated=-3`、`feet_contact=+1`、`contact_forces=-0.005` |
| Flat-DomainRand | 45 | 40 | 同 Flat | −1 | −0.2 | −10 | −2 | +1 | −5 / −10 | 0.08 | 同上；域随机化开启（质量 ±0.1、质心 ±3 mm、增益 0.98–1.02） |
| Rough | 45 | 40 | 0 @0.32 | −1 | −0.2 | −10 | −2 | +1 | −5 / −10 | 0.05 | `applied_torque_limits=-0.01`、`feet_contact=+1` |
| Stairs | 80 | 70 | −10 @0.33 | −20（base/hip/thigh） | −1.5 | −2 | −10 | 0 | −7.5 / −15 | 0.12 | `stair_landing=-3`、`feet_stumble=-1`、`feet_distance_y=2.0` |
| Sand | 55 | 30 | −50 @0.34 | −100（非足端） | −2.0 | −2 | −30 | 0 | −5 / −10 | 0.07 | `feet_distance_y=2.0` |
| Slope | 70 | 65 | −5 @0.30 | −100 | −2.0 | −5 | −30 | +25 | −5 / −10 | 0.07 | `action_smoothness_2_l2=-0.3` |
| Bar | 55 | 30 | −8 @0.30（传感器 None） | −20 | −2.0 | −5 | −30 | 0 | −5 / −10 | 0.10 | |
| Tires | 95 | 80 | −100 @0.33 | −100 | −2.0 | −2 | −30 | 0 | −5 / −10 | 0.10 | `feet_distance_y=2.0`（std 0.15） |

速度跟踪均为指数型，`std`：Flat/Rough 0.8，其余沿用父类默认。关节位置惩罚的 `stand_still_scale`：Flat/Rough 5.0，others 8.0（Tires 4.0，Stairs 4.0）。

### 7. 训练超参（rsl_rl）

HIM（`--agent=rsl_rl_him_cfg_entry_point`）：

| 任务 | num_steps_per_env | max_iterations | save_interval | learning_rate | entropy_coef | schedule |
|---|---:|---:|---:|---|---:|---|
| Flat | 48 | 1000 | 50 | 5e-4 | 0.01 | adaptive |
| Flat-DomainRand | 48 | 1000 | 50 | 5e-5 | 0.003 | fixed |
| Stairs | 48 | 20000 | 100 | 5e-5 | 0.003 | fixed |
| Rough / Sand / Slope | 24 | 20000 | 100 | 1e-3 | 0.01 | adaptive |
| Bar | 24 | 5000 | 100 | 1e-3 | 0.01 | adaptive |
| Tires | 24 | 20000 | 100 | 1e-3 | 0.01 | adaptive |

- 网络：actor/critic 各 `[512, 256, 128]` + ELU，`init_noise_std=1.0`，不做观测归一化；PPO `epochs=5`、`mini_batches=4`、`clip=0.2`、`gamma=0.99`、`lam=0.95`、`desired_kl=0.01`、`max_grad_norm=1.0`。
- HIM 估计器：`num_one_step_obs=47`、`history_length=6`、`history_order=oldest_to_latest`、encoder `[128, 64, 16]`、target `[128, 64]`、`num_prototypes=32`、`temperature=3.0`、`estimator_lr=1e-3`、`velocity_loss_coef=1.0`、`swap_loss_coef=0.1`。
- 非 HIM 的 PPO runner：`num_steps_per_env=24`，`max_iterations` 为 Rough/Flat 2000、Bar 5000、其余 20000，`lr=1e-3`、`entropy_coef=0.01`、`adaptive`。
- cusrl 入口：`num_steps_per_update=24`、AdamW `lr=1e-3`、`epochs=5`、`mini_batches=4`、`EntropyLoss(0.008)`、`AdaptiveLRSchedule(kl=0.01)`；迭代上限 Rough/Sand/Slope/Stairs/Tires 20000、Flat/Bar 5000。
- `max_iterations` 可被 `--max_iterations` 覆盖（如近期 Flat 运行实际跑到 2400 轮）。

### 8. 迭代耗时参考

单轮耗时 ≈ `num_steps_per_env × num_envs` 的采集时间 + PPO 学习时间，与任务配置无关。4000 envs 实测：48 步/轮约 3.38 s 采集 + 0.65 s 学习（fps ≈ 47.6k），24 步/轮约 1.61 s + 0.30 s（fps ≈ 50.5k）。因此 Flat / Flat-DomainRand / Stairs 的单轮时间约为其他任务的 2 倍，总训练步数相同时墙钟时间基本一致；若要缩短单轮显示时间可把 `num_steps_per_env` 改回 24。

## ATDOG2 逐任务观测明细与策略配置

更新日期：2026-11-14。本节补充上一节总览中 8 个任务共用的观测项逐项定义、逐任务 policy / critic 观测对照、各任务实际挂载的传感器差异，以及三种训练器入口下的策略网络与超参。数值以当前代码以及 `logs/rsl_rl/atdog_dog2_flat_him/*/params/env.yaml`、`exported/policy_metadata.json` 的实测结果为准；本轮按 `periodic_gait.py`、各任务 `__post_init__()` 和 `stairs_rewards.py` 重新核对，Stairs 已改用 `configure_stair_rewards()` 且 `critic.height_scan=None`。

### 1. 观测项逐项定义（8 个任务共用）

Policy 组来自 `dog2/periodic_gait.py:ATDog2PolicyCfg()`，`configure_periodic_gait()` 在环境初始化时替换整个 policy 组、给 critic 加 `phase` 项并把 `critic.height_scan` 置 `None`；各任务随后统一把 `policy.base_lin_vel`、`policy.height_scan` 置为 `None`。`stairs_rewards.py` 不新增观测项。单帧 47 维，顺序如下：

| 顺序 | Policy 项 | 来源函数 | 维度 | scale | 训练噪声（加性均匀） | clip |
|---|---|---|---:|---|---|---|
| 1 | `phase` | `mdp.stairs_phase`（读 `PhaseVelocityCommand.gait_clock` 的 sin/cos） | 2 | 1（配置中为 `None`，不缩放） | 无 | 无 |
| 2 | `base_ang_vel` | `isaaclab.envs.mdp.observations.base_ang_vel` | 3 | 0.25 | ±0.2 | ±100 |
| 3 | `projected_gravity` | `isaaclab.envs.mdp.observations.projected_gravity` | 3 | 1.0 | ±0.05 | ±100 |
| 4 | `velocity_commands` | `mdp.generated_commands`（`base_velocity`） | 3 | `(1, 1, 0.25)` | 无 | ±100 |
| 5 | `joint_pos` | `mdp.joint_pos_rel`（相对默认关节角） | 12 | 1.0 | ±0.01 | ±100 |
| 6 | `joint_vel` | `mdp.joint_vel_rel` | 12 | 0.05 | ±1.5 | ±100 |
| 7 | `actions` | `mdp.last_action` | 12 | 1.0 | 无 | ±100 |
| — | `base_lin_vel` | 已置 `None` | — | — | — | — |
| — | `height_scan` | 已置 `None` | — | — | — | — |

- `policy.enable_corruption=True`（训练加噪）；`play.py` 评估/导出时改为 `False`。`critic.enable_corruption=False`，critic 观测不带噪声。
- HIM 运行时 `apply_him_observation_history()` 给 policy 组设置 `history_length=6`、`flatten_history_dim=True`；Isaac Lab 先按"每个 term 独立展开历史"拼接，`HIMVecEnvWrapper._reorder_policy_history()` 再重组为按帧排列的 282 维（`oldest_to_latest`，当前帧为 `[235:282]`）。标准 PPO / cusrl 入口不启用历史，actor 输入仍是 47 维。
- 步态时钟参数：Flat / Flat-DomainRand / Rough / Sand / Slope / Bar / Tires 使用 `TrotClockCfg()` 默认值（`cycle_time=0.7`、`duty_factor=0.6`、`transition_width=0.05`，平移门槛 0.08/0.04 m/s、偏航门槛 0.10/0.05 rad/s）；Stairs 通过 `StairsVelocityCommandCfg` 覆盖为 `cycle_time=0.5`、`duty_factor=0.55`、`transition_width=0.07`，平移与偏航门槛均为 0.05/0.025。`phase` 项读同一个时钟，因此 Stairs 的 sin/cos 频率、占空比与其他任务不同；其余任务的相位数值仍随启停规则和命令分布变化。

Critic 组当前为 50 维，切片（导出时记录在 `policy_metadata.json` 的 `critic_terms`）如下：

| 切片 | Critic 项 | 维度 | scale / clip | 说明 |
|---|---|---:|---|---|
| `[0:3]` | `base_lin_vel` | 3 | 2.0，无噪声 | 特权线速度，也是 estimator 的回归目标 `estimator_vel_slice=[0,3]` |
| `[3:6]` | `base_ang_vel` | 3 | 0.25 | 与 policy 同名项对齐 |
| `[6:9]` | `projected_gravity` | 3 | 1.0 | |
| `[9:12]` | `velocity_commands` | 3 | `(1, 1, 0.25)` | |
| `[12:24]` | `joint_pos` | 12 | 1.0 | |
| `[24:36]` | `joint_vel` | 12 | 0.05 | |
| `[36:48]` | `actions` | 12 | 1.0 | |
| `[48:50]` | `phase` | 2 | 无 | 与 policy 同一时钟 |
| — | `height_scan` | 0 | — | 全部 ATDog2 任务关闭该观测 |

Estimator 的 teacher 目标项由 `RslRlHimActorCriticCfg.estimator_target_term_names` 固定为 `base_ang_vel, projected_gravity, joint_pos, joint_vel, actions, base_lin_vel`，对应 critic 切片 `[[3,6],[6,9],[12,24],[24,36],[36,48],[0,3]]`，共 45 维。

### 2. 逐任务观测差异

`P` 表示上表 7 项、47 维的 policy 单帧（HIM 为 `P×6=282`；标准 PPO / cusrl 为 47），`C` 表示本节上表 8 项、50 维的 critic 组。每个任务的具体观测组内容如下：

| 任务 | policy 组 | critic 组 | `phase` / `velocity_commands` 实际语义 | 地形传感器与观测的关系 |
|---|---|---|---|---|
| Flat | `P`（47 / 282） | `C`（50） | 默认时钟 0.7/0.6/0.05；`heading_command=False`，yaw 直接采样 | `height_scanner`（187 射线）与 `height_scanner_base`（9 射线）仍挂载，但均不进入两个观测组；187 射线供摆腿净空等奖励，`base_height_l2.sensor_cfg=None` |
| Flat-DomainRand | `P`（47 / 282） | `C`（50） | 同 Flat；命令范围 vx −0.8~1.2、vy ±0.6，站立比例 0.05；域随机化只改数值分布 | 同 Flat；`base_height_l2` 重新指向 9 射线传感器（零权重，被清理） |
| Rough | `P`（47 / 282） | `C`（50） | 同 Flat；`rel_heading_envs=0` | 显式给 `critic.joint_pos/joint_vel` 指定 12 关节顺序；`terrain_levels` 课程不输入观测 |
| Stairs | `P`（47 / 282） | `C`（50），`height_scan` 再次显式置 `None` | `StairsVelocityCommandCfg`（heading off），时钟 0.5/0.55/0.07；相位频率、占空比与其他任务不同 | 187 射线供 `configure_stair_rewards()` 的地形查询；9 射线供 `StairsVelocityCommand` 的进度/抬升指标；`terrain_levels_stairs` 不输入观测 |
| Sand | `P`（47 / 282） | `C`（50） | 默认时钟；未显式关闭 `heading_command`（父类默认 `True`），yaw 由 heading 误差生成 | 187 射线供奖励，9 射线供 `base_height_l2`；都不是观测输入 |
| Slope | `P`（47 / 282） | `C`（50） | 同 Sand | 同 Sand；坡度变化不进入观测 |
| Bar | `P`（47 / 282） | `C`（50） | 同 Sand | `height_scanner_base=None`；`height_scanner` 由 `MySceneCfg()` 重新挂到 plane 上，仅供奖励/几何查询 |
| Tires | `P`（47 / 282） | `C`（50） | 同 Sand | 两个射线传感器都重新挂到 `base`；均不进入观测组 |

所有任务的 actor 都不接收真实机身线速度和地形高度：线速度只进 critic 并作为 estimator 回归目标；地形扫描不输入 actor 或 critic。当前各任务为 282/50 维，但跨任务仍需检查相位、缩放和 teacher 语义，不能仅凭维度相等判断可迁移（尤其 Stairs 与其他任务的相位频率不同、Sand/Slope/Bar/Tires 的 yaw 命令语义不同）。`align_him_critic_observations()` 在 HIM 启动时把 critic 的同名项替换为 policy 项的深拷贝并清除 critic 噪声（`him_clean_critic=True`），因此 HIM 下所有任务的 critic 关节顺序与 policy 一致；标准 PPO / cusrl 不调用该函数，critic 保留父类 `joint_names=".*"`（Rough 例外）。完整逐项切片和尺度见上一节「观测契约」与本节第 1 小节。

### 3. 三种训练器入口的策略结构

| 入口 | 类 | actor 输入 | critic 输入 | 网络 | 动作分布 |
|---|---|---|---|---|---|
| `rsl_rl_him_cfg_entry_point` | `HIMOnPolicyRunner` + `HIMActorCritic` / `HIMPPO` | 282 维历史（内部 actor 用 66 维） | 50 | estimator + actor `[512,256,128]` + critic `[512,256,128]`，ELU | `Normal(mean, std)`，`std` 为 12 个可学习参数、初值 1.0 |
| `rsl_rl_cfg_entry_point`（默认） | `OnPolicyRunner` + 标准 ActorCritic | 47 维单帧 | 50（特权组） | actor / critic `[512,256,128]`，ELU | `Normal`，`init_noise_std=1.0` |
| `cusrl_cfg_entry_point` | `cusrl.ActorCritic.Factory` | 47 维单帧 | 由 cusrl 环境包装决定 | MLP `[512,256,128]` + ELU（`ends_with_activation=True`），`cusrl.Value` | `cusrl.NormalDist` |

HIM 策略的关键结构（`him_actor_critic.py` / `him_estimator.py`）：

| 模块 | 输入 → 输出 | 说明 |
|---|---|---|
| estimator encoder | 282 → 128 → 64 → 19 | 输出前 3 维为预测机身线速度，后 16 维为 L2 归一化隐变量；`forward` 全程 `detach`，不向 actor 回传梯度 |
| estimator target | 45 → 128 → 64 → 16 | 由下一时刻 critic 的 teacher 切片生成目标隐变量，同样 L2 归一化 |
| actor | `47 + 3 + 16 = 66` → 512 → 256 → 128 → 12 | 输入为历史最后一帧（`actor_current_frame="last"`）+ 预测速度 + 隐变量；不是直接吃 282 维 |
| critic | 50 → 512 → 256 → 128 → 1 | 标准状态价值函数 |
| 原型层 | `Embedding(32, 16)` | SwAV 式 swap 预测，Sinkhorn 3 次迭代、温度 3.0 |

损失与优化：PPO surrogate（`clip_param=0.2`）、clipped value loss（`value_loss_coef=1.0`）、熵正则、`desired_kl=0.01` 自适应学习率、`max_grad_norm=1.0`；estimator 独立 Adam(`lr=1e-3`, `max_grad_norm=10.0`)，损失为 `1.0 × MSE(预测线速度, critic 真实线速度) + 0.1 × swap_loss`。训练时 actor/critic/std 用 Adam(`lr` 见下表)，`num_learning_epochs=5`、`num_mini_batches=4`、`gamma=0.99`、`lam=0.95`。

动作是 12 维关节位置偏移（`use_default_offset=True`），环境侧 scale 为 0.25（Flat/Flat-DomainRand 全部关节；其余 hip 0.125、thigh/calf 0.25），`clip_actions` 未启用，动作项 `clip=±100` 仅作兜底。

### 4. 逐任务策略配置

| 任务 | HIM runner | HIM：步数 / 迭代 / lr / entropy / schedule | 标准 PPO runner | PPO：步数 / 迭代 | cusrl trainer：步数 / 迭代 |
|---|---|---|---|---|---|
| Flat | `ATDogDog2FlatHIMRunnerCfg` | 48 / 1000 / 5e-4 / 0.01 / adaptive | `ATDogDog2FlatPPORunnerCfg` | 24 / 2000 | `ATDogDog2FlatTrainerCfg`：24 / 5000 |
| Flat-DomainRand | `ATDogDog2FlatDomainRandHIMRunnerCfg` | 48 / 1000 / 5e-5 / 0.003 / fixed | `ATDogDog2FlatPPORunnerCfg` | 24 / 2000 | `ATDogDog2FlatTrainerCfg`：24 / 5000 |
| Rough | `ATDogDog2RoughHIMRunnerCfg` | 24 / 20000 / 1e-3 / 0.01 / adaptive | `ATDogDog2RoughPPORunnerCfg` | 24 / 2000 | `ATDogDog2RoughTrainerCfg`：24 / 20000 |
| Stairs | `ATDogDog2StairsHIMRunnerCfg` | 48 / 20000 / 5e-5 / 0.003 / fixed | `ATDogDog2StairsPPORunnerCfg` | 24 / 20000 | `ATDogDog2StairsTrainerCfg`：24 / 20000 |
| Sand | `ATDogDog2SandHIMRunnerCfg` | 24 / 20000 / 1e-3 / 0.01 / adaptive | `ATDogDog2SandPPORunnerCfg` | 24 / 20000 | `ATDogDog2SandTrainerCfg`：24 / 20000 |
| Slope | `ATDogDog2SlopeHIMRunnerCfg` | 24 / 20000 / 1e-3 / 0.01 / adaptive | `ATDogDog2SlopePPORunnerCfg` | 24 / 20000 | `ATDogDog2SlopeTrainerCfg`：24 / 20000 |
| Bar | `ATDogDog2BarHIMRunnerCfg` | 24 / 5000 / 1e-3 / 0.01 / adaptive | `ATDogDog2BarPPORunnerCfg` | 24 / 5000 | `ATDogDog2BarTrainerCfg`：24 / 5000 |
| Tires | `ATDogDog2TiresHIMRunnerCfg` | 24 / 20000 / 1e-3 / 0.01 / adaptive | `ATDogDog2TiresPPORunnerCfg` | 24 / 2000 | `ATDogDog2TiresTrainerCfg`：24 / 20000 |

说明：

- HIM 的 `save_interval`：Flat / Flat-DomainRand 50，其余 100；`max_iterations` 可被 `--max_iterations` 覆盖。
- 标准 PPO 三种任务的 `lr=1e-3`、`entropy_coef=0.01`、`schedule="adaptive"` 全部相同，仅迭代上限不同。
- cusrl 各任务统一 `AdamW(lr=1e-3)`、`epochs=5`、`mini_batches=4`、`EntropyLoss(0.008)`、`AdaptiveLRSchedule(kl=0.01)`、`gamma=0.99`、`lam=0.95`。
- 两个 rsl_rl 入口共用 `clip_param=0.2`、`gamma=0.99`、`lam=0.95`、`desired_kl=0.01`、`max_grad_norm=1.0`、`value_loss_coef=1.0`、`use_clipped_value_loss=True`，且显式关闭观测归一化（`actor_obs_normalization=False`、`critic_obs_normalization=False`）；cusrl 的配置未启用归一化 hook。

### 5. 部署侧输入契约

`play.py` 导出的 `policy.pt` / `policy.onnx` 把 estimator encoder 与 actor 融合成一个模块：输入名 `obs_history`、形状 `[1, 282]`，输出名 `actions`、形状 `[1, 12]`。也就是说部署端只需喂 6 帧 47 维历史，不需要 critic 的 50 维；模块内部自行计算 3 维预测速度与 16 维隐变量，拼成 66 维 actor 输入。

`policy_metadata.json` 记录 `critic_obs_dim=50`、`one_step_obs_dim=47`、`history_length=6`、`history_order=oldest_to_latest`、`history_indices_latest_to_oldest=[5,4,3,2,1,0]`、`actor_current_frame=last`、`estimator_vel_slice=[0,3]`、`estimator_target_slices`、各 policy/critic term 的切片与缩放、以及步态时钟 `gait_clock`；`gait_clock.py`、`at_robot_lab_history.yaml` 供部署端重建相位。导出时 `enable_corruption=False`，训练期的观测噪声不进入部署策略。

## DogW1 轮足机器人：训练、网络与导出（2026-11-24）

DogW1 是在 Dog2 基础上增加四个驱动轮的轮足机器人。资产定义为 `source/robot_lab/robot_lab/assets/atdog.py` 中的 `AT_DOG_W1_CFG`；URDF 为 `source/robot_lab/data/Robots/atdog/dog_w1/urdf/dog_w1.urdf`。机身名称为 `base`。腿部初始角度与 Dog2 一致：hip=0、左 thigh=0.8、右 thigh=−0.8、calf=0；四轮初始角度及全部关节初始速度均为 0，参考 ATDog 轮子配置。初始机身高度为 0.35 m。

独立任务目录为 `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/dog_w1/`。五个环境分别直接基于通用 `LocomotionVelocityRoughEnvCfg`，不继承 ATDog、Dog2 或其他 DogW1 地形任务；各自文件完整定义动作、观测、奖励、地形和课程设置。`agents/` 内的 PPO、HIM、CUSRL 配置也分别定义每个任务，不继承其他机器人的训练配置。相位实现位于本目录的 `phase_observations.py`，相位步态目标位于 `rewards.py`，轮角度屏蔽位于 `observations.py`。

### 1. 注册任务与训练入口

| 地形 | 完整任务 ID | 环境文件 | HIM 实验目录名 | 默认迭代数（HIM / PPO / CUSRL） |
| --- | --- | --- | --- | --- |
| 平地 | `RobotLab-Isaac-Velocity-Flat-ATDog-DogW1-v0` | `flat_env_cfg.py` | `atdog_dog_w1_flat_him` | 5000 / 5000 / 5000 |
| 粗糙地形 | `RobotLab-Isaac-Velocity-Rough-ATDog-DogW1-v0` | `rough_env_cfg.py` | `atdog_dog_w1_rough_him` | 20000 / 20000 / 20000 |
| 坡地 | `RobotLab-Isaac-Velocity-Slope-ATDog-DogW1-v0` | `slope_env_cfg.py` | `atdog_dog_w1_slope_him` | 20000 / 20000 / 20000 |
| 沙地入口 | `RobotLab-Isaac-Velocity-Sand-ATDog-DogW1-v0` | `sand_env_cfg.py` | `atdog_dog_w1_sand_him` | 20000 / 20000 / 20000 |
| 楼梯 | `RobotLab-Isaac-Velocity-Stairs-ATDog-DogW1-v0` | `stairs_env_cfg.py` | `atdog_dog_w1_stairs_him` | 20000 / 20000 / 20000 |

`Sand` 沿用原 ATDog 的任务名称，当前实际地形为重复长条箱体障碍，不是可变形沙土：16 个箱体，高 0.30 m，平面尺寸 0.03×1.6 m，起终难度参数一致。`Slope` 坡度范围为 0～0.3（坡度比）；`Stairs` 为正向/倒向金字塔楼梯，台阶高 0.09～0.11 m、宽 0.30 m。`Flat` 关闭地形课程；其余保留地形等级课程，各任务关闭指令等级课程。任务定义和参数应直接在对应地形文件修改。

每个任务都注册以下三个训练入口：

| 入口 | 文件 | 类名格式 |
| --- | --- | --- |
| `rsl_rl_him_cfg_entry_point` | `agents/rsl_rl_him_cfg.py` | `ATDogDogW1{地形}HIMRunnerCfg` |
| `rsl_rl_cfg_entry_point` | `agents/rsl_rl_ppo_cfg.py` | `ATDogDogW1{地形}PPORunnerCfg` |
| `cusrl_cfg_entry_point` | `agents/cusrl_ppo_cfg.py` | `ATDogDogW1{地形}TrainerCfg` |

以下命令沿用本文容器路径；在本机运行时，把 Isaac Lab 启动器路径和仓库路径替换为实际位置。先进入仓库：

```bash
cd /workspace/isaaclab_extension_template
```

HIM 平地训练：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-DogW1-v0 \
  --agent=rsl_rl_him_cfg_entry_point \
  --num_envs=4096 --headless
```

训练其他地形时，将任务 ID 中的 `Flat` 替换成 `Rough`、`Slope`、`Sand` 或 `Stairs`。可使用 `--max_iterations=1000` 覆盖默认迭代数。HIM checkpoint 保存在 `logs/rsl_rl/atdog_dog_w1_<地形>_him/<运行目录>/model_<迭代>.pt`。

从同一任务的 checkpoint 恢复训练，下面的运行目录和文件名须替换为实际值：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-DogW1-v0 \
  --agent=rsl_rl_him_cfg_entry_point \
  --resume --load_run=2026-11-24_12-00-00 --checkpoint=model_4900.pt \
  --num_envs=4096 --headless
```

从 DogW1 平地权重开始新的楼梯训练（不恢复旧优化器和迭代计数）：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Stairs-ATDog-DogW1-v0 \
  --agent=rsl_rl_him_cfg_entry_point \
  --him_init_checkpoint=logs/rsl_rl/atdog_dog_w1_flat_him/2026-11-24_12-00-00/model_4900.pt \
  --num_envs=4096 --headless
```

各任务当前使用相同的动作/观测布局；修改相位语义、关节顺序或输入缩放后，需要重新核对 checkpoint 的观测契约。Dog2 的 12 动作、47 维单帧模型不能直接作为 DogW1 的 16 动作、59 维单帧模型加载。

标准 PPO 与 CUSRL 平地训练：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-DogW1-v0 \
  --agent=rsl_rl_cfg_entry_point --num_envs=4096 --headless

/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/cusrl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-DogW1-v0 \
  --agent=cusrl_cfg_entry_point --num_envs=4096 --headless
```

### DogW1 16 电机实测正反转摩擦

全部 12 个腿电机和 4 个轮电机采用一阶滑动摩擦模型，`|τf|=a·|ω|+b`，摩擦力矩始终与关节速度方向相反：

| 关节速度方向 | a：黏性系数，Nm/(rad/s) | b：库仑摩擦力矩，Nm |
| --- | --- | --- |
| 正转 ω>0 | 0.0107065786 | 0.231393197 |
| 反转 ω<0 | 0.0122314048 | 0.21220256 |

参数保存在 `robot_lab/actuators.py` 的 `FrictionDelayedPDActuatorCfg` / `FrictionImplicitActuatorCfg`，由 `assets/atdog.py::AT_DOG_W1_CFG` 选择。正反方向按 URDF 关节速度坐标判断，不是机身前进/后退。速度为零时取摩擦力矩为零；该滑动拟合没有定义静摩擦/起动摩擦。

腿部保留 PD stiffness=25、damping=0.5 及 1～2 个物理步延迟，在电机输出限幅后叠加被动摩擦；轮子保留隐式速度驱动 stiffness=0、damping=0.8，通过外加关节力矩施加摩擦。PD damping 是控制器 KD，不能替代实测的 a。DogW1 旧关节 friction 设置归零，避免重复计算，其他 ATDog 型号不变。新参数只对重新创建的环境生效，正在运行的训练不会自动更新。

### 2. 动作与 actor / critic 观测排布

物理步长 0.005 s，控制降采样 4，即控制周期 0.02 s、50 Hz。动作总共 16 维，腿部走位置控制、轮子走速度控制：

| 动作切片 | 关节顺序 | 缩放及控制 |
| --- | --- | --- |
| `[0:12]` | FR hip/thigh/calf、FL hip/thigh/calf、RR hip/thigh/calf、RL hip/thigh/calf | 相对初始姿态的位置目标；hip 缩放 0.125，其余 0.25；Dog2 的 12 个独立延迟 PD 电机，stiffness=25、damping=0.5、friction=0.5、delay=1～2 个物理步 |
| `[12:16]` | FR、FL、RR、RL 的 `foot_joint` | 轮速目标缩放 5.0；参考 ATDog 的隐式速度驱动，stiffness=0、damping=0.8、friction=0、effort_limit_sim=23.5、velocity_limit_sim=30 |

Actor 在 ATDog 57 维观测基础上增加两个相位分量，并按 Dog2 的 phase-first 顺序排列，单帧 59 维：

| 切片 | 观测项 | 维度 | 缩放 / 说明 |
| --- | --- | --- | --- |
| `[0:2]` | `phase` | 2 | `[sin(2πφ), cos(2πφ)]`，缩放 1 |
| `[2:5]` | `base_ang_vel` | 3 | 机身角速度，缩放 0.25 |
| `[5:8]` | `projected_gravity` | 3 | 重力投影，缩放 1 |
| `[8:11]` | `velocity_commands` | 3 | vx、vy、yaw 速度命令，缩放 1；沿用 ATDog 命令观测 |
| `[11:27]` | `joint_pos` | 16 | 按动作顺序排列的相对初始角度；四个轮角度在加噪前置零，保留这四个观测位置 |
| `[27:43]` | `joint_vel` | 16 | 按动作顺序排列的相对初始速度，缩放 0.05 |
| `[43:59]` | `actions` | 16 | 上次动作，缩放 1 |

HIM 使用 6 帧，总输入 354 维，从旧到新排列，当前帧为 `[295:354]`。标准 PPO / CUSRL 使用 59 维单帧。Actor 不输入真实线速度或地形扫描。训练 policy 保留观测噪声，play / 导出关闭观测噪声。

Critic 延续 Dog2 的字段排布：真实线速度在最前，phase 在 actions 后，共 62 维，不展开历史：

| 切片 | 观测项 | 维度 | 缩放 |
| --- | --- | --- | --- |
| `[0:3]` | `base_lin_vel` | 3 | 2.0；同时作为 HIM 速度监督目标 |
| `[3:6]` | `base_ang_vel` | 3 | 0.25 |
| `[6:9]` | `projected_gravity` | 3 | 1 |
| `[9:12]` | `velocity_commands` | 3 | 1 |
| `[12:28]` | `joint_pos` | 16 | 1，轮角度屏蔽 |
| `[28:44]` | `joint_vel` | 16 | 0.05 |
| `[44:60]` | `actions` | 16 | 1 |
| `[60:62]` | `phase` | 2 | 1 |

五个任务均使用这套 59 / 62 维观测，不向 critic 添加地形扫描或额外接触力。HIM 入口调用 `align_him_critic_observations()`，复制 policy 同名项的函数、关节选择及缩放；DogW1 设置 `him_clean_critic=True`，随后清除 critic 噪声。轮关节的绝对 articulation ID 会先转换为观测列索引，再将轮角度置零，避免显式关节重排后屏蔽错列。

相位使用 Dog2 的命令门控 sin/cos 约定，在 DogW1 本地独立实现。当前周期 0.5 s；当平移命令模长或 yaw 命令绝对值达到 0.05 时，每个控制步推进一次 `φ=(φ+0.02/0.5)%1`。同一步重复读取不会重复推进；静止命令时将 φ 清零，输出 `[0,1]`；每个环境 reset 时也清零。虽然配置保留 `stop_cycle_time` / `stop_duration` 参数，当前实现没有额外停止过渡，它们不改变相位输出。

### 相位与接触奖励修正（2026-11-24）

DogW1 五类任务均改用本目录 `rewards.py` 的相位步态目标，奖励参数在各自的 `*_configure_rewards()` 中独立设置。Actor、critic、接触奖励和摆动高度奖励共享环境内同一个 `DogW1Phase` 状态，每个控制步只推进一次；局部 reset 清零相同环境的时钟。奖励的周期和命令门限直接复制自该任务的 policy phase 配置，配置不一致时会报错，避免奖励与输入悄然错相。

- 原 `mdp.feet_contact()` 统计本步新触地数量，并在数量“不等于预期”时返回 1；正权重会奖励不符合预期。DogW1 的 `feet_contact` 已替换为 `PhaseContactReward`：根据持续接触时间判断当前支撑状态，按 FL、FR、RL、RR 排列，计算每轮接触状态与该轮相位目标的一致率。取值范围 0～1，保持正权重。不能仅靠支撑数量相同获得满分，支撑轮必须匹配相位指定的对角组。
- `feet_contact_without_cmd` 改为 `standing_contact`：静止时按持续支撑轮数返回 0～4，保持该任务原来的正权重。仅原地转向也视为运动，不触发静止奖励。两者命令门限统一为平移模长 0.05、yaw 绝对值 0.05。
- 默认 trot：FL/RR 同相，FR/RL 偏移 0.5 周期；周期 0.5 s，`stance_ratio=0.55`。每轮前 55% 周期目标为支撑，后 45% 为摆动，摆动间隙为 `0.08*sin(π*swing_progress)` m。0.55 的支撑比例允许切换阶段短暂四轮支撑。
- 新增 `phase_swing_clearance`，权重 −160：只对摆动轮的实际离地间隙与相位目标的均方误差扣分。轮心高度扣除轮半径 0.08 m 后计算间隙；平地使用环境原点地面高度，其他四类任务在各轮下方射线查询局部地形高度，无效射线屏蔽。`swing_height` 和 `foot_radius`（轮半径）都可在各任务文件单独调整。
- 关闭原 `feet_gait`、`feet_air_time`、`feet_air_time_variance`、`feet_height`、`feet_height_body`，由上述相位目标负责接触/抬轮协调，避免 0.35 s 腾空阈值及固定足高与 0.5 s 相位目标冲突。其余用户配置的奖励权重保留。

相位步态配置格式与 Dog2 对齐：各地形独立定义 `<地形>_configure_dog_w1_phase_swing_reward(cfg)`，用 `RewTerm(func=PhaseSwingClearance, weight=..., params={...})` 显式列出 `phase_offset=0.5`、`stance_ratio=0.55`、`contact_smoothing=0.07`、`foot_radius=0.08`、`swing_height=0.08` 和 `turn_height_bonus=0.02`。周期、停止参数和门限从该任务的 policy phase 读取。`<地形>_configure_rewards()` 先调用相位配置，再调用独立权重配置，并把偏移、占空比、平滑参数同步到接触奖励；不继承或导入 Dog2 配置。接触切换使用 Dog2 同形式的 sigmoid 平滑窗口，接触奖励为 `1-mean((actual_contact-desired_contact)^2)`；摆动高度误差按 `1-desired_contact` 加权。原地转向（yaw 绝对值大于 0.2，平移小于启动门限）时在摆动最大高度上增加 0.02 m。轮半径使用与 Dog2 相同的字段名 `foot_radius`，保持 DogW1 自己的 0.08 m 数值。

本次修正在 DogW1 本地实现，其他机器人的旧 `mdp.feet_contact` 配置保持原行为。Actor 59 维单帧、HIM 354 维历史、critic 62 维和 estimator teacher 57 维不变。URDF 与奖励语义已经变化，建议开新训练目录验证，不把旧 run 的总奖励与新 run 直接比较。

无仿真数值验证：`python3 -m unittest discover -s tests -p 'test_dog_w1*.py' -v`。测试覆盖持续接触、错误对角组、相位半周期切换、共享时钟、局部 reset、静止/转向、轮半径与局部地形间隙、无效射线及五类任务的配置接线。未安装 PyTorch 时测试使用 NumPy 适配器计算；完整 Isaac Lab 检查仍使用 `tests/check_dog_w1_sim.py`。

### Flat 5000 次训练复盘后的参数修订（2026-11-24）

本小节为 DogW1 Flat 当前配置，覆盖前文相应旧参数；其他地形不变。完整5000次日志分析、窗口对比、修改表和曲线见 [训练复盘](outputs/dog_w1_5000_review/review.md)。

- Flat 相位接触启用 `centered=True`，使用 `mean((2*actual_contact-1)*(2*desired_contact-1))`，范围−1～1，权重+15；其他地形默认保持原0～1打分。四轮持续着地不再稳定获得约半额奖励。新增独立横向速度平方误差项−20，vx/vy命令范围仍为±1 m/s。
- Flat 高度权重改为−500，目标保持0.36 m；姿态角速度−3，轮力矩−0.1，轮速L2−0.005，关节位置偏差−1。关闭朝上正奖励、轮速绝对值方差及raw关节镜像项，保留flat_orientation惩罚。phase_swing_clearance权重−1500，轮半径0.08 m、swing_height=0.05 m、turn_height_bonus=0.03 m；周期和观测维度不变。
- 初始roll/pitch范围±0.15 rad，COM三轴±0.01 m，PD增益乘0.9～1.1，reset外力/力矩±2，关闭间隔推搡。启用机身base触地终止和倾角大于1 rad终止。1000步回合不再是必然结果，应结合失败率与固定命令误差判断。
- 仅 Flat HIM runner 改为 `init_noise_std=0.6`、`entropy_coef=0.003`，其余网络尺寸、学习率、rollout和迭代数保持。建议用 `--run_name=phase_lateral_fix` 开新训练目录，从头验证；新旧总reward不适合直接比较。

已通过21项配置/数值检查，真实仿真与新训练效果尚未验证。配置修改不是已证实的性能提升，下一轮需逐轮比较期望/实际接触和间隙，并观察横向控制是否形成。

### 3. 各任务的网络与训练参数

| 任务 | HIM actor 输入 → 输出 | 标准 PPO / CUSRL actor 输入 → 输出 | critic 输入 → 输出 | actor / critic 隐藏层 |
| --- | --- | --- | --- | --- |
| Flat | 354 维历史，经估计器组合为 78 → 16 | 59 → 16 | 62 → 1 | 各自 `[512,256,128]` |
| Rough | 354 维历史，经估计器组合为 78 → 16 | 59 → 16 | 62 → 1 | 各自 `[512,256,128]` |
| Slope | 354 维历史，经估计器组合为 78 → 16 | 59 → 16 | 62 → 1 | 各自 `[512,256,128]` |
| Sand | 354 维历史，经估计器组合为 78 → 16 | 59 → 16 | 62 → 1 | 各自 `[512,256,128]` |
| Stairs | 354 维历史，经估计器组合为 78 → 16 | 59 → 16 | 62 → 1 | 各自 `[512,256,128]` |

所有网络使用 ELU。RSL-RL PPO / HIM 的输出为 16 维高斯均值，16 个可学习标准差初值为 1.0；actor / critic 均关闭运行均值方差归一化，使用观测项的固定缩放。CUSRL 使用 `Mlp.Factory`、`NormalDist.Factory`，actor / critic backbone 的 `ends_with_activation=True`，优化器为 AdamW。

HIM 的估计器与网络尺寸：

| 模块 | 输入 → 隐藏层 → 输出 | 用途 |
| --- | --- | --- |
| Student encoder | 354 → 128 → 64 → 19 | 历史编码：3 维估计线速度 + 16 维 latent |
| Teacher target 网络 | 57 → 128 → 64 → 16 | 编码下一步 critic 特征，生成对比学习目标 |
| Actor | 78 → 512 → 256 → 128 → 16 | 当前帧 59 + 估计速度 3 + latent 16 |
| Critic | 62 → 512 → 256 → 128 → 1 | 价值估计 |

Teacher 选项与 Dog2 一致：依次读取 `base_ang_vel`、`projected_gravity`、`joint_pos`、`joint_vel`、`actions`、`base_lin_vel`，共 `3+3+16+16+16+3=57` 维，不使用 phase 和命令作为 teacher target。对应 critic 切片为 `[3:6]、[6:9]、[12:28]、[28:44]、[44:60]、[0:3]`；速度回归单独读取 `[0:3]`。这些切片由 HIM wrapper / runner 根据实际字段布局推导，不在任务代码中写死。估计器 lr=1e-3、max_grad_norm=10、prototype 数=32、temperature=3.0；速度损失系数=1.0，swap 损失系数=0.1。

五个任务的 RSL-RL PPO / HIM 均为每次 rollout 24 步，保存间隔 100，学习率 1e-3，adaptive schedule，entropy_coef=0.01，5 epochs、4 minibatches，gamma=0.99、lambda=0.95、clip=0.2、desired_kl=0.01、max_grad_norm=1.0。Flat 默认 5000 次迭代，其余 20000。CUSRL 每次更新 24 步、保存间隔 100，lr=1e-3、entropy 权重 0.008，5 epochs、4 minibatches，其余详见对应 `TrainerCfg`。

### 4. 评估与导出教程

将 `--checkpoint` 路径替换为实际文件。以下以 HIM 平地模型为例，只导出并退出：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-DogW1-v0 \
  --agent=rsl_rl_him_cfg_entry_point \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog_w1_flat_him/2026-11-24_12-00-00/model_4900.pt \
  --num_envs=1 --headless --export_only
```

需要观察仿真时，去掉 `--headless --export_only`，可把 `--num_envs` 改为 16。其他地形同时替换任务 ID 和 checkpoint 路径。标准 PPO 使用相同 play 脚本，但将 `--agent` 改成 `rsl_rl_cfg_entry_point`，并选择对应 PPO checkpoint。

HIM 的导出文件位于 checkpoint 同级的 `exported/`：

| 文件 | 内容 |
| --- | --- |
| `policy.pt` | TorchScript，融合 estimator encoder 与 actor |
| `policy.onnx` | ONNX opset 17，输入 `obs_history` 为 float32 `[1,354]`，输出 `actions` 为 `[1,16]` |
| `policy_metadata.json` | 59 维单帧、6 帧历史、62 维 critic、16 动作、各项布局/缩放、teacher 切片、观测契约、周期等 |
| `at_robot_lab_history.yaml` | 部署侧历史配置片段：`num_observations=59`、`expected_policy_input_dim=354`、`observations_history=[5,4,3,2,1,0]` |

部署侧按从旧到新顺序构造六帧，每帧严格遵循上面的 59 维布局，当前帧放最后；模型内部自行估计 3 维速度和 16 维 latent，部署输入不包含 critic 或真实线速度。模型输出仍是动作，需要按上面的 12 个位置目标和 4 个轮速目标进行缩放、偏置和关节映射。标准 PPO 导出输入为 59 维单帧、输出为 16 维，不包含 HIM 估计器与历史。

DogW1 当前使用本地 `DogW1Phase`，导出信息含 `gait_cycle_time=0.5`；它没有命令模块的 `gait_clock`，因此当前不会额外导出 Dog2 的 `gait_clock.py`。部署端应按本节门控、更新和 reset 规则重建相位；仅配置一个不受命令门控的正弦周期不能复现训练输入。完整参数可核对 `policy_metadata.json` 的 `observation_contract` 与本地 `phase_observations.py`。

### 5. 配置检查与仿真检查

无需 Isaac Sim 的配置检查覆盖 URDF 关节/电机匹配、初始化一致性、mesh 路径、注册入口、独立配置、观测顺序、teacher 切片和轮角度屏蔽：

```bash
python3 -m unittest discover -s tests -p test_dog_w1_configs.py -v
```

在已安装 Isaac Lab / Isaac Sim 的环境中运行以下检查，加载五套任务配置，再用四个平地环境验证 354 / 59 / 62 / 16 维接线、估计器更新与 JIT 导出一致性：

```bash
/workspace/isaaclab/isaaclab.sh -p tests/check_dog_w1_sim.py --headless
```

本次编辑环境已通过配置检查；该环境未安装 Isaac Lab / PyTorch，仿真、实际训练和模型导出检查尚未执行。

## 开始训练

```bash
cd /workspace/isaaclab_extension_template
/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-Dog2-v0 \
  --agent=rsl_rl_him_cfg_entry_point \
  --num_envs=6500 \
  --max_iterations=1000 \
  --headless

cd /workspace/isaaclab_extension_template
/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-Arm-v0 \
  --num_envs=10000 \
  --max_iterations=2000\
  --headless \
  --resume \
  --load_run=2026-06-12_13-55-44 \
  --checkpoint=model_39600.pt 

/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Slope-ATDog-Dog3-v0 \
  --num_envs=30000 \
  --max_iterations=5000\
  --headless \
  --resume \
  --load_run=2026-06-13_07-04-36\
  --checkpoint=model_18400.pt 


/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-Dog_Arm-v0 \
  --num_envs=10000 \
  --max_iterations=2000\
  --headless



/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-Arm-v0 \
  --num_envs=6000 \
  --max_iterations=2000\
  --headless \
  --resume \
  --load_run=2026-06-14_11-56-28 \
  --checkpoint=model_41599.pt

/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Slope-ATDog-Dog3-v0 \
  --num_envs=4000 \
  --max_iterations=2000\
  --headless \
  --resume \
  --load_run=2026-06-14_00-27-05 \
  --checkpoint=model_11400.pt 

```

## 导出模型：

```bash
cd /workspace/isaaclab_extension_template

 /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py   --task=RobotLab-Isaac-Velocity-Flat-ATDog-Dog2-v0   --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog2_flat_him/2026-10-22_08-46-12/model_2450.pt     --num_envs=50  --agent=rsl_rl_him_cfg_entry_point 
  


/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-Arm-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog2_arm_rough/2026-06-14_13-10-55/model_43598.pt \
  --num_envs=1\
  --headless

/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-ATDog-Dog3-Arm-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog3_arm_rough/2026-06-10_14-05-13/model_38986.pt \
  --num_envs=1\
  --headless



/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-Dog2-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog_flat/2026-05-31_14-32-45/model_4999.pt\
  --num_envs=10

/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Slope-ATDog-Dog2-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog2_slope_him/2026-09-12_11-58-40/model_15900.pt \
  --num_envs=10\
  --agent=rsl_rl_him_cfg_entry_point 
```

## 复制导出的模型到宿主机

初始：2026-05-23_14-26-00

scp -P 3022 sw@shenweitechnology.com:/home/sw/code/AT_rl_sar/logs/rsl_rl/atdog_dog2_flat/2026-06-07_18-30-45/exported/policy.pt ~/桌面/

scp -P 3022 sw@shenweitechnology.com:/home/sw/code/AT_rl_sar/logs/rsl_rl/atdog_dog2_arm_rough/2026-06-14_03-42-01/exported/policy.pt ~/桌面/

scp -P 3022 sw@shenweitechnology.com:/home/sw/code/AT_rl_sar/logs/rsl_rl/atdog_dog3_arm_rough/2026-06-10_14-05-13/exported/policy.pt ~/桌面/


scp -P 3022 sw@shenweitechnology.com:/home/sw/code/AT_rl_sar/logs/rsl_rl/atdog_dog2_arm_rough/2026-06-10_11-04-22 ~/桌面/

```bash
docker cp robot-lab:/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog2_arm_rough/2026-06-10_11-04-22  robot-lab:/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog3_arm_rough/

docker cp ~/桌面/2026-06-12_13-55-44 robot-lab:/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog2_arm_rough/


cp atdog_dog2_arm_rough/2026-06-10_11-04-22  atdog_dog3_arm_rough/
```





## ATDog2 Critic 扫描关闭及 checkpoint 输入迁移（2026-11-14）

统一配置入口 `dog2/periodic_gait.py::configure_periodic_gait` 将
`observations.critic.height_scan=None`。覆盖 Flat、Flat-DomainRand、Rough、Stairs、
Sand、Bar、Tires、Slope，Stairs 不再重新添加扫描观测。Actor 仍为单帧 47 维、
HIM 六帧 282 维；Critic 从历史 237 维变为 50 维。

这是关闭 Critic 扫描输入，不是删除射线传感器。足端净空、台阶专用五项奖励和相关
地形指标仍使用地形查询，不能直接删除传感器。`lazy_sensor_update=True` 时这些奖励每步访问
`height_scanner.data`，187 条网格射线仍会每步计算；关闭观测项主要省去观测组里的裁剪/缩放/拷贝，
实际迭代提速需测量。

### HIM resume / checkpoint

`HIMOnPolicyRunner.save()` 保存 `checkpoint_format_version=2`、模型、两个优化器、
迭代计数及完整 `observation_contract`。旧 checkpoint 只要包含可验证的观测契约，
无需重写文件即可通过原 `--resume --load_run=...` 自动迁移 Critic 输入。
训练 resume、`--him_init_checkpoint` 和 HIM play 使用相同迁移逻辑。

- 依据观测项名称及完整定义（维度、函数、缩放、裁剪、噪声、参数等）匹配第一层输入列，
  支持增删、重排、同维度更换和非 teacher 项维度变化；同名同定义列保留。
- 删除项对应列丢弃；新增或定义变化项的列置零。Critic 后续层和 bias 保留。
  不按总维度或前缀盲目截取；同维度重排也会迁移。
- Actor/历史布局、HIM teacher 字段定义和 teacher 目标顺序必须兼容。
  `base_lin_vel` 与 teacher 项不可静默删除或改义；允许其索引随 Critic 重排变化，
  但切片必须与契约声明一致。网络隐藏层尺寸变化不在此次迁移范围内。
- Critic 输入变化时 PPO 优化器状态清空；teacher 语义未变时 estimator 权重和优化器保留。
  普通 resume 保留 checkpoint 迭代计数；`--him_init_checkpoint` 的 weights-only 模式
  从第 0 轮开始且不恢复两个优化器。完全同契约仍按原方式恢复两个优化器。
- 旧相位/critic 噪声语义迁移仍须显式 `--him_allow_observation_migration`，
  该选项不允许任意 Actor/teacher 语义变化。
- 没有观测契约的历史文件仍需要 `--him_allow_legacy_checkpoint`，且无法安全推断
  Critic 输入列的含义；不提供无契约的自动输入维度迁移。

移除扫描后价值估计需要重新适应，不等价于数值完全连续的恢复；Actor 权重不会因此重置。
新保存的 checkpoint 会记录当前 50 维契约，后续同配置恢复无需再次迁移。
标准 RSL-RL PPO 训练保留原有按参数形状回退加载流程；它没有 HIM 观测契约，不能承诺
按观测语义重排 Critic 列。上述自动语义迁移专用于本项目 HIM runner。

验证记录：10 项 checkpoint 数值/加载测试通过；使用现有 `2026-11-13_23-22-21/model_5700.pt`
验证 Critic 237→50 列迁移及 Actor/estimator 权重保留，未改写源 checkpoint。未为本次改动重启训练。

## ATDog2 台阶迭代耗时优化与实测（2026-11-14）

硬件采样：RTX 5060 Laptop 8 GB，显存约 6.7–6.9 GB，GPU 利用率多数 62–82%，
功耗约 50 W、温度约 56–62°C。`nvidia-smi -q` 报告当前功耗上限为 50 W，
`SW Power Cap` 生效；没有更改功耗上限、CPU powersave governor 或系统频率配置。

本次保持 4000 环境、48 步/环境/迭代（192000 transitions）、PPO epochs/mini-batches、
物理 dt/decimation、足端 72 条射线的几何精度不变：

1. 五项台阶奖励不再访问 `height_scanner.data`：直接使用已初始化的 mesh，
   初始支撑高度兜底复用四足地面高度，不再为此刷新 187 条 Critic 网格射线。
2. 新增 `robot_lab/terrain_raycast.py`，使用上游同一 Warp kernel，显式绑定
   PyTorch 当前 CUDA stream，避免每次射线调用后的设备级 `wp.synchronize()`。
   CPU/跨设备输入仍回退到上游 wrapper；未修改 Isaac Lab 安装。
3. 缓存采样偏置/索引/mesh，摆腿锁定和台阶指标初始化改用固定形状选择，减少
   CUDA 布尔索引同步及每步常量分配。

每组均取运行末尾 20 轮，以下是顺序训练测量，受地形分布、负载与频率变化影响，
不能将全部差异归因于单一改动：

|阶段/运行目录|平均采集|平均学习|平均迭代|迭代范围|
|---|---:|---:|---:|---:|
|改动前 `2026-11-14_04-21-56`|7.334 s|0.844 s|8.178 s|7.758–9.330 s|
|移除多余扫描/布尔索引同步 `2026-11-14_12-41-06`|6.780 s|0.837 s|7.617 s|7.507–7.750 s|
|再启用同 stream 射线 `2026-11-14_13-01-41`|6.690 s|0.834 s|7.524 s|7.432–7.623 s|

总迭代平均缩短约 8%，吞吐从约 2.35 万提高到约 2.55 万 transitions/s。
`ROBOT_LAB_PROFILE=1` 的分段计时显示环境仍是主要开销；Actor 和 post 合计约 0.10 s。
当前训练从上一轮已保存的 `model_9700.pt` 恢复，日志在容器
`/tmp/atdog2_speed_async_training.log`。这些优化不改变 Actor/Critic 输入维度，
可继续使用既有 checkpoint。机身高度首次初始化兜底的地形参考略有变化，已在奖励设计中说明。

验证：10 项台阶奖励测试（含禁止读取网格 scanner 的回归检查）通过；
`tests/check_terrain_raycast_stream.py` 的 16 批随机 CUDA 查询在默认和非默认 stream
下与上游的命中点/法线一致；语法与 `git diff --check` 通过。

### 当前运行日志驱动的权重修订（2026-11-15）

读取运行中 `2026-11-15_00-57-37` 的 1121–1220 次窗口：
线速度误差 .3415、楼梯进展 .9883 m、成功率 31.20%、非法接触 23.63%；
joint_acc 惩罚约 −38.45，明显大于五项楼梯惩罚。
默认 Dog2 楼梯任务调整 joint_acc −1e−5→−7.5e−6、yaw tracking +50→+25、
越边 −3→−5、落足 −2→−3、支撑稳定 −1→−2。
保留线速度、动态摆腿/机身高度、动作平滑及接触/姿态保护。
当前进程不热更新，改动下次新启动/resume 生效；未打断现有训练。
不以改权重后的总奖励判断改善；需固定高度验证成功率和摔倒率。
详见 `docs/ATDOG2_REWARD_HIM_AUDIT.md` 的本次快照与调整记录。
