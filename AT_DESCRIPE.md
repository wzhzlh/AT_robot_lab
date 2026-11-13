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

当前 ATDOG2 的 actor 使用相位 sin/cos、机身角速度、重力投影、速度命令、关节位置、关节速度和上次动作，单帧共 47 维；HIM 使用 6 帧历史，共 282 维，按从旧到新排列，当前帧位于最后。Actor 不输入真实机身线速度及地形扫描，critic 保留这些特权信息。默认控制频率为 50 Hz，步态周期为 0.7 s（约 1.43 Hz）。

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

时钟参考实现位于 `source/robot_lab/robot_lab/gait.py`，配置入口为 `commands.base_velocity.gait`。当前固定周期 0.7 s，支撑占空比 0.6，相位边界宽容区宽度为周期的 0.05。平移进入/退出阈值为 0.08/0.04 m/s，偏航为 0.10/0.05 rad/s。停止时推进到下一段计划双支撑并冻结，最长不超过半个周期；这是按相位结束摆动，不根据接触反馈等待实际落地。该版本没有额外停止计时器、可变频率或活动强度观测。

各 dog2 环境统一通过 `dog2/periodic_gait.py:configure_gait_rewards()` 配置：

| 奖励项 | 权重 | 定义 |
|---|---:|---|
| `gait_contact` | -5 | 支撑未接触、摆动仍接触的相位违约 |
| `swing_clearance` | -2 | 摆动足底相对脚下地形的归一化间隙误差 |
| `feet_slide` | -1 | 接触脚中心的水平滑移，包含刚体转动引起的速度 |
| `standing_support` | +4 | 持续接触、低机身/脚速、低动作变化、直立姿态 |
| `standing_joint_motion` | -2 | 停止过渡结束后的关节速度与动作变化惩罚 |

摆动包络峰值为 flat/rough/slope/sand 0.07 m、flat_domain_rand 0.08 m、stairs/bar/tires 0.10 m，误差尺度为 0.04 m。足底位置使用 URDF 固定足关节偏移，并按足部网格外接圆柱计算随姿态变化的竖向支撑范围；它是足底几何近似，不是 calf 原点高度。训练奖励复用特权地形扫描器加载的网格，在脚中心竖直向下射线查询地面；无效射线和过陡侧面命中不计入间隙项。Actor 没有新增地形输入。

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



