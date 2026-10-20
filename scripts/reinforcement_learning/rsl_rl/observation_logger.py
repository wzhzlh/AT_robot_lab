"""Record one ATDog2 HIM environment's policy observations during training."""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path


OBSERVATIONS = (
    ("base_ang_vel", "机身角速度", ("X", "Y", "Z")),
    ("projected_gravity", "投影重力", ("X", "Y", "Z")),
    ("velocity_commands", "速度指令", ("vx", "vy", "yaw")),
    ("joint_pos", "关节位置", tuple(f"关节{i}" for i in range(1, 13))),
    ("joint_vel", "关节速度", tuple(f"关节{i}" for i in range(1, 13))),
    ("actions", "上一时刻动作", tuple(f"动作{i}" for i in range(1, 13))),
    ("phase", "步态相位", ("sin", "cos")),
)


class ObservationLogger:
    def __init__(self, env, log_dir: str, env_id: int = 0, sample_every: int = 10, plot_every: int = 500):
        if sample_every < 1 or plot_every < 1:
            raise ValueError("sample_every and plot_every must be positive")
        if not 0 <= env_id < env.num_envs:
            raise ValueError(f"env_id={env_id} is outside 0..{env.num_envs - 1}")
        self.slices = {}
        for name, _, labels in OBSERVATIONS:
            term_slice = env.policy_one_step_term_slices.get(name)
            if term_slice is None or term_slice.stop - term_slice.start != len(labels):
                raise ValueError(f"HIM observation {name!r} must have {len(labels)} dimensions")
            self.slices[name] = term_slice
        self.output_dir = Path(log_dir) / "observations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_file = (self.output_dir / "policy_observations.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(
            ["step", "time_s", "episode_step"]
            + [f"{name}_{i}" for name, _, labels in OBSERVATIONS for i in range(len(labels))]
        )
        self.env = env
        self.env_id = env_id
        self.sample_every = sample_every
        self.plot_every = plot_every
        self.step = 0
        self.samples = 0
        self.series = deque(maxlen=2000)

    def record(self, observations):
        if self.step % self.sample_every == 0:
            frame = observations[self.env_id, : self.env.num_one_step_obs].detach().cpu().tolist()
            values = [value for name, _, _ in OBSERVATIONS for value in frame[self.slices[name]]]
            time_s = self.step * self.env.unwrapped.step_dt
            episode_step = int(self.env.unwrapped.episode_length_buf[self.env_id].item())
            row = [self.step, time_s, episode_step, *values]
            self.writer.writerow(row)
            self.series.append(row)
            self.samples += 1
            if self.samples % self.plot_every == 0:
                self.csv_file.flush()
                self.plot()
        self.step += 1

    def plot(self):
        if not self.series:
            return
        plot_rows(self.series, self.output_dir)

    def close(self):
        if not self.csv_file.closed:
            self.csv_file.flush()
            try:
                self.plot()
            finally:
                self.csv_file.close()


def plot_rows(rows, output_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    cjk_fonts = [font.name for font in font_manager.fontManager.ttflist if font.name.startswith("Noto Sans CJK")]
    if cjk_fonts:
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC" if "Noto Sans CJK SC" in cjk_fonts else cjk_fonts[0]]
    plt.rcParams["axes.unicode_minus"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    times = [float(row[1]) for row in rows]
    column = 3
    for name, title, labels in OBSERVATIONS:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for offset, label in enumerate(labels):
            ax.plot(times, [float(row[column + offset]) for row in rows], label=label, linewidth=1)
        ax.set(title=title, xlabel="训练时间（秒）", ylabel="观测值")
        ax.grid(alpha=0.25)
        ax.legend(ncol=min(4, len(labels)), fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}.png", dpi=140)
        plt.close(fig)
        column += len(labels)


def main():
    parser = argparse.ArgumentParser(description="从训练观测 CSV 重新绘制中文曲线图")
    parser.add_argument("csv", type=Path, help="policy_observations.csv 的路径")
    args = parser.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file))[1:]
    plot_rows(rows, args.csv.parent)


if __name__ == "__main__":
    main()
