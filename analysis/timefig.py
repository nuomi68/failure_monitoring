import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter, MaxNLocator
from matplotlib import font_manager, rcParams
candidates = [
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
]
available = {f.name for f in font_manager.fontManager.ttflist}
for name in candidates:
    if name in available:
        rcParams["font.sans-serif"] = [name] + rcParams.get("font.sans-serif", [])
        break
rcParams["axes.unicode_minus"] = False
# ===== 1) 读数据 =====
path = r"./data/平均功率_seg1.xlsx"
df = pd.read_excel(path)

# ===== 2) 分钟级时间 -> Day(从1开始) =====
time_col = "时间"  # 你的表里就是这个列名

t = df[time_col]

# 情况A：时间列是“数值分钟”（例如 0, 4, 8, ... 或累计分钟）
if np.issubdtype(t.dtype, np.number):
    minutes_from_start = t - t.iloc[0]
    df["Day"] = minutes_from_start / (60 * 24) + 1

# 情况B：时间列是 datetime 或可解析为 datetime 的字符串
else:
    t = pd.to_datetime(t, errors="coerce")
    if t.isna().all():
        raise ValueError(f"列 {time_col} 无法解析为时间，也不是数值分钟。请检查数据。")
    df[time_col] = t
    df = df.sort_values(time_col).reset_index(drop=True)
    df["Day"] = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds() / 86400 + 1

# ===== 2.5) 可选：降采样让图更“干净”和更快（强烈建议）=====
# 你数据有 7万多点，直接画也行，但会显得密。
# 想更清爽：把下面 RESAMPLE 打开（比如 30min / 1H）
RESAMPLE = "30min"  # None / "30min" / "1H" / "2H" ...

if RESAMPLE is not None:
    if not np.issubdtype(df[time_col].dtype, np.number):
        df2 = (df.set_index(time_col)
                 .resample(RESAMPLE)
                 .mean(numeric_only=True)
                 .dropna()
                 .reset_index())
        df2["Day"] = (df2[time_col] - df2[time_col].iloc[0]).dt.total_seconds() / 86400 + 1
        df_plot = df2
    else:
        # 数值分钟的情况：用窗口平均（按点数）
        # 假设原始是 4分钟一条，30min≈7-8条；这里简单用 rolling
        window = 8
        df_plot = df.copy()
        for c in df_plot.columns:
            if c not in [time_col, "Day"] and np.issubdtype(df_plot[c].dtype, np.number):
                df_plot[c] = df_plot[c].rolling(window, min_periods=1).mean()
else:
    df_plot = df

# ===== 3) 画 2×2 紧凑图（中文标注 + 横轴不挤） =====
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})

fig, axs = plt.subplots(2, 2, figsize=(14, 7), sharex=True, constrained_layout=True)

def stylize(ax, title, ylabel=None):
    ax.set_title(title, pad=6)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(5))  # y轴刻度少一点更清爽

x = df_plot["Day"].to_numpy()
xmax = float(np.nanmax(x))
# 横轴上限取整到10天，观感更整齐
xmax_round = int(np.ceil(xmax / 10) * 10)

# 横轴刻度：自动稀疏（避免挤）
# <=120天：20天一格；>120天：40天一格（你也可改成固定40/50）
tick_step = 20 if xmax_round <= 120 else 40

for ax in axs.flat:
    ax.set_xlim(0, xmax_round)
    ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax.margins(x=0)

# 上排两张图不显示x轴刻度文字（避免重复、拥挤）
for ax in axs[0, :]:
    ax.tick_params(axis="x", labelbottom=False)

# 底排显示中文横轴名
axs[1, 0].set_xlabel("时间（天）")
axs[1, 1].set_xlabel("时间（天）")

# 你表里这四个量的列名：
y_power = df_plot["平均核功率"]
y_burn  = df_plot["燃耗"]
y_temp  = df_plot["一回路平均温度"]
y_boron = df_plot["硼浓度（硼表）"]

# ---- 左上：平均核功率 ----
axs[0, 0].plot(x, y_power, lw=1.2)
stylize(axs[0, 0], " ", "功率（%）")
axs[0, 0].set_ylim(-2, 105)  # 0~100更舒服

# ---- 右上：燃耗 ----
axs[0, 1].plot(x, y_burn, lw=1.2)
# 单位请按你的实际修改（我这里先写“燃耗（单位同表）”）
stylize(axs[0, 1], " ", "燃耗（EFPD）")

# ---- 左下：一回路平均温度 ----
axs[1, 0].plot(x, y_temp, lw=1.2)
stylize(axs[1, 0], " ", "温度（K）")

# ---- 右下：硼浓度 ----
axs[1, 1].plot(x, y_boron, lw=1.2)
stylize(axs[1, 1], " ", "硼浓度（ppm）")

fig.suptitle("平均功率_seg1 关键指标（紧凑视图）", y=1.02, fontsize=13)
plt.show()