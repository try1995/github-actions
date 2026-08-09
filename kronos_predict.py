# -*- coding: utf-8 -*-
"""
main.py - Kronos 股票预测脚本
使用 baostock 获取 A 股历史 K 线数据，加载 Kronos 模型进行未来价格预测。

用法：
    python kronos_predict.py

可同时预测多只股票（STOCK_CODE 用 | 分隔，如 "601601|002185"），
STOCK_CODE 可通过环境变量配置（GitHub Actions 里用仓库 Variables），不设置时用默认值。
每次运行会为每只股票按 TIMEFRAMES 配置生成多个周期的 K 线预测图（日线 + 5分钟），
并配置 SMTP 环境变量后，每只股票跑完所有周期就立即发送一封 HTML 邮件
（K 线图以内嵌图片形式显示在正文，邮件标题包含股票代码）。
"""

import base64
import glob
import os
import smtplib
import socket
import sys
import time
import warnings
from datetime import datetime, timedelta
from html import escape
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ⚠️ 必须在 import model 之前设置 HF_ENDPOINT！
# huggingface_hub 在 import 时就会把 ENDPOINT 缓存为 https://huggingface.co，
# 之后再设 os.environ 不会生效。huggingface.co 在国内被墙，会导致
# from_pretrained 联网检查版本时长时间超时（即使模型已本地缓存）。
# 如不需要镜像可注释掉这一行。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import baostock as bs
import torch

from model import Kronos, KronosTokenizer, KronosPredictor

warnings.filterwarnings("ignore")

# ==================== 配置参数 ====================
STOCK_CODE = os.environ.get("STOCK_CODE", "").strip() or "601601|002185"
# 股票代码，支持多个，用 | 分隔；GitHub Actions 里通过仓库 Variables.STOCK_CODE 配置
END_DATE = datetime.now().strftime("%Y-%m-%d")  # 数据结束日期（今天）

# 解析多股票代码：按 | 分割，逐只自动判断交易所前缀
STOCK_CODES = [c.strip() for c in STOCK_CODE.split("|") if c.strip()]
if not STOCK_CODES:
    raise ValueError("STOCK_CODE 配置为空，请设置至少一只股票代码")

# 预测周期配置：每次运行对每个周期分别拉取数据、预测并生成 K 线图
TIMEFRAMES = [
    {
        "label": "5分钟",
        "file_tag": "5min",
        "frequency": "5",            # baostock 频率：5=5分钟
        "start_date": "2024-01-01",  # 分钟线数据量大，不宜取太早
        "adjust_flag": "3",          # 复权：1=后复权，2=前复权，3=不复权
        "lookback": 400,             # 回看 K 线根数（喂给模型）
        "pred_len": 48,              # 预测 K 线根数（48 根 ≈ 1 个交易日）
        "chart_bars": 120,           # 图中显示的历史 K 线根数（太多会挤成一团）
    },
    {
        "label": "日线",
        "file_tag": "daily",
        "frequency": "d",            # baostock 频率：d=日线
        "start_date": "2020-01-01",
        "adjust_flag": "3",
        "lookback": 250,             # 回看约一年
        "pred_len": 20,              # 预测约一个月
        "chart_bars": 150,           # 图中显示的历史 K 线根数
    },
]


def detect_market_prefix(stock_code: str) -> str:
    """
    根据股票代码自动判断交易所前缀（sh=上海，sz=深圳，bj=北交所）。

    规则：
        6 开头    -> sh（上海主板、科创板：600/601/603/605/688）
        0/3 开头  -> sz（深圳主板、创业板：000/001/002/300/301）
        4/8/9 开头 -> bj（北交所：43x/83x/87x/92x）
    """
    if stock_code.startswith("6"):
        return "sh"
    if stock_code.startswith(("0", "3")):
        return "sz"
    if stock_code.startswith(("4", "8", "9")):
        return "bj"
    raise ValueError(f"无法根据股票代码 {stock_code} 判断交易所，请手动指定前缀")


# 模型参数
TOKENIZER_PRETRAINED = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_PRETRAINED = "NeoQuasar/Kronos-base"

# 自动选择设备：优先 GPU（cuda），不可用时回退 CPU
if torch.cuda.is_available():
    DEVICE = "cuda:0"
    print(f"✅ 使用 GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("⚠️ CUDA 不可用，回退到 CPU")

MAX_CONTEXT = 512                   # 最大上下文长度（各周期 lookback + pred_len 需小于该值）
TEMPUTER = 0.1                             # 采样温度
TOP_P = 0.9                         # top-p 采样
SAMPLE_COUNT = 1                    # 采样次数

# 输出目录
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== 数据获取 ====================
def fetch_stock_data(stock_code: str, market_prefix: str,
                     start_date: str, end_date: str,
                     frequency: str, adjust_flag: str) -> pd.DataFrame:
    """
    使用 baostock 获取 A 股历史 K 线数据。

    参数：
        stock_code: 股票代码，如 "601601"
        market_prefix: 市场前缀，"sh" 或 "sz"
        start_date: 起始日期 "YYYY-MM-DD"
        end_date: 结束日期 "YYYY-MM-DD"
        frequency: K线周期 "d"/"w"/"m"
        adjust_flag: 复权类型

    返回：
        pandas.DataFrame，包含 date, open, high, low, close, volume, amount 等列
    """
    full_code = f"{market_prefix}.{stock_code}"
    print(f"📡 正在通过 baostock 获取 {full_code} 的 {frequency} 周期历史K线数据...")

    # 分钟线与日线字段不同：
    #   分钟线（5/15/30/60）：date,time,code,open,high,low,close,volume,amount,adjustflag
    #   日线/周/月（d/w/m）：多出 preclose, turn, tradestatus, pctChg, isST
    is_minute = frequency.lower() in ("5", "15", "30", "60")
    if is_minute:
        fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"
    else:
        fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"

    # 登录 baostock
    lg = bs.login()
    if lg.error_code != "0":
        print(f"❌ baostock 登录失败: {lg.error_msg}")
        sys.exit(1)
    print(f"✅ baostock 登录成功")

    rs = bs.query_history_k_data_plus(
        full_code,
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        adjustflag=adjust_flag,
    )

    if rs.error_code != "0":
        print(f"❌ 查询失败: {rs.error_msg}")
        bs.logout()
        sys.exit(1)

    # 解析结果
    data_list = []
    while (rs.error_code == "0") & rs.next():
        data_list.append(rs.get_row_data())

    bs.logout()
    print(f"✅ baostock 已登出")

    if not data_list:
        print("❌ 未获取到任何数据")
        sys.exit(1)

    # 转为 DataFrame
    df = pd.DataFrame(data_list, columns=rs.fields)

    # 统一时间列：分钟线 = date + time（time 格式 YYYYMMDDHHMMSSsss）；日线 = date
    if is_minute:
        df["datetime"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S%f", errors="coerce")
    else:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")

    # 类型转换（分钟线没有 preclose, turn, tradestatus, pctChg, isST）
    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 日线专属：换手率空值补 0、过滤停牌
    if "turn" in df.columns:
        df["turn"] = df["turn"].fillna(0.0)
    if "tradestatus" in df.columns:
        df = df[df["tradestatus"] == "1"].copy()

    # 按时间排序
    df = df.sort_values("datetime").reset_index(drop=True)

    # 处理缺失的 amount（成交额 = 收盘价 * 成交量）
    if df["amount"].isna().all() or (df["amount"] == 0).all():
        df["amount"] = df["close"] * df["volume"]

    # 修复无效的开盘价
    open_bad = (df["open"] == 0) | df["open"].isna()
    if open_bad.any():
        print(f"⚠️ 修复 {open_bad.sum()} 条无效开盘价")
        df.loc[open_bad, "open"] = df["close"].shift(1)
        df["open"] = df["open"].fillna(df["close"])

    print(f"✅ 数据获取完成: {len(df)} 条记录")
    print(f"   时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"   最新收盘价: {df['close'].iloc[-1]:.2f} 元")

    return df


# ==================== 数据准备 ====================
def prepare_inputs(df: pd.DataFrame, lookback: int, pred_len: int, frequency: str = "5"):
    """
    准备 Kronos 模型所需的输入数据。

    参数：
        df: 历史数据 DataFrame（含 datetime 列）
        lookback: 回看窗口大小（K 线条数）
        pred_len: 预测 K 线条数
        frequency: K 线周期（"5"/"15"/"30"/"60"/"d"）

    返回：
        x_df: 特征数据 (lookback 行)
        x_timestamp: 历史时间戳
        y_timestamp: 未来预测时间戳
    """
    # 取最近 lookback 行作为输入特征
    x_df = df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]].copy()
    x_timestamp = df.iloc[-lookback:]["datetime"].copy()

    # 生成未来时间戳（分钟线按交易时段，日线仅工作日）
    last_ts = df["datetime"].iloc[-1]
    if frequency.lower() in ("5", "15", "30", "60"):
        future_ts = _generate_future_5min_timestamps(last_ts, pred_len)
    else:
        future_ts = _generate_future_daily_timestamps(last_ts, pred_len)

    y_timestamp = pd.Series(future_ts[:pred_len])

    print(f"📊 输入数据: {x_df.shape[0]} 条历史 K 线 -> 预测 {pred_len} 条 K 线")
    print(f"   历史范围: {x_timestamp.iloc[0]} ~ {x_timestamp.iloc[-1]}")
    print(f"   预测范围: {y_timestamp.iloc[0]} ~ {y_timestamp.iloc[-1]}")

    return x_df, x_timestamp, y_timestamp


def _generate_future_5min_timestamps(last_ts: pd.Timestamp, count: int) -> list:
    """
    生成未来 5 分钟 K 线的时间戳序列（仅 A 股交易时段）。

    A 股交易时间（北京时间）：
      - 上午：9:30 – 11:30
      - 下午：13:00 – 15:00

    返回：list of pd.Timestamp
    """
    timestamps = []
    current = last_ts + timedelta(minutes=5)

    while len(timestamps) < count:
        # 跳过周末
        if current.weekday() >= 5:
            current = current + timedelta(days=1)
            current = current.replace(hour=9, minute=35, second=0, microsecond=0)
            continue

        t = current.time()
        # 在上午交易时段内（9:30–11:30）
        in_morning = t >= pd.Timestamp("09:30").time() and t <= pd.Timestamp("11:30").time()
        # 在下午交易时段内（13:00–15:00）
        in_afternoon = t >= pd.Timestamp("13:00").time() and t <= pd.Timestamp("15:00").time()

        if in_morning or in_afternoon:
            timestamps.append(current)

        current = current + timedelta(minutes=5)

        # 跨越午休：上午结束后跳到下午第一根 K 线 (13:05)
        if current.time() > pd.Timestamp("11:30").time() and current.time() < pd.Timestamp("13:00").time():
            current = current.replace(hour=13, minute=5, second=0, microsecond=0)

        # 跨越收盘：当天下午结束后跳到下一交易日上午第一根 K 线 (9:35)
        if current.time() > pd.Timestamp("15:00").time():
            current = current + timedelta(days=1)
            current = current.replace(hour=9, minute=35, second=0, microsecond=0)

    return timestamps


def _generate_future_daily_timestamps(last_ts: pd.Timestamp, count: int) -> list:
    """
    生成未来日线 K 线的时间戳序列（仅工作日，取当天 00:00）。

    返回：list of pd.Timestamp
    """
    timestamps = []
    current = last_ts.normalize() + timedelta(days=1)
    while len(timestamps) < count:
        if current.weekday() < 5:  # 周一至周五
            timestamps.append(current)
        current = current + timedelta(days=1)
    return timestamps


# ==================== 涨跌停限制 ====================
def apply_price_limits(pred_df: pd.DataFrame, last_close: float,
                       limit_rate: float = 0.1) -> pd.DataFrame:
    """
    对预测结果应用 A 股 ±10% 涨跌停限制。

    参数：
        pred_df: 预测结果 DataFrame
        last_close: 最后一个历史收盘价
        limit_rate: 涨跌停幅度（默认 10%）

    返回：
        限制后的预测 DataFrame
    """
    print(f"🔒 应用 ±{limit_rate * 100:.0f}% 涨跌停限制...")
    pred_df = pred_df.reset_index(drop=True)
    cols = ["open", "high", "low", "close"]
    pred_df[cols] = pred_df[cols].astype("float64")

    prev_close = last_close
    for i in range(len(pred_df)):
        limit_up = prev_close * (1 + limit_rate)
        limit_down = prev_close * (1 - limit_rate)
        for col in cols:
            value = pred_df.at[i, col]
            if pd.notna(value):
                pred_df.at[i, col] = float(max(min(value, limit_up), limit_down))
        prev_close = float(pred_df.at[i, "close"])

    return pred_df


# ==================== 可视化（K线蜡烛图） ====================
def _is_minute_data(df: pd.DataFrame) -> bool:
    """根据相邻时间戳间距判断是否为分钟级数据（间距 < 1 天）。"""
    if len(df) < 2:
        return False
    delta = (df["datetime"].iloc[1] - df["datetime"].iloc[0]).total_seconds()
    return 0 < delta < 24 * 3600


def _draw_candles(ax, xs, df: pd.DataFrame, body_w: float, alpha: float = 1.0):
    """绘制一组 K 线蜡烛（红涨绿跌）。xs 为整数序号坐标（类别轴）。"""
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    up = c >= o
    colors = np.where(up, "#d62728", "#2ca02c")  # 红涨绿跌

    # 影线（最高/最低价）
    ax.vlines(xs, l, h, color=colors, linewidth=1.0, alpha=alpha)

    # 实体（开-收区间）
    body_bottom = np.minimum(o, c)
    body_height = np.abs(c - o)
    ax.bar(xs, body_height, bottom=body_bottom, width=body_w,
           color=colors, edgecolor=colors, linewidth=0.5, alpha=alpha, align="center")

    # 开收同价（十字星）：画一条短横线
    flat = body_height == 0
    if flat.any():
        ax.plot(xs[flat], o[flat], marker="_", linestyle="None",
                color="#d62728", markersize=7, alpha=alpha)


def _draw_volume(ax, xs, df: pd.DataFrame, body_w: float, alpha: float = 1.0):
    """绘制成交量柱状图（按涨跌红绿着色）。"""
    up = df["close"].to_numpy(dtype=float) >= df["open"].to_numpy(dtype=float)
    colors = np.where(up, "#d62728", "#2ca02c")
    ax.bar(xs, df["volume"].to_numpy(dtype=float), width=body_w,
           color=colors, alpha=alpha, align="center")


def plot_kline_result(df_hist: pd.DataFrame, df_pred: pd.DataFrame,
                      stock_code: str, label: str, chart_path: str,
                      chart_bars: int = None):
    """
    绘制 K 线蜡烛图：历史 K 线 + 预测 K 线（红涨绿跌）+ 成交量副图 + 均线。

    使用类别坐标（每根 K 线等宽连续排列），消除隔夜/午休/周末的时间空洞，
    观感更接近行情软件。图上只显示最近 chart_bars 根历史 K 线。

    参数：
        df_hist: 历史 K 线（含 datetime, open, high, low, close, volume）
        df_pred: 预测 K 线（同列结构）
        stock_code: 股票代码
        label: 周期名称（如 "5分钟" / "日线"）
        chart_path: 输出图片路径
        chart_bars: 图中最多显示的历史 K 线根数（默认全部；分钟线建议 120 左右）
    """
    # 设置中文字体（Linux 无 SimHei，使用文泉驿正黑；Windows 可用 SimHei）
    plt.rcParams["font.sans-serif"] = [
        "WenQuanYi Zen Hei", "SimHei", "Noto Sans CJK SC", "Microsoft YaHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # 显示窗口：默认全量；chart_bars 截取尾部，避免分钟线画太多根挤成一团
    chart_bars = len(df_hist) if chart_bars is None else max(10, min(int(chart_bars), len(df_hist)))

    # 均线在完整历史上计算，再截取显示窗口，避免窗口开头 MA 缺失
    ma5 = df_hist["close"].rolling(5).mean()
    ma20 = df_hist["close"].rolling(20).mean()
    show_hist = df_hist.iloc[-chart_bars:].reset_index(drop=True)
    ma5 = ma5.iloc[-chart_bars:].reset_index(drop=True)
    ma20 = ma20.iloc[-chart_bars:].reset_index(drop=True)

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # 类别坐标：第 i 根 K 线落在整数 x 上，等宽、无空洞
    x_hist = np.arange(len(show_hist))
    x_pred = np.arange(len(show_hist), len(show_hist) + len(df_pred))
    body_w = 0.7

    # ---- 价格面板：历史 + 预测 K 线 ----
    _draw_candles(ax_price, x_hist, show_hist, body_w, alpha=1.0)
    _draw_candles(ax_price, x_pred, df_pred, body_w, alpha=0.85)

    # 预测起点竖线 + 预测区域背景
    last_hist_x = x_hist[-1]
    ax_price.axvline(x=last_hist_x, color="#555555", linestyle=":", alpha=0.6, linewidth=1.2)
    ax_price.axvspan(x_pred[0] - 0.5, x_pred[-1] + 0.5, color="orange", alpha=0.08, label="预测区域")

    # 均线（MA5 / MA20）
    if len(show_hist) >= 5 and ma5.notna().any():
        ax_price.plot(x_hist, ma5, color="#ff7f0e", linewidth=1.2, alpha=0.9, label="MA5")
    if len(show_hist) >= 20 and ma20.notna().any():
        ax_price.plot(x_hist, ma20, color="#9467bd", linewidth=1.2, alpha=0.9, label="MA20")

    ax_price.set_ylabel("价格 (元)", fontsize=12)
    ax_price.set_title(
        f"{stock_code} - {label}K线预测\n"
        f"当前价: {show_hist['close'].iloc[-1]:.2f} 元 | 预测 {len(df_pred)} 根 K 线",
        fontsize=14, fontweight="bold",
    )
    ax_price.legend(loc="upper left", fontsize=9)

    # 网格：只保留横向细网格，避免纵向噪点
    ax_price.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax_price.grid(axis="x", color="none")
    ax_vol.grid(axis="y", alpha=0.3, linewidth=0.5)

    # X 轴刻度：均匀抽 ~8 个点显示时间标签
    is_minute = _is_minute_data(df_hist)
    fmt = "%m-%d %H:%M" if is_minute else "%Y-%m-%d"
    step = max(1, len(show_hist) // 8)
    tick_idx = np.arange(0, len(show_hist), step)
    ax_price.set_xticks(x_hist[tick_idx])
    ax_price.set_xticklabels([show_hist["datetime"].iloc[i].strftime(fmt) for i in tick_idx],
                             fontsize=8)

    # 预测起始处标注
    ax_price.annotate(
        "预测", xy=(x_pred[0], df_pred["high"].iloc[0]),
        xytext=(4, 8), textcoords="offset points", fontsize=9,
        color="#e23636", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#e23636"),
    )

    # ---- 成交量面板 ----
    _draw_volume(ax_vol, x_hist, show_hist, body_w, alpha=0.7)
    _draw_volume(ax_vol, x_pred, df_pred, body_w, alpha=0.6)

    ax_vol.set_ylabel("成交量 (股)", fontsize=12)
    ax_vol.set_xlabel("时间", fontsize=12)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 图表已保存: {chart_path}")


# ==================== 主预测流程 ====================
def run_prediction(tf: dict, predictor: KronosPredictor, stock_code: str, market_prefix: str):
    """
    对单个股票、单个周期执行完整预测流程：拉数据 -> 预测 -> 限制 -> 保存 -> 画K线图 -> 摘要。
    """
    label = tf["label"]
    freq = tf["frequency"]

    print("\n" + "=" * 60)
    print(f"📈 预测周期: {label}（frequency={freq}） | 股票: {stock_code}")
    print("=" * 60)

    # ---- 获取历史数据 ----
    print("\n📥 获取历史K线数据...")
    df = fetch_stock_data(
        stock_code=stock_code,
        market_prefix=market_prefix,
        start_date=tf["start_date"],
        end_date=END_DATE,
        frequency=freq,
        adjust_flag=tf["adjust_flag"],
    )

    # ---- 准备输入数据 ----
    print("\n📊 准备输入数据...")
    x_df, x_timestamp, y_timestamp = prepare_inputs(
        df, tf["lookback"], tf["pred_len"], frequency=freq,
    )

    # ---- 执行预测 ----
    print("\n🔮 执行价格预测...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=tf["pred_len"],
        T=TEMPUTER,
        top_p=TOP_P,
        sample_count=SAMPLE_COUNT,
    )
    pred_df["datetime"] = y_timestamp.values
    print("✅ 预测完成")

    # ---- 应用涨跌停限制 ----
    print("\n🔒 应用价格限制...")
    last_close = df["close"].iloc[-1]
    pred_df = apply_price_limits(pred_df, last_close, limit_rate=0.1)

    # ---- 保存结果 ----
    print("\n💾 保存预测结果...")
    hist_part = df.iloc[-tf["lookback"]:][["datetime", "open", "high", "low", "close", "volume", "amount"]].copy()
    hist_part["type"] = "历史"
    pred_part = pred_df[["datetime", "open", "high", "low", "close", "volume", "amount"]].copy()
    pred_part["type"] = "预测"
    df_out = pd.concat([hist_part, pred_part], ignore_index=True)

    csv_path = os.path.join(OUTPUT_DIR, f"pred_{stock_code}_{tf['file_tag']}_data.csv")
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"📁 预测数据已保存: {csv_path}")

    # ---- 生成 K 线图 ----
    print("\n📊 生成 K 线预测图...")
    hist_for_plot = df.iloc[-tf["lookback"]:][["datetime", "open", "high", "low", "close", "volume"]].copy()
    chart_path = os.path.join(OUTPUT_DIR, f"pred_{stock_code}_{tf['file_tag']}_chart.png")
    plot_kline_result(hist_for_plot, pred_df, stock_code, label, chart_path,
                      chart_bars=tf.get("chart_bars"))

    # ---- 打印预测摘要 ----
    print("\n" + "=" * 60)
    print(f"📈 {label} 预测摘要")
    print("=" * 60)
    current_price = df["close"].iloc[-1]
    pred_final_close = pred_df["close"].iloc[-1]
    change_pct = (pred_final_close / current_price - 1) * 100

    print(f"   股票: {stock_code}")
    print(f"   当前价格: {current_price:.2f} 元")
    print(f"   预测 {tf['pred_len']} 根 K 线后价格: {pred_final_close:.2f} 元")
    print(f"   预测涨跌幅: {change_pct:+.2f}%")

    # 预测区间统计
    pred_min = pred_df["close"].min()
    pred_max = pred_df["close"].max()
    pred_mean = pred_df["close"].mean()
    print(f"   预测区间最低: {pred_min:.2f} 元")
    print(f"   预测区间最高: {pred_max:.2f} 元")
    print(f"   预测区间均价: {pred_mean:.2f} 元")

def _collect_summaries(stock_code: str = None) -> list:
    """
    从 outputs/ 下的 *_data.csv 收集预测摘要。

    参数：
        stock_code: 指定时只收集该股票；为 None 时收集全部。

    返回 dict 列表，每项包含：code, tag, label, current, final_pred, change,
    pred_min, pred_max, pred_mean, n_bars, chart（文件名或 None）, pred_start, pred_end。
    """
    tf_label = {"5min": "5分钟", "daily": "日线"}
    summaries = []
    for csv_path in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_data.csv"))):
        base = os.path.basename(csv_path)              # pred_601601_daily_data.csv
        code_tf = base[len("pred_"):-len("_data.csv")]  # 601601_daily
        code, _, tag = code_tf.rpartition("_")
        if not code:
            continue
        if stock_code and code != stock_code:
            continue
        try:
            df = pd.read_csv(csv_path)
            hist = df[df["type"] == "历史"]
            pred = df[df["type"] == "预测"]
            if hist.empty or pred.empty:
                continue
            chart_name = f"pred_{code}_{tag}_chart.png"
            chart_path = os.path.join(OUTPUT_DIR, chart_name)
            summaries.append({
                "code": code,
                "tag": tag,
                "label": tf_label.get(tag, tag),
                "current": float(hist["close"].iloc[-1]),
                "final_pred": float(pred["close"].iloc[-1]),
                "change": (float(pred["close"].iloc[-1]) / float(hist["close"].iloc[-1]) - 1) * 100,
                "pred_min": float(pred["close"].min()),
                "pred_max": float(pred["close"].max()),
                "pred_mean": float(pred["close"].mean()),
                "n_bars": len(pred),
                "chart": chart_name if os.path.exists(chart_path) else None,
                "pred_start": pred["datetime"].iloc[0],
                "pred_end": pred["datetime"].iloc[-1],
            })
        except Exception as e:
            print(f"⚠️ 解析 {csv_path} 生成摘要失败: {e}")
    return summaries


def build_summary_text() -> str:
    """生成纯文本预测摘要（HTML 正文之外的 text 部分，供不支持 HTML 的客户端回退）。"""
    lines = [f"Kronos 股票预测日报 - {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
    for s in _collect_summaries():
        lines.append(f"【{s['code']} {s['label']}】")
        lines.append(f"  当前价: {s['current']:.2f} 元 | 预测 {s['n_bars']} 根后: {s['final_pred']:.2f} 元 ({s['change']:+.2f}%)")
        lines.append(f"  预测区间: 最低 {s['pred_min']:.2f} / 最高 {s['pred_max']:.2f} / 均价 {s['pred_mean']:.2f} 元")
        lines.append("")
    return "\n".join(lines)


def _dir_color(change: float) -> str:
    """A 股配色：涨红跌绿。"""
    return "#dc2626" if change >= 0 else "#16a34a"


def _fmt_pred_range(start, end) -> str:
    """把预测起止时间格式化成可读区间（分钟级精确到时分，日级精确到日期）。"""
    try:
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        if (e - s).total_seconds() < 24 * 3600:
            return f"{s.strftime('%m-%d %H:%M')} ~ {e.strftime('%m-%d %H:%M')}"
        return f"{s.strftime('%Y-%m-%d')} ~ {e.strftime('%Y-%m-%d')}"
    except Exception:
        return f"{start} ~ {end}"


def _stat_cell(label: str, value: str, sub: str = "", color: str = "#0f172a") -> str:
    """HTML 邮件里的一个统计格（表格布局 + 内联样式，跨邮件客户端兼容）。"""
    sub_html = f'<div style="font-size:11px;color:#94a3b8;margin-top:4px;">{sub}</div>' if sub else ""
    return (
        f'<td align="center" style="padding:10px 6px;vertical-align:top;">'
        f'<div style="font-size:12px;color:#64748b;margin-bottom:6px;">{label}</div>'
        f'<div style="font-size:20px;font-weight:700;color:{color};">{value}</div>'
        f'{sub_html}'
        f'</td>'
    )


def _chart_data_uri(chart_name: str) -> str:
    """读取 K 线图 PNG，转成 base64 data URI 直接内嵌进 HTML 正文（HTML 自包含，不引用本地文件）。"""
    path = os.path.join(OUTPUT_DIR, chart_name)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _summary_card(s: dict) -> str:
    """单个股票×周期的摘要卡片：统计区 + 内嵌 K 线图（二进制 base64 直接写在 img 里）。"""
    code = escape(s["code"])
    color = _dir_color(s["change"])
    arrow = "▲" if s["change"] >= 0 else "▼"
    range_str = _fmt_pred_range(s["pred_start"], s["pred_end"])
    chart_html = ""
    if s["chart"]:
        chart_html = (
            f'<tr><td style="padding:2px 20px 20px;">'
            f'<img src="{_chart_data_uri(s["chart"])}" alt="{code} {escape(s["label"])} K线预测图" '
            f'style="width:100%;height:auto;border-radius:8px;border:1px solid #e2e8f0;display:block;">'
            f'</td></tr>'
        )
    return f'''
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;margin:14px 0;">
      <tr>
        <td style="padding:16px 20px 6px;">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
            <tr>
              <td style="font-size:16px;font-weight:700;color:#0f172a;">{code}
                <span style="display:inline-block;font-size:12px;font-weight:400;color:#ffffff;background:#3b82f6;border-radius:4px;padding:2px 8px;margin-left:8px;vertical-align:middle;">{escape(s["label"])}</span>
              </td>
              <td align="right" style="font-size:12px;color:#64748b;">预测 {s["n_bars"]} 根 K 线</td>
            </tr>
          </table>
        </td>
      </tr>
      <tr>
        <td style="padding:4px 20px 8px;">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
            <tr>
              {_stat_cell("当前价", f'{s["current"]:.2f}', "元")}
              {_stat_cell("预测收盘", f'{s["final_pred"]:.2f}', "元", color)}
              {_stat_cell("预测涨跌幅", f'{arrow} {s["change"]:+.2f}%', "", color)}
              {_stat_cell("预测区间均价", f'{s["pred_mean"]:.2f}', "元")}
            </tr>
          </table>
        </td>
      </tr>
      <tr>
        <td style="padding:0 20px 12px;font-size:11px;color:#94a3b8;">
          预测区间最低 {s["pred_min"]:.2f} 元 · 最高 {s["pred_max"]:.2f} 元 · 时间 {escape(range_str)}
        </td>
      </tr>
      {chart_html}
    </table>'''


def _stock_header(code: str) -> str:
    code = escape(code)
    return f'''
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin:22px 0 2px;">
      <tr>
        <td style="border-left:4px solid #2563eb;padding-left:10px;font-size:15px;font-weight:700;color:#0f172a;">📌 股票 {code}</td>
      </tr>
    </table>'''


def build_summary(stock_code: str = None) -> str:
    """
    生成好看的 HTML 邮件正文（内联样式 + 表格布局，兼容主流邮件客户端）。

    参数：
        stock_code: 指定时只生成该股票的摘要卡片；为 None 时生成全部股票。

    K 线图以 base64 data URI 直接内嵌进 HTML（<img src="data:image/png;base64,...">），
    生成的 HTML 是自包含文件，不引用任何本地文件。
    纯文本版本见 build_summary_text()（供不支持 HTML 的客户端回退）。
    """
    summaries = _collect_summaries(stock_code)
    if not summaries:
        return "<html><body><p>本次未生成任何预测摘要。</p></body></html>"

    # 按股票代码分组，保持原有 CSV 顺序
    by_code = {}
    for s in summaries:
        by_code.setdefault(s["code"], []).append(s)

    body_html = "".join(
        _stock_header(code) + "".join(_summary_card(s) for s in items)
        for code, items in by_code.items()
    )

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kronos 股票预测日报</title>
</head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:'Microsoft YaHei','PingFang SC','Helvetica Neue',Arial,sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#f1f5f9;">
    <tr>
      <td align="center" style="padding:28px 16px;">
        <table role="presentation" width="680" cellpadding="0" cellspacing="0" border="0" style="max-width:680px;width:100%;background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(15,23,42,0.08);">
          <tr>
            <td style="background:linear-gradient(135deg,#1e3a8a,#2563eb);padding:26px 32px;">
              <h1 style="margin:0;color:#ffffff;font-size:22px;font-weight:700;">📈 Kronos 股票预测日报</h1>
              <p style="margin:8px 0 0;color:#bfdbfe;font-size:13px;">生成时间：{now_str} · A股 K线 智能预测</p>
            </td>
          </tr>
          <tr>
            <td style="padding:8px 32px 24px;">
              {body_html}
            </td>
          </tr>
          <tr>
            <td style="padding:16px 32px;background:#f8fafc;border-top:1px solid #e2e8f0;">
              <p style="margin:0;font-size:11px;color:#94a3b8;line-height:1.7;">
                免责声明：本邮件由 Kronos 时间序列模型自动生成，预测结果仅供研究参考，不构成任何投资建议。股市有风险，投资需谨慎。<br>
                Generated by Kronos · GitHub Actions
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>'''


def write_summary_file(summary: str) -> None:
    """把 HTML 摘要写入 outputs/summary.html，并同时写一份纯文本 outputs/summary.txt。"""
    html_path = os.path.join(OUTPUT_DIR, "summary.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(summary)
    txt_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(build_summary_text())
    print(f"📄 邮件摘要已写入 {html_path}（K线图内嵌）与 {txt_path}")


def send_email(stock_code: str):
    """
    发送单只股票的预测邮件（每只股票跑完所有周期后调用一次），邮件标题包含股票代码。

    正文为 HTML（K 线图以 base64 二进制内嵌），只包含该股票各周期的摘要卡片；
    配置与 .github/workflows/kronos-predict.yml 中注入的 SMTP Secrets 一致：
        SMTP_HOST        发件服务器，如 smtp.qq.com
        SMTP_PORT        SMTP 端口，SSL 通常 465（默认 465）
        SMTP_PROTOCOL    ssl 或 starttls（默认 ssl = secure: true）
        SMTP_USER        发件邮箱账号
        SMTP_PASS        发件邮箱密码 / 授权码（QQ/163 用授权码）
        SMTP_RECIPIENT   收件邮箱，多个用逗号或分号分隔

    未配置 SMTP 环境变量时静默跳过（本地跑不配邮箱不报错）。
    """
    # ---- 生成该股票的 HTML 摘要（K 线图 base64 内嵌）----
    summaries = _collect_summaries(stock_code)
    if not summaries:
        print(f"⚠️ 股票 {stock_code} 没有预测结果，跳过发信")
        return
    summary_html = build_summary(stock_code=stock_code)

    host = os.environ.get("SMTP_HOST", "").strip()
    user = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASS", "").strip()
    recipient = os.environ.get("SMTP_RECIPIENT", "").strip()
    if not (host and user and password and recipient):
        print(f"📧 未配置 SMTP 环境变量（SMTP_HOST/SMTP_USER/SMTP_PASS/SMTP_RECIPIENT），跳过 {stock_code} 发信")
        return

    try:
        port = int(os.environ.get("SMTP_PORT", "465"))
    except ValueError:
        port = 465
    # ---- 组装邮件：HTML 正文（K 线图已 base64 内嵌），标题包含股票代码 ----
    msg = MIMEMultipart()
    msg["From"] = user
    msg["To"] = recipient
    msg["Subject"] = f"Kronos 股票预测日报 {stock_code} {datetime.now().strftime('%Y-%m-%d')}"
    msg.attach(MIMEText(summary_html, "html", "utf-8"))

    recipients = [r.strip() for r in recipient.replace(";", ",").split(",") if r.strip()]

    # ---- 发送（强制 IPv4，海外 runner 无 IPv6 路由）----
    try:
        print(host, port)
        server = smtplib.SMTP_SSL(host, port)
        server.login(user, password)
        server.set_debuglevel(0)
        server.sendmail(user, recipients, msg.as_string())
        n = summary_html.count("data:image/png;base64,")
        print(f"✅ 已发送 {stock_code} 预测邮件至 {recipient}（HTML 内嵌 {n} 张 K 线图）")
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")
        raise


def main():
    print("=" * 60)
    print(f"🚀 Kronos 股票预测系统")
    print(f"   股票: {', '.join(f'{detect_market_prefix(c)}.{c}' for c in STOCK_CODES)}")
    print(f"   模型: {MODEL_PRETRAINED}")
    print(f"   分词器: {TOKENIZER_PRETRAINED}")
    print(f"   设备: {DEVICE}")
    print(f"   预测周期: {', '.join(tf['label'] for tf in TIMEFRAMES)}")
    print("=" * 60)

    # ---- 步骤1：加载模型（只加载一次，多周期复用）----
    print("\n🤖 步骤1: 加载 Kronos 模型和分词器...")
    print(f"   分词器: {TOKENIZER_PRETRAINED}")
    print(f"   模型: {MODEL_PRETRAINED}")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PRETRAINED)
    model = Kronos.from_pretrained(MODEL_PRETRAINED)
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CONTEXT)
    print("✅ 模型加载完成")

    # 打印 GPU 使用证据（模型权重所在设备 + 显存占用）
    if DEVICE.startswith("cuda"):
        print(f"   🎯 模型权重设备: {next(model.parameters()).device}")
        print(f"   🎯 分词器设备: {next(tokenizer.parameters()).device}")
        print(f"   🎯 显存占用: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

    # ---- 步骤2：逐股票、逐周期预测（模型只加载一次，复用）----
    for code in STOCK_CODES:
        market_prefix = detect_market_prefix(code)
        for tf in TIMEFRAMES:
            run_prediction(tf, predictor, stock_code=code, market_prefix=market_prefix)
        # 每只股票跑完所有周期后，立即发送一封包含该股票代码的预测邮件
        send_email(code)

    print(f"\n✅ 全部预测完成! 结果保存在 {OUTPUT_DIR}/ 目录下")
    for code in STOCK_CODES:
        for tf in TIMEFRAMES:
            print(f"   - {code} {tf['label']}: pred_{code}_{tf['file_tag']}_data.csv / _chart.png")

    # ---- 步骤3：写一份汇总摘要（HTML + 纯文本）到 outputs/，便于查看/留档（邮件已按股票分别发送）----
    write_summary_file(build_summary())


if __name__ == "__main__":
    main()
