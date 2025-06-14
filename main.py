import os
# 设置 LOKY 最大 CPU 数以避免物理核心探测警告
os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count()-1)
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count()-1)
import argparse
from pathlib import Path
from typing import Tuple, Any, Dict

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from tools import logger,scale_features,plot_scores,save_model, load_model
from model import train_knn, train_iforest, score_model
from data_loader import load_dataframe, generate_noisy_test


def score_samples(
    X: np.ndarray,
    model: Any,
    scaler: StandardScaler,
    meta: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """为 *X* 计算异常 **分数** 与布尔 **标签**。"""
    Xs = scaler.transform(X)
    mtype = meta["model_type"]

    if mtype == "knn":
        knn: NearestNeighbors = model
        dists, _ = knn.kneighbors(Xs)
        scores = dists[:, -1]
        labels = scores > meta["tau"]
    elif mtype == "iforest":
        scores = -model.decision_function(Xs)
        labels = scores > meta["tau"]
    else:
        raise ValueError(f"未知模型类型: {mtype}")

    return scores, labels


def cmd_train(args: argparse.Namespace) -> None:
    """处理 `train` 子命令。"""
    df = load_dataframe(Path(args.input))
    X_raw = df.values.astype(np.float32)
    Xs, scaler = scale_features(X_raw)
    contam = args.contamination

    if args.model_type == "knn":
        quantile = 1.0 - contam if contam < 1 else None  # 若设 1 表示全部正常
        model, tau = train_knn(Xs, k=args.k, quantile=quantile)
        meta = {"tau": tau, "model_type": "knn"}
        title = "k‑NN k‑距离 时序图"
        scores = model.kneighbors(Xs)[0][:, -1]
    elif args.model_type == "iforest":
        model, tau = train_iforest(
            Xs,
            n_estimators=args.n_estimators,
            contamination=contam,
            random_state=args.random_state,
        )
        meta = {"tau": tau, "model_type": "iforest"}
        title = "IsolationForest 异常分数 时序图"
        scores = -model.decision_function(Xs)
    else:
        logger.error("模型类型 '%s' 尚未实现", args.model_type)

    if args.model_out:
        save_model(Path(args.model_out), model, scaler, meta)

    if args.plot:
        plot_scores(df.index, scores, threshold=meta["tau"], title=title)

    if args.test:
        X_test, y_test = generate_noisy_test(
            X_raw,
            small_sigma=args.small_noise,
            large_sigma=args.large_noise,
            n_per_class=args.n_test // 2,
            random_state=args.random_state,
        )
        _, y_pred = score_samples(X_test, model, scaler, meta)
        acc = accuracy_score(y_test, y_pred)
        prec, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )

        logger.info("\n----- 模型噪声测试评估 -----\n准确率 ACC: %.4f\n精准率 PRE: %.4f\n召回率 REC: %.4f\nF1 分数  : %.4f\n%s",
                    acc, prec, recall, f1,
                    classification_report(y_test, y_pred, zero_division=0))


def cmd_score(args: argparse.Namespace) -> None:
    """处理 `score` 子命令。"""
    model, scaler, meta = load_model(Path(args.model_in))

    df = load_dataframe(Path(args.input))
    X_raw = df.values.astype(np.float32)
    scores, labels = score_samples(X_raw, model, scaler, meta)

    df_out = df.copy()
    df_out["score"] = scores
    df_out["is_anomaly"] = labels

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output)
    logger.info(
        "已评分 %d 行，检测到 %d 条异常，结果保存至 %s",
        len(df_out),
        int(labels.sum()),
        args.output,
    )


def build_parser():
    """
    构建命令行参数解析器，包含 train 和 score 子命令。
    """
    parser = argparse.ArgumentParser(description="统一异常检测管道")
    sub = parser.add_subparsers(dest="command", required=True)

    # train 子命令
    p_train = sub.add_parser('train', help='训练模型')
    p_train.add_argument('--input', default="", help='输入 Excel/CSV 文件路径')
    p_train.add_argument("--random-state", type=int, default=42, help="随机种子")
    p_train.add_argument('--model-out', default=False, help='输出模型文件 (.joblib)')
    p_train.add_argument('--out-name', default=False, help='输出模型文件 (.joblib)')
    p_train.add_argument('--model-type', choices=['knn','iforest','bp'], default='iforest', help='模型类型')
    p_train.add_argument('--k', type=int, default=5, help='kNN 的邻居数')
    p_train.add_argument(
        "--contamination",
        type=float,
        default=0.001,
        help="预期异常占比 (0–1)。k‑NN 将使用 1‑contamination 作为 quantile，iForest 直接使用该值",
    )
    p_train.add_argument('--metric', default='euclidean', help='距离度量方式')
    p_train.add_argument('--n-estimators', type=int, default=100, help='随机森林树数量')
    p_train.add_argument('--hidden-layer-sizes', nargs='+', type=int, default=[100], help='BP 网络隐藏层规模')
    p_train.add_argument('--plot', default=False, help='训练后绘制结果图')

    # ------ 噪声测试相关参数 ------ #
    p_train.add_argument("--test", default=True, help="训练后进行噪声测试评估")
    p_train.add_argument("--small-noise", type=float, default=0.01, help="无异常样本噪声系数")
    p_train.add_argument("--large-noise", type=float, default=0.1, help="有异常样本噪声系数")
    p_train.add_argument("--n-test", type=int, default=2000, help="噪声测试样本总量 (偶数)")

    # score 子命令
    p_score = sub.add_parser('score', help='对新数据进行评分')
    p_score.add_argument('--input', required=True, help='输入 Excel/CSV 文件路径')
    p_score.add_argument('--model-in', required=True, help='已训练模型文件 (.joblib)')
    p_score.add_argument('--output', required=True, help='评分结果 CSV 输出路径')

    return parser


def main() :
    """程序入口。"""
    parser = build_parser()
    args = parser.parse_args(["train"])

    if args.command == "train":
        cmd_train(args)
    elif args.command == "score":
        cmd_score(args)
    else:
        parser.error("未知命令：%s" % args.command)


if __name__ == "__main__":
    main()
