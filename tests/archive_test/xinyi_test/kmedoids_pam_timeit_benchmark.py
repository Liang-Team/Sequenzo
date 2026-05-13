"""
使用 timeit.repeat 对比 Sequenzo K-Medoids 与 PAM 在固定距离矩阵上的耗时。

数据来源（N 代入路径模板中的 %d）：
/Users/xinyi/Projects/sequenzo/sequenzo/data_and_output/orignal data/not_real_detailed_data/synthetic_detailed_U5_N%d.csv

说明：
- 距离矩阵 OM(TRATE) 在计时段之外预先算好；timeit 只包住 KMedoids 调用。
- K-Medoids 与 PAM 使用相同的 initialclust（1-based 中心下标）、k、npass，便于横向对比。
- 大样本下完整 n×n 距离矩阵占用内存极大，请将 N_VALUES 调成机器可承受的规模。
"""

from __future__ import annotations

import argparse
import os
import timeit
from pathlib import Path

import numpy as np
import pandas as pd

from sequenzo import SequenceData, get_distance_matrix
from sequenzo.clustering import KMedoids

DATA_PATH_TEMPLATE = (
    "/Users/xinyi/Projects/sequenzo/sequenzo/data_and_output/"
    "orignal data/not_real_detailed_data/synthetic_detailed_U5_N%d.csv"
)

DEFAULT_N_VALUES = (15000, 20000, 25000)

# 与 synthetic 序列及 KMedoids.py 示例一致的状态空间
SYNTH_STATES = [
    "Data",
    "Data science",
    "Hardware",
    "Research",
    "Software",
    "Support & test",
    "Systems & infrastructure",
]


def _resolve_data_path(template: str, n: int) -> Path:
    return Path(template % n)


def load_sequence_data(csv_path: Path) -> SequenceData:
    df = pd.read_csv(csv_path)
    # 列形式：(, id, 1, 2, ..., 10) —— 与时间列对齐用 columns[2:]
    time_cols = list(df.columns)[2:]
    return SequenceData(
        df,
        time=time_cols,
        id_col="id",
        states=SYNTH_STATES,
        labels=SYNTH_STATES,
    )


def distance_matrix_numpy(seqdata: SequenceData) -> np.ndarray:
    om = get_distance_matrix(
        seqdata=seqdata,
        method="OM",
        sm="TRATE",
        indel="auto",
    )
    return np.asarray(om, dtype=np.float64)


def _initial_medoids_1_based(k: int) -> np.ndarray:
    """与 tests/test_pam_and_kmedoids.py 一致的 1..k 起算下标。"""
    return np.arange(1, k + 1, dtype=np.int32)


def bench_method(
    diss: np.ndarray,
    *,
    method: str,
    k: int,
    npass: int,
    initialclust: np.ndarray,
    repeat: int,
    number: int,
) -> tuple[list[float], float]:
    """返回 (单次 repeat×number 的原始秒数列表, best of min chunk)."""

    def stmt() -> None:
        KMedoids(
            diss=diss,
            k=k,
            method=method,
            npass=npass,
            initialclust=initialclust,
            verbose=False,
        )

    # timeit 在计时循环内关闭 GC；对两次方法一致
    timed = timeit.repeat(
        stmt,
        repeat=repeat,
        number=number,
        timer=timeit.default_timer,
    )
    # 若 number>1，取每组 chunk 总和中的最小值为「一次 stmt 复制的 best」的常见报告方式
    if number > 1:
        # timed 长度为 repeat；每项是 number 次 stmt 叠加时间
        per_run = [t / number for t in timed]
        best = min(per_run)
        return timed, best
    best = min(timed)
    return timed, best


def run_for_n(
    n: int,
    *,
    data_template: str,
    k: int,
    npass: int,
    repeat: int,
    number: int,
) -> dict:
    path = _resolve_data_path(data_template, n)
    if not path.is_file():
        raise FileNotFoundError(path)

    t_load_0 = timeit.default_timer()
    seqdata = load_sequence_data(path)
    t_load_1 = timeit.default_timer()

    t_diss_0 = timeit.default_timer()
    diss = distance_matrix_numpy(seqdata)
    t_diss_1 = timeit.default_timer()

    init = _initial_medoids_1_based(k)

    km_times, km_best = bench_method(
        diss,
        method="KMedoids",
        k=k,
        npass=npass,
        initialclust=init,
        repeat=repeat,
        number=number,
    )
    pam_times, pam_best = bench_method(
        diss,
        method="PAM",
        k=k,
        npass=npass,
        initialclust=init,
        repeat=repeat,
        number=number,
    )

    return {
        "n": n,
        "path": str(path),
        "matrix_shape": diss.shape,
        "load_seconds": t_load_1 - t_load_0,
        "diss_seconds": t_diss_1 - t_diss_0,
        "kmedoids_times": km_times,
        "pam_times": pam_times,
        "kmedoids_best_seconds": km_best,
        "pam_best_seconds": pam_best,
    }


def _format_trials(times: list[float], number: int) -> str:
    if number > 1:
        vals = [f"{t / number:.6f}" for t in times]
    else:
        vals = [f"{t:.6f}" for t in times]
    return "[" + ", ".join(vals) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequenzo K-Medoids vs PAM，timeit.repeat 耗时")
    parser.add_argument(
        "--template",
        default=DATA_PATH_TEMPLATE,
        help="数据路径模板，包含一个 %%d 占位符表示 N",
    )
    parser.add_argument(
        "-n",
        "--n-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_N_VALUES),
        help="样本量 N（会代入路径模板 %%d），默认 %(default)s",
    )
    parser.add_argument("--k", type=int, default=5, help="簇数")
    parser.add_argument("--npass", type=int, default=5, help="传给 KMedoids 的 npass")
    parser.add_argument("--repeat", type=int, default=7, help="timeit.repeat 的 repeat")
    parser.add_argument("--number", type=int, default=1, help="timeit.repeat 的 number")
    args = parser.parse_args()

    if "%d" not in args.template:
        raise SystemExit("路径模板必须包含单个 %d 作为 N 的占位符")

    header = True
    for n in args.n_values:
        try:
            r = run_for_n(
                n,
                data_template=args.template,
                k=args.k,
                npass=args.npass,
                repeat=args.repeat,
                number=args.number,
            )
        except FileNotFoundError as e:
            print(f"[skip] N={n}: 文件不存在 ({e})")
            continue

        if header:
            print(
                os.linesep.join(
                    [
                        "Sequenzo K-Medoids vs PAM（timeit.repeat）",
                        f"  repeat={args.repeat}, number={args.number}, k={args.k}, npass={args.npass}",
                        f"  timer={timeit.default_timer.__module__}.{getattr(timeit.default_timer, '__name__', 'default_timer')}",
                        "",
                    ]
                ),
                end="",
            )
            header = False

        unit = "s/trial" if args.number > 1 else "s"
        print(f"--- N={n}  ({r['path']}) ---")
        print(f"  距离矩阵: {r['matrix_shape']}")
        print(f"  读入 CSV + SequenceData: {r['load_seconds']:.6f} s")
        print(f"  计算 OM 距离矩阵:       {r['diss_seconds']:.6f} s")
        print(f"  K-Medoids 各次 ({unit}): {_format_trials(r['kmedoids_times'], args.number)}  best(min): {r['kmedoids_best_seconds']:.6f}")
        print(f"  PAM 各次 ({unit}):       {_format_trials(r['pam_times'], args.number)}  best(min): {r['pam_best_seconds']:.6f}")
        print()


if __name__ == "__main__":
    main()
