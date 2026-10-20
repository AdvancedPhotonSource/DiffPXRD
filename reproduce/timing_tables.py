"""Summarize saved per-structure timings, pairing devices before taking ratios."""

import numpy as np
import pandas as pd


def paired(rows, device_column, value_column, cpu, gpu):
    values = rows.pivot(index="case_id", columns=device_column, values=value_column)
    if (len(values) != 200 or set(values.columns) != {cpu, gpu}
            or not np.isfinite(values.to_numpy()).all() or (values <= 0).any().any()):
        raise ValueError("Timing table requires exactly 200 positive paired CPU/A100 observations")
    return values.rename(columns={cpu: "cpu", gpu: "gpu"})


def statistics(values):
    return [float(np.median(values)), *map(float, np.quantile(values, [.25, .75]))]


def interval(values, *, time=False):
    def display(value):
        return (f"{value:.1f}" if value < 10 else f"{value:.0f}") if time else f"{value:#.2g}"
    median, low, high = map(display, values)
    return f"{median} [{low}--{high}]"


def write(output, stem, headers, records, cells):
    pd.DataFrame(records).to_csv(output / f"{stem}.csv", index=False)
    lines = [r"\begin{tabular}{@{}" + "l" * len(headers) + "@{}}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule",
             *[" & ".join(row) + r" \\" for row in cells], r"\bottomrule", r"\end{tabular}", ""]
    (output / f"{stem}.tex").write_text("\n".join(lines))


def timing_tables(complete, counts, merged, separate, output):
    values = paired(complete, "engine", "median_sec", "diffpxrd_cpu_pymatgen", "diffpxrd_gpu_pymatgen")
    counts = counts.set_index("case_id")
    if len(counts) != 200 or not counts.index.is_unique or set(counts.index) != set(values.index):
        raise ValueError("Peak counts and complete timings have different structure rosters")
    values = values.join(counts).reset_index().sort_values(["merged_peaks", "case_id"], kind="stable")
    peaks = values.merged_peaks.to_numpy()
    if not np.isfinite(peaks).all() or (peaks <= 0).any() or not np.equal(peaks, np.floor(peaks)).all():
        raise ValueError("Merged peak counts must be positive integers")
    records, cells = [], []
    for fifth, group in enumerate(np.array_split(np.arange(200), 5), 1):
        part = values.iloc[group]
        cpu, gpu = float(part.cpu.median()), float(part.gpu.median())
        speedup = float((part.cpu / part.gpu).median())
        faster = int((part.gpu < part.cpu).sum())
        low, high = int(part.merged_peaks.min()), int(part.merged_peaks.max())
        records.append({"fifth": fifth, "minimum_peaks": low, "maximum_peaks": high, "structures": len(part),
                        "cpu_seconds": cpu, "a100_seconds": gpu, "speedup": speedup, "a100_faster": faster})
        cells.append([f"{low:,}--{high:,}", f"{cpu:.3f}", f"{gpu:.3f}", f"{speedup:.2f}", f"{faster}/40"])
    write(output, "single_timing", ["Merged peaks", "CPU", "A100", "Speedup", "A100 faster"], records, cells)
    records, cells = [], []
    for path, rows, batches in (("Merged", merged, (1, 4, 16, 32)), ("Separate", separate, (1, 4))):
        expected = {(case, mode, batch, device) for case in counts.index
                    for mode in ("forward", "wls_gradient") for batch in batches for device in ("cpu", "cuda:0")}
        keys = list(rows[["case_id", "mode", "batch_size", "device"]].itertuples(index=False, name=None))
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError(f"Incomplete or duplicate {path} timing workloads")
        for mode, label in (("forward", "Profile"), ("wls_gradient", "Loss + gradient")):
            for batch in batches:
                selected = rows.loc[rows["mode"].eq(mode) & rows.batch_size.eq(batch)]
                pair = paired(selected, "device", "median_runtime_sec", "cpu", "cuda:0")
                cpu, gpu = statistics(pair.cpu * 1000 / batch), statistics(pair.gpu * 1000 / batch)
                speedup = statistics(pair.cpu / pair.gpu)
                faster = int((pair.gpu < pair.cpu).sum())
                records.append(dict(path=path, mode=mode, batch=batch, structures=200, a100_faster=faster,
                    **{f"{name}_{stat}": value for name, values in (("cpu_ms", cpu), ("a100_ms", gpu), ("speedup", speedup))
                       for stat, value in zip(("median", "q25", "q75"), values, strict=True)}))
                cells.append([path if mode == "forward" and batch == 1 else "", label if batch == 1 else "",
                              str(batch), interval(cpu, time=True), interval(gpu, time=True), interval(speedup), f"{faster}/200"])
    write(output, "repeated_timing", ["Path", "Mode", "Batch", "CPU [ms]", "A100 [ms]", "Speedup", "A100 faster"], records, cells)
