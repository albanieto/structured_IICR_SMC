"""Ditto: turn a simulated IICR into a panmictic fastsimcoal2 model."""

import os
import shlex

import numpy as np
import pandas as pd


def sampled_demes_for_model(model, p=1):
    if model.mtype == "SST":
        return list(map(int, model.sampled_demes))
    if model.mtype == "Free":
        raw_sampling_demes = list(range(1, model.islands + 1))
        return [deme for deme, val in zip(raw_sampling_demes, model.samples) if int(val) != 0]
    return [int(str(p).split(",")[0])]


def iicr_path(model, deme):
    if isinstance(model.iicr_log_path, dict):
        return model.iicr_log_path[int(deme)]
    return model.iicr_log_path


def interpolate(x, y, new_x, kind="linear"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    new_x = np.asarray(new_x, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_index = np.unique(x, return_index=True)
    x = unique_x
    y = y[unique_index]

    if kind == "nearest-up":
        idx = np.searchsorted(x, new_x, side="left")
        idx = np.clip(idx, 0, len(y) - 1)
        return y[idx]
    if kind == "cubic":
        try:
            from scipy.interpolate import interp1d

            f_interp = interp1d(x, y, kind="cubic", bounds_error=False, fill_value="extrapolate")
            return f_interp(new_x)
        except Exception:
            print("Cubic interpolation unavailable for Ditto; falling back to linear.")
    return np.interp(new_x, x, y, left=y[0], right=y[-1])


def read_iicr(filepath, preprocess=True, cdf_threshold=9800000, log=False, pool=True, num_windows=50, deviation=0.01):
    """
    Read one simulated IICR and simplify it into change-points for Ditto.
    """
    df_org = pd.read_csv(filepath, sep="\t", compression="gzip")
    df_org = df_org.drop(columns=[col for col in ["lmd", "x", "f_x"] if col in df_org.columns])

    required = {"scaled_lmd", "F_x", "generations"}
    if not required.issubset(set(df_org.columns)):
        raise ValueError("File %s is missing required IICR columns: %s" % (filepath, ", ".join(sorted(required))))

    df_result = df_org
    if preprocess:
        generations = df_org["generations"].values
        df = df_org.dropna(subset=["scaled_lmd", "F_x", "generations"])
        df_filtered = df[df["F_x"] <= cdf_threshold]
        df_filtered = df_filtered[df_filtered["scaled_lmd"] > 0]

        time_values = df_filtered["generations"].values
        iicr_values = df_filtered["scaled_lmd"].values
        if log:
            iicr_values = np.log(iicr_values)

        if len(time_values) < 2:
            raise ValueError("Not enough valid time points in %s after filtering." % filepath)

        interpolated_iicr = interpolate(time_values, iicr_values, generations, kind="nearest-up")

        df_result = pd.DataFrame({"generations": generations, "scaled_lmd": interpolated_iicr})
        if pool and len(generations) >= num_windows:
            ntimes = len(generations)
            new_len = (ntimes // num_windows) * num_windows
            if new_len >= num_windows:
                generations_trim = generations[:new_len]
                iicr_trim = interpolated_iicr[:new_len]
                window_size = len(generations_trim) // num_windows
                gens_reshaped = generations_trim.reshape(num_windows, window_size)
                iicr_reshaped = iicr_trim.reshape(num_windows, window_size)
                df_result = pd.DataFrame({
                    "generations": gens_reshaped.mean(axis=1).astype(int),
                    "scaled_lmd": iicr_reshaped.mean(axis=1).astype(int),
                })

    df_result = df_result.dropna(subset=["generations", "scaled_lmd"])
    df_result = df_result[df_result["scaled_lmd"] > 0].reset_index(drop=True)
    if deviation and len(df_result) > 1:
        kept = [0]
        last_val = float(df_result.loc[0, "scaled_lmd"])
        for idx in range(1, len(df_result)):
            curr = float(df_result.loc[idx, "scaled_lmd"])
            if last_val == 0 or abs(curr - last_val) / last_val >= deviation:
                kept.append(idx)
                last_val = curr
        df_result = df_result.loc[kept].reset_index(drop=True)

    return df_result


def transform_iicr(df, interp="linear", ext_events=100):
    df = df[["generations", "scaled_lmd"]].dropna()
    df = df[df["scaled_lmd"] > 0].sort_values("generations").drop_duplicates("generations").reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Ditto needs at least two IICR points to build historical events.")

    gens = df["generations"].astype(float).values
    iicr = df["scaled_lmd"].astype(float).values
    N0_haploid = max(1, int(round(2 * iicr[0])))

    if interp in ("linear", "cubic"):
        positive_gens = gens[gens > 0]
        if len(positive_gens) == 0:
            raise ValueError("Ditto cannot build historical events without positive generations.")
        min_g = positive_gens.min()
        max_g = positive_gens.max()
        dense_gens = np.logspace(np.log10(min_g), np.log10(max_g), ext_events)
        df_sampled = pd.DataFrame({
            "generations": dense_gens,
            "scaled_lmd": interpolate(gens, iicr, dense_gens, kind=interp),
        })
    else:
        df_sampled = df[df["generations"] > 0].reset_index(drop=True)

    events = []
    for idx in range(len(df_sampled)):
        gen = int(round(float(df_sampled.loc[idx, "generations"])))
        N_haploid = max(1, int(round(2 * float(df_sampled.loc[idx, "scaled_lmd"]))))
        if gen > 0:
            events.append((gen, N_haploid))

    if events:
        ev_df = pd.DataFrame(events, columns=["generation", "N_haploid"])
        ev_df = ev_df.drop_duplicates(subset=["generation"], keep="first")
        events = list(zip(ev_df["generation"].astype(int), ev_df["N_haploid"].astype(int)))

    return N0_haploid, events


def write_panmictic_par(filename, N0, events, source_model, sample_size):
    n_events = len(events)
    sizes = list(map(int, source_model.sizes))

    lines = [
        "// Number of population samples (demes)",
        "1",
        "// Population effective sizes (number of genes)",
        str(int(N0)),
        "// Sample sizes",
        str(int(sample_size)),
        "// Growth rates",
        "0",
        "// Number of migration matrices : 0 implies no migration",
        "0",
        "// Historical events: time, source, sink, migrants, new size, new growth, new migr matrix",
        str(n_events) + " events",
    ]
    for t, Nnew in events:
        lines.append("%s    0    0    0    %s    0    0    absoluteResize" % (int(t), int(Nnew)))

    lines.append("// Number of independent loci [chromosome] (Number of sequences of 300 bp per gamete)")
    lines.append(str(int(source_model.chr)) + " 0")
    for c in sizes:
        lines.append("// Chromosome structure 1 begins with number of loci")
        lines.append("1")
        lines.append("// per block: data type, number of loci, per generation recombination and mutation rates and optional parameters")
        lines.append("DNA  %s %s %s" % (int(c), source_model.rho, source_model.mu))

    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")

    return filename


def generate_par_from_iicr(iicr_filepath, source_model, deme, sample_size=None):
    if sample_size is None:
        sample_size = source_model.samples[int(deme) - 1]
    sample_size = int(sample_size)
    if sample_size <= 0:
        sample_size = 2

    ditto_dir = os.path.join(source_model.name, source_model.name + "_DITTO")
    os.system("mkdir -p " + shlex.quote(ditto_dir))
    base = source_model.name + "_deme_" + str(deme) + "_ditto"
    filename = os.path.join(ditto_dir, base + ".par")

    df = read_iicr(iicr_filepath)
    N0, events = transform_iicr(df)
    par_file = write_panmictic_par(filename, N0, events, source_model, sample_size)
    print("Generated Ditto panmictic par:", par_file)
    return par_file


def run_ditto_for_model(source_model, p, niter, mode, psmc_patterns=None, psmc_s=100, run_gyarados=None):
    if run_gyarados is None:
        raise ValueError("run_gyarados callback is required to launch Ditto models.")

    demes = sampled_demes_for_model(source_model, p)
    print("Ditto enabled. Creating panmictic mimic(s) for demes:", demes)
    for deme in demes:
        iicr_filepath = iicr_path(source_model, deme)
        if not os.path.exists(iicr_filepath):
            raise FileNotFoundError(
                "Ditto requested, but IICR file is missing: %s. Run with mode including iicr first." % iicr_filepath
            )
        par_file = generate_par_from_iicr(iicr_filepath, source_model=source_model, deme=deme)
        run_gyarados(
            type=3,
            par=par_file,
            niter=niter,
            p=1,
            mode=mode,
            psmc_patterns=psmc_patterns,
            psmc_s=psmc_s,
            ditto=False,
        )
