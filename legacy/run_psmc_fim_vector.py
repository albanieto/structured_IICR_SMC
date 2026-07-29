#!/usr/bin/env python3
"""Submit island-model Gyarados runs with a configurable PSMC -p vector."""

import argparse
import subprocess


DEFAULT_PSMC_PATTERNS = ["4+25*2+4+6"]


def csv_ints(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def csv_floats(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def calc_m(M, N=1000, d=20):
    # M = 2N*m(d-1)
    # m = M/((d-1)*2N)
    return "{:.4E}".format(M / ((d - 1) * 2 * N))


def str_bool(value):
    return "true" if value else "false"


def build_command(args, d, N, M):
    m = calc_m(M, N, d)
    Nt = N * d
    patterns = args.psmc_patterns or DEFAULT_PSMC_PATTERNS
    cmd = [
        "sbatch", "gyarados_call.sh",
        "-n", str(d),
        "-N", str(N),
        "-m", str(m),
        "-b", args.sizes,
        "-c", str(args.chromosomes),
        "-t", "1",
        "-p", str(args.population),
        "-i", str(args.iterations),
        "-d", str(args.mu),
        "-s", str(args.samples),
        "-M", str(M),
        "-P", str(Nt),
        "--mode", args.mode,
        "--psmc-s", str(args.psmc_s),
        "--psmc-patterns", ",".join(patterns),
    ]
    if args.ditto:
        cmd.extend(["--ditto", str_bool(args.ditto)])
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Submit StSi island models with a configurable PSMC -p time vector."
    )
    parser.add_argument("--d-vector", default="50", help="Comma-separated deme counts.")
    parser.add_argument("--n-vector", default="1400", help="Comma-separated haploid deme sizes.")
    parser.add_argument("--m-vector", default="5", help="Comma-separated M values.")
    parser.add_argument("--samples", default=20, type=int, help="Haploid samples per model.")
    parser.add_argument("--sizes", default="100000000", help="Chromosome sizes passed to -b.")
    parser.add_argument("--chromosomes", default=5, type=int, help="Chromosome count passed to -c.")
    parser.add_argument("--iterations", default=1, type=int, help="Number of repetitions.")
    parser.add_argument("--population", default=1, type=int, help="Sampled deme.")
    parser.add_argument("--mu", default="1e-8", help="Mutation rate.")
    parser.add_argument(
        "--mode",
        default="iicr,simulate,stats,psmc",
        help="Comma-separated steps: iicr,simulate,stats,psmc,smcpp,transition_matrix or none.",
    )
    parser.add_argument(
        "--ditto",
        action="store_true",
        help="Also build and run the Ditto panmictic mimic from the simulated IICR.",
    )
    parser.add_argument("--psmc-s", default=100, type=int, help="fq2psmcfa -s value.")
    parser.add_argument(
        "--psmc-pattern",
        dest="psmc_patterns",
        action="append",
        help="PSMC -p vector. Can be repeated, but one vector per output directory is easiest to audit.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch commands without submitting.")
    args = parser.parse_args()

    for M in csv_floats(args.m_vector):
        for N in csv_ints(args.n_vector):
            for d in csv_ints(args.d_vector):
                cmd = build_command(args, d=d, N=N, M=M)
                print(" ".join(cmd))
                if not args.dry_run:
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
