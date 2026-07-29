#!/usr/bin/env python3
"""Portable command-line interface for the Gyarados pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, Iterable

import gyarados
import yaml


VERSION = "0.3.0"

PROGRAMS = (
    "fastsimcoal2",
    "psmc",
    "smcpp",
)


def default_config_path() -> Path:
    """Find the editable configuration without guessing any tool commands."""
    if os.environ.get("GYARADOS_CONFIG"):
        return Path(os.environ["GYARADOS_CONFIG"]).expanduser().resolve()

    candidates = (
        Path.cwd() / "tools.yml",
        Path(__file__).with_name("tools.yml"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def load_tool_config(config_path: str) -> Dict[str, object]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Tool configuration not found: %s. "
            "Pass --config /path/to/tools.yml." % path
        )

    with path.open() as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Tool configuration must contain a YAML mapping.")
    data["_path"] = str(path)
    return data


def tool_command(config: Dict[str, object], key: str) -> str:
    command = str(config.get(key, "")).strip()
    return os.path.expandvars(os.path.expanduser(command))


def is_configured(command: str) -> bool:
    return (
        bool(command)
        and command.lower() != "none"
        and "EDIT_" not in command
        and "/EDIT/" not in command
    )


def environment_is_configured(environment: str) -> bool:
    return environment.lower() == "none" or is_configured(environment)


def required_programs(modes: Iterable[str]) -> set:
    modes = set(modes)
    required = {"fastsimcoal2"} if modes else set()
    if "psmc" in modes:
        required.add("psmc")
    if "smcpp" in modes:
        required.add("smcpp")
    return required


def configure_programs(
    config: Dict[str, object], modes: Iterable[str], strict: bool
) -> Dict[str, str]:
    required = required_programs(modes)
    configured = {}
    unconfigured = []
    smcpp_environment = tool_command(config, "smcpp_environment")
    commands = {
        "seqtk": "seqtk",
        "bgzip": "bgzip",
        "tabix": "tabix",
    }

    for key in PROGRAMS:
        command = tool_command(config, key)
        configured[key] = command
        if is_configured(command):
            commands[key] = command
        elif key in required:
            unconfigured.append(key)

    configured["smcpp_environment"] = smcpp_environment
    if (
        "smcpp" in required
        and not environment_is_configured(smcpp_environment)
    ):
        unconfigured.append("smcpp_environment")

    if strict and unconfigured:
        raise RuntimeError(
            "Required tool setting(s) are not configured: %s.\n"
            "Edit %s and replace every EDIT_... value needed by this mode.\n"
            "Gyarados does not guess executable names or paths."
            % (", ".join(sorted(unconfigured)), config["_path"])
        )

    if "psmc" in commands:
        psmc_path = Path(commands["psmc"])
        commands["fq2psmcfa"] = str(
            psmc_path.parent / "utils" / "fq2psmcfa"
        )

    if (
        "smcpp" in commands
        and smcpp_environment.lower() != "none"
        and environment_is_configured(smcpp_environment)
    ):
        commands["smcpp"] = (
            "conda run --no-capture-output -n "
            + smcpp_environment
            + " "
            + commands["smcpp"]
        )

    gyarados.configure_tools(commands)
    return configured


def doctor(args: argparse.Namespace) -> int:
    import numpy
    import pandas

    config = load_tool_config(args.config)
    configured = configure_programs(config, (), strict=False)

    print("Gyarados", VERSION)
    print("Configuration", config["_path"])
    print("Python", sys.version.split()[0], "-", sys.executable)
    print("NumPy", numpy.__version__)
    print("pandas", pandas.__version__)
    try:
        import scipy

        print("SciPy", scipy.__version__)
    except ImportError:
        print("SciPy MISSING (only needed for Ditto's preferred interpolation)")

    print("\nConfigured external programs")
    for key in PROGRAMS:
        command = configured[key]
        status = "CONFIGURED" if is_configured(command) else "EDIT"
        print("  [%-10s] %-16s %s" % (status, key, command))
    environment = configured["smcpp_environment"]
    status = (
        "CONFIGURED"
        if environment_is_configured(environment)
        else "EDIT"
    )
    print("  [%-10s] %-16s %s" % (status, "smcpp environment", environment))

    print("\nRequired combinations")
    print("  All runs:         fastsimcoal2")
    print("  PSMC:            psmc installation + seqtk from gyarados")
    print("  SMC++:           smcpp + bgzip/tabix from gyarados")
    print("\nGyarados uses these values exactly; it does not guess other names.")
    return 0


def calculate_migration(M: float, population_size: int, demes: int) -> float:
    return M / ((demes - 1) * 2.0 * population_size)


def run(args: argparse.Namespace) -> int:
    modes = gyarados.parse_run_modes(args.mode)
    tool_config = load_tool_config(args.config)
    programs = configure_programs(tool_config, modes, strict=not args.dry_run)
    migration = (
        args.migration
        if args.migration is not None
        else calculate_migration(args.M, args.population_size, args.demes)
    )
    scaled_migration = (
        args.M
        if args.M is not None
        else 2.0 * args.population_size * migration * (args.demes - 1)
    )
    output_dir = Path(args.output_dir).expanduser().resolve()

    configuration = {
        "model": "symmetric island",
        "output_dir": str(output_dir),
        "demes": args.demes,
        "haploid_population_size_per_deme": args.population_size,
        "migration_rate": migration,
        "M": scaled_migration,
        "sampled_haploid_genomes": args.samples,
        "chromosome_sizes": args.sizes,
        "iterations": args.iterations,
        "sampled_deme": args.population,
        "mutation_rate": args.mu,
        "recombination_rate": args.rho,
        "modes": sorted(modes),
        "iicr_replicates": args.iicr_replicates,
        "psmc_s": args.psmc_s,
        "psmc_patterns": args.psmc_pattern,
        "ditto": args.ditto,
        "tool_config": tool_config["_path"],
        "programs": {
            key: programs[key]
            for key in PROGRAMS
        },
        "smcpp_environment": programs["smcpp_environment"],
    }
    print(json.dumps(configuration, indent=2))
    if args.dry_run:
        print("\nDry run: no directories were created and no programs were executed.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    previous_dir = Path.cwd()
    try:
        os.chdir(output_dir)
        gyarados.GYARADOS_PAR(
            type=1,
            N=args.population_size,
            sizes=",".join(str(size) for size in args.sizes),
            sample=args.samples,
            niter=args.iterations,
            chr=len(args.sizes),
            nislands=args.demes,
            mig=migration,
            p=args.population,
            mu=args.mu,
            rho=args.rho,
            evs="0",
            gr=0,
            M=scaled_migration,
            Nt=args.population_size * args.demes,
            mode=args.mode,
            psmc_patterns=",".join(args.psmc_pattern),
            psmc_s=args.psmc_s,
            ditto=args.ditto,
            backend="local",
            iicr_replicates=args.iicr_replicates,
        )
    finally:
        os.chdir(previous_dir)
    print("\nFinished. Results are in:", output_dir / "RESULTS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gyarados",
        description=(
            "Run the Gyarados structured-IICR pipeline locally. "
            "No SLURM installation is required."
        ),
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + VERSION)
    parser.add_argument(
        "--config",
        default=str(default_config_path()),
        help="Editable YAML file containing exact external-tool calls.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Report Python dependencies and parsed YAML tool settings."
    )
    doctor_parser.set_defaults(function=doctor)

    run_parser = subparsers.add_parser(
        "run", help="Run a symmetric island model sequentially on this machine."
    )
    run_parser.add_argument(
        "--output-dir",
        default="gyarados_output",
        help="Working/output directory (default: ./gyarados_output).",
    )
    run_parser.add_argument("--demes", type=int, default=5)
    run_parser.add_argument(
        "--population-size",
        type=int,
        default=1400,
        help="Haploid population size per deme.",
    )
    migration_group = run_parser.add_mutually_exclusive_group()
    migration_group.add_argument(
        "--M",
        type=float,
        default=None,
        help="Scaled migration M=2*N*m*(d-1) (default: 5).",
    )
    migration_group.add_argument(
        "--migration",
        type=float,
        help="Per-generation migration rate m; overrides calculation from M.",
    )
    run_parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Sampled haploid genomes; must be even for PSMC (default: 2).",
    )
    run_parser.add_argument(
        "--size",
        dest="sizes",
        type=int,
        action="append",
        default=None,
        help="Chromosome length in bp. Repeat for multiple chromosomes.",
    )
    run_parser.add_argument("--iterations", type=int, default=1)
    run_parser.add_argument("--population", type=int, default=1)
    run_parser.add_argument("--mu", type=float, default=1e-8)
    run_parser.add_argument("--rho", type=float, default=1e-8)
    run_parser.add_argument(
        "--mode",
        default="iicr,simulate,stats,psmc",
        help="Comma-separated: iicr,simulate,stats,psmc,smcpp,transition_matrix,none.",
    )
    run_parser.add_argument(
        "--iicr-replicates",
        type=int,
        default=100000,
        help="Independent fastsimcoal2 loci used for the IICR (default: 100000).",
    )
    run_parser.add_argument("--psmc-s", type=int, default=100)
    run_parser.add_argument(
        "--psmc-pattern",
        action="append",
        default=None,
        help="PSMC -p vector. Repeat for more than one vector.",
    )
    run_parser.add_argument("--ditto", action="store_true")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print the configuration without executing.",
    )
    run_parser.set_defaults(function=run)
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.command != "run":
        return
    if args.sizes is None:
        args.sizes = [10_000_000]
    if args.M is None and args.migration is None:
        args.M = 5.0
    if args.psmc_pattern is None:
        args.psmc_pattern = [gyarados.DEFAULT_PSMC_PATTERN]
    if args.demes < 2:
        parser.error("--demes must be at least 2 for an island model.")
    if args.population_size <= 0:
        parser.error("--population-size must be positive.")
    if args.M is not None and args.M <= 0:
        parser.error("--M must be positive.")
    if args.migration is not None and args.migration <= 0:
        parser.error("--migration must be positive.")
    if args.samples <= 0:
        parser.error("--samples must be positive.")
    if "psmc" in gyarados.parse_run_modes(args.mode) and args.samples % 2:
        parser.error("--samples must be even when PSMC is requested.")
    if not 1 <= args.population <= args.demes:
        parser.error("--population must be between 1 and --demes.")
    if any(size <= 0 for size in args.sizes):
        parser.error("Every --size must be positive.")
    if args.mu <= 0 or args.rho < 0:
        parser.error("--mu must be positive and --rho cannot be negative.")
    if args.iicr_replicates <= 0:
        parser.error("--iicr-replicates must be positive.")
    if args.ditto and "iicr" not in gyarados.parse_run_modes(args.mode):
        parser.error("--ditto requires a mode containing iicr.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args, parser)
        return args.function(args)
    except (RuntimeError, ValueError, FileNotFoundError, OSError, yaml.YAMLError) as exc:
        parser.exit(2, "error: %s\n" % exc)


if __name__ == "__main__":
    raise SystemExit(main())
