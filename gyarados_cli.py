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


VERSION = "0.5.0"

MODEL_TYPES = ("fim", "2dsst", "1dsst", "panmictic", "free")

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


def calculate_stepping_stone_migration(M: float, population_size: int) -> float:
    return M / population_size


def write_1dsst_par(
    path: Path,
    population_size: int,
    demes: int,
    migration: float,
    sampled_populations: Iterable[int],
    samples: int,
    sizes: Iterable[int],
    mu: float,
    rho: float,
) -> None:
    """Write the free-model parameter file for a one-dimensional stepping stone."""
    sizes = list(sizes)
    selected = set(sampled_populations)
    with path.open("w") as stream:
        stream.write("//Number of population samples (demes)\n%d\n" % demes)
        stream.write("//Population effective sizes (number of genes 2*diploids)\n")
        for _ in range(demes):
            stream.write("%d\n" % population_size)
        stream.write("//Sample sizes (number of genes 2*diploids)\n")
        for deme in range(1, demes + 1):
            stream.write("%d\n" % (samples if deme in selected else 0))
        stream.write("//Grow rates: negative grow rates implies population expansion\n")
        for _ in range(demes):
            stream.write("0\n")
        stream.write("//Number of migration matrixes\n1\n")
        stream.write("//migration matrix\n")
        for source in range(demes):
            row = [
                str(migration) if abs(source - destination) == 1 else "0"
                for destination in range(demes)
            ]
            stream.write(" ".join(row) + "\n")
        stream.write(
            "//historical event: time, source, sink, migrants, new deme size, "
            "new growth rate, migration matrix index\n0 events\n"
        )
        stream.write(
            "//Number of independent loci [chromosome] "
            "(Number of sequences of 300 bp per gamete)\n%d %d\n"
            % (len(sizes), 0 if len(set(sizes)) == 1 else 1)
        )
        for size in sizes:
            stream.write("//Chromosome structure 1 begins with number of loci\n1\n")
            stream.write(
                "//per block: data type, number of loci, per generation "
                "recombination and mutation rates and optional parameters\n"
            )
            stream.write("DNA %d %s %s\n" % (size, rho, mu))


def run(args: argparse.Namespace) -> int:
    modes = gyarados.parse_run_modes(args.mode)
    tool_config = load_tool_config(args.config)
    programs = configure_programs(tool_config, modes, strict=not args.dry_run)
    if args.model == "fim":
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
        model_demes = args.demes
    elif args.model in {"1dsst", "2dsst"}:
        migration = (
            args.migration
            if args.migration is not None
            else calculate_stepping_stone_migration(args.M, args.population_size)
        )
        scaled_migration = (
            args.M
            if args.M is not None
            else args.population_size * migration
        )
        model_demes = args.demes if args.model == "1dsst" else args.grid_size ** 2
    else:
        migration = None
        scaled_migration = None
        model_demes = 1 if args.model == "panmictic" else None

    output_dir = Path(args.output_dir).expanduser().resolve()
    par_file = (
        Path(args.par_file).expanduser().resolve()
        if args.par_file is not None
        else None
    )

    configuration = {
        "model": args.model,
        "output_dir": str(output_dir),
        "demes": model_demes,
        "grid_size": args.grid_size if args.model == "2dsst" else None,
        "par_file": str(par_file) if par_file is not None else None,
        "haploid_population_size_per_deme": (
            None if args.model == "free" else args.population_size
        ),
        "migration_rate": migration,
        "M": scaled_migration,
        "sampled_haploid_genomes": None if args.model == "free" else args.samples,
        "chromosome_sizes": None if args.model == "free" else args.sizes,
        "number_of_chromosomes": (
            None if args.model == "free" else args.chromosomes
        ),
        "iterations": args.iterations,
        "sampled_populations": (
            None if args.model == "free" else args.sampled_populations
        ),
        "mutation_rate": None if args.model == "free" else args.mu,
        "recombination_rate": None if args.model == "free" else args.rho,
        "modes": sorted(modes),
        "iicr_T2_simulations": gyarados.IICR_T2_SIMULATIONS,
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
        common = {
            "niter": args.iterations,
            "mode": args.mode,
            "psmc_patterns": ",".join(args.psmc_pattern),
            "psmc_s": args.psmc_s,
            "ditto": args.ditto,
            "backend": "local",
        }
        sizes = ",".join(str(size) for size in args.sizes)

        if args.model == "fim":
            gyarados.GYARADOS_PAR(
                type=1,
                N=args.population_size,
                sizes=sizes,
                sample=args.samples,
                chr=len(args.sizes),
                nislands=args.demes,
                mig=migration,
                p=args.sampled_populations[0],
                mu=args.mu,
                rho=args.rho,
                evs="0",
                gr=0,
                M=scaled_migration,
                Nt=args.population_size * args.demes,
                **common,
            )
        elif args.model == "2dsst":
            gyarados.GYARADOS_PAR(
                type=4,
                L=args.grid_size,
                N=args.population_size,
                sizes=sizes,
                sample=args.samples,
                chr=len(args.sizes),
                mig=migration,
                p=",".join(str(population) for population in args.sampled_populations),
                mu=args.mu,
                rho=args.rho,
                evs="0",
                gr=0,
                M=scaled_migration,
                **common,
            )
        elif args.model == "1dsst":
            generated_par = output_dir / (
                "1DSST_d%d_N%d_m%.4E.par"
                % (args.demes, args.population_size, migration)
            )
            write_1dsst_par(
                generated_par,
                args.population_size,
                args.demes,
                migration,
                args.sampled_populations,
                args.samples,
                args.sizes,
                args.mu,
                args.rho,
            )
            gyarados.GYARADOS_PAR(
                type=3,
                par=str(generated_par),
                p=args.sampled_populations[0],
                **common,
            )
        elif args.model == "panmictic":
            gyarados.GYARADOS_PAR(
                type=2,
                N=args.population_size,
                sizes=sizes,
                sample=args.samples,
                chr=len(args.sizes),
                p=1,
                mu=args.mu,
                rho=args.rho,
                evs="0",
                gr=0,
                **common,
            )
        else:
            gyarados.GYARADOS_PAR(
                type=3,
                par=str(par_file),
                p=1,
                **common,
            )
    finally:
        os.chdir(previous_dir)
    print("\nFinished. Results are in:", output_dir / "RESULTS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gyarados",
        description="Run the Gyarados structured-IICR pipeline.",
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
        "run", help="Run one demographic model sequentially on this machine."
    )
    run_parser.add_argument(
        "--model",
        choices=MODEL_TYPES,
        default="fim",
        help="Demographic model: fim, 2dsst, 1dsst, panmictic, or free.",
    )
    run_parser.add_argument(
        "--output-dir",
        default="gyarados_output",
        help="Working/output directory (default: ./gyarados_output).",
    )
    run_parser.add_argument(
        "--grid-size",
        type=int,
        default=3,
        help="Side length L of the square grid used by 2dsst (default: 3).",
    )
    run_parser.add_argument(
        "--par-file",
        help="Existing fastsimcoal2 .par file required by the free model.",
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
        help="M=2*N*m*(d-1) for fim and M=N*m for stepping stones (default: 5).",
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
    run_parser.add_argument(
        "--chromosomes",
        type=int,
        default=1,
        help="Number of chromosomes (default: 1).",
    )
    run_parser.add_argument("--iterations", type=int, default=1)
    run_parser.add_argument(
        "--sampled-population",
        dest="sampled_populations",
        type=int,
        action="append",
        default=None,
        help=(
            "Population used for inference. Repeat for several populations "
            "in 1dsst/2dsst. The default is population 1 for FIM and "
            "panmictic; free reads sampling from its .par file."
        ),
    )
    run_parser.add_argument("--mu", type=float, default=1e-8)
    run_parser.add_argument("--rho", type=float, default=1e-8)
    run_parser.add_argument(
        "--mode",
        default="iicr,simulate,stats,psmc",
        help="Comma-separated: iicr,simulate,stats,psmc,smcpp,transition_matrix,none.",
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
    if args.chromosomes <= 0:
        parser.error("--chromosomes must be positive.")
    if args.model == "free":
        args.sizes = []
    elif args.sizes is None:
        args.sizes = [10_000_000] * args.chromosomes
    elif len(args.sizes) == 1 and args.chromosomes > 1:
        args.sizes = args.sizes * args.chromosomes
    elif len(args.sizes) != args.chromosomes:
        parser.error(
            "Provide one --size for all chromosomes or one --size per chromosome."
        )
    if args.M is None and args.migration is None:
        args.M = 5.0
    if args.psmc_pattern is None:
        args.psmc_pattern = [gyarados.DEFAULT_PSMC_PATTERN]
    populations_were_provided = args.sampled_populations is not None
    if args.model == "free":
        if populations_were_provided:
            parser.error(
                "Sampling for a free model is defined inside its --par-file."
            )
        args.sampled_populations = []
    elif args.sampled_populations is None:
        if args.model == "1dsst":
            args.sampled_populations = [(args.demes + 1) // 2]
        elif args.model == "2dsst":
            args.sampled_populations = [(args.grid_size ** 2 + 1) // 2]
        else:
            args.sampled_populations = [1]
    else:
        args.sampled_populations = list(dict.fromkeys(args.sampled_populations))
    if args.model in {"fim", "1dsst"} and args.demes < 2:
        parser.error("--demes must be at least 2 for fim and 1dsst.")
    if args.model == "2dsst" and args.grid_size < 2:
        parser.error("--grid-size must be at least 2 for 2dsst.")
    if args.model == "free":
        if args.par_file is None:
            parser.error("--par-file is required when --model free is selected.")
        if not Path(args.par_file).expanduser().is_file():
            parser.error("--par-file does not exist: %s" % args.par_file)
    if args.population_size <= 0:
        parser.error("--population-size must be positive.")
    if args.M is not None and args.M <= 0:
        parser.error("--M must be positive.")
    if args.migration is not None and args.migration <= 0:
        parser.error("--migration must be positive.")
    if args.samples <= 0:
        parser.error("--samples must be positive.")
    if (
        args.model != "free"
        and "psmc" in gyarados.parse_run_modes(args.mode)
        and args.samples % 2
    ):
        parser.error("--samples must be even when PSMC is requested.")
    if (
        args.model in {"fim", "1dsst"}
        and any(
            population < 1 or population > args.demes
            for population in args.sampled_populations
        )
    ):
        parser.error("--sampled-population must be between 1 and --demes.")
    if (
        args.model == "2dsst"
        and any(
            population < 1 or population > args.grid_size ** 2
            for population in args.sampled_populations
        )
    ):
        parser.error("--sampled-population must be within the 2dsst grid.")
    if args.model == "panmictic" and args.sampled_populations != [1]:
        parser.error("Only population 1 exists in a panmictic model.")
    if any(size <= 0 for size in args.sizes):
        parser.error("Every --size must be positive.")
    if args.mu <= 0 or args.rho < 0:
        parser.error("--mu must be positive and --rho cannot be negative.")
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
