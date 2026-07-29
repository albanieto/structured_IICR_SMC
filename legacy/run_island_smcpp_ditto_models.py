### Run island models with SMC++ and Ditto
### Same idea as run_island_models.py, but matching:
### run_example_fim_M1_d50_N1400_smcpp_ditto.sh

import subprocess


def calc_m(M, N=1000, d=20):
    # M = 2N*m(d-1)
    # m = M/((d-1)*2N)
    m = "{:.4E}".format(M / ((d - 1) * 2 * N))
    return m


def call_gyarados(
    d,
    N,
    m,
    M,
    Nt,
    mode="iicr,simulate,stats,smcpp",
    psmc_patterns="4+25*2+4+6",
    psmc_s=100,
    samples=20,
    sizes="100000000",
    chromosomes=1,
    iterations=1,
    population=1,
    mu="1e-8",
    ditto=True,
    dry_run=False,
):
    M = str(M)
    cmd = [
        "sbatch", "gyarados_call.sh",
        "-n", str(d),
        "-N", str(N),
        "-m", str(m),
        "-b", str(sizes),
        "-c", str(chromosomes),
        "-t", "1",
        "-p", str(population),
        "-i", str(iterations),
        "-d", str(mu),
        "-s", str(samples),
        "-M", M,
        "-P", str(Nt),
        "--mode", str(mode),
        "--psmc-s", str(psmc_s),
        "--psmc-patterns", str(psmc_patterns),
    ]
    if ditto:
        cmd.extend(["--ditto", "true"])

    sentence = " ".join(cmd)
    print(d, N, m, M, Nt)
    print(sentence)
    if not dry_run:
        subprocess.run(cmd, check=True)
    return sentence


# Vectors to run.
# If N_vector is empty, N is calculated from N_tot and d.
N_tot = [70000]
N_vector = [1400]
d_vector = [5, 10, 20, 50]
M_vector = [1, 2.5, 5, 15, 50]

# Same settings as run_example_fim_M1_d50_N1400_smcpp_ditto.sh,
# except CHROMOSOMES is 1 here.
MODE = "iicr,simulate,stats,smcpp"
PSMC_S = 100
PSMC_PATTERNS = "4+25*2+4+6"
SAMPLES = 20
SIZES = "100000000"
CHROMOSOMES = 1
ITERATIONS = 1
POPULATION = 1
MU = "1e-8"
DITTO = True
DRY_RUN = False


if N_vector:
    for M in M_vector:
        for N in N_vector:
            for d in d_vector:
                m = calc_m(M, N, d)
                print(m)
                Nt = N * d
                call_gyarados(
                    d,
                    N,
                    m,
                    M,
                    Nt,
                    mode=MODE,
                    psmc_patterns=PSMC_PATTERNS,
                    psmc_s=PSMC_S,
                    samples=SAMPLES,
                    sizes=SIZES,
                    chromosomes=CHROMOSOMES,
                    iterations=ITERATIONS,
                    population=POPULATION,
                    mu=MU,
                    ditto=DITTO,
                    dry_run=DRY_RUN,
                )
else:
    for M in M_vector:
        for Nt in N_tot:
            for d in d_vector:
                N = Nt // d
                if N % 2 != 0:
                    print(N, "no par")
                    N = N + 1
                    print("New N is", N)
                m = calc_m(M, N, d)
                print(m)
                call_gyarados(
                    d,
                    N,
                    m,
                    M,
                    Nt,
                    mode=MODE,
                    psmc_patterns=PSMC_PATTERNS,
                    psmc_s=PSMC_S,
                    samples=SAMPLES,
                    sizes=SIZES,
                    chromosomes=CHROMOSOMES,
                    iterations=ITERATIONS,
                    population=POPULATION,
                    mu=MU,
                    ditto=DITTO,
                    dry_run=DRY_RUN,
                )
