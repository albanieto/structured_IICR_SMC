Gyarados: simulations & demographic inference for structured populations

Gyarados is a small toolkit + Slurm workflow for simulating structured populations and running demographic inference (PSMC, SMC++; optional SNIF/Stairway), with helpers to compute/collect IICR and summary stats. It supports Finite Island (StSi), 2D Stepping-Stone (SST), a panmictic baseline, and a Free model that reads a .par file. Core logic lives in gyarados.py.

✨ Features

Models

StSi (finite island) with symmetric migration

SST (2D stepping-stone) on an 
L×L
L×L grid; supports one or two sampled demes

Panmictic one-deme baseline

Free: parse a custom FastSimCoal2 .par and run the pipeline

Pipelines

End-to-end orchestration via GYARADOS_PAR(...) and GYARADOS_WORK(...)

Slurm job launchers for standard/high-mem tiers

Outputs & bookkeeping

Auto-creates a RESULTS/<model_name>/ folder with the model JSON/.par, compressed outputs, and logs

Generates dummy FASTA per chromosome so PSMC can run out-of-the-box

📂 Repo contents
