# Training with SLURM

To launch training, run the following command:
```bash
sbatch -A <account> launcher.slurm $RUN $WORKDIR $TRAIN $VAL --project $PROJECT
```
where:
- `<account>` is the SLURM account to use
- `$RUN` is the name of the run
- `$WORKDIR` is the working directory
- `$TRAIN` is the path to the training manifest
- `$VAL` is the path to the validation manifest
- `$PROJECT` is the name of the project

The manifest files are CSV files with the columns:
- `fileid`: unique identifier
- `path`: absolute path to the audio file
- `num_frames`: number of frames in the full file
- `speaker`: speaker identifier

## Customization

If you do not use `micromamba` to manage your environment you need to
adapt this script so that it activates the environment.

If you want to use a different configuration set the `FASTCPC_CONFIG` environment
variable to the path of the JSON configuration file.
The entries in the configuration file will override the default ones.
Set it so that it is visible by the SLURM job.

If you want to train on more nodes, modify:
- the `#SBATCH --nodes` line in `launcher.slurm`
- the `num_machines` field in `accelerate.yaml`

I you want to run on a different number of GPUs per node, modify:
- the `#SBATCH --gres` line in `launcher.slurm`
- the `GPUS_PER_NODE` variable in `launcher.slurm`
- the `num_processes` field in `accelerate.yaml`