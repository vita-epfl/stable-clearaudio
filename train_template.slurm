#!/bin/bash
#SBATCH --job-name=stable-clearaudio
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=00:02:00
#SBATCH --partition=h100
#SBATCH --output=/work/vita/alefevre/programs/stable-clearaudio/slurm/logs/%j.out
#SBATCH --error=/work/vita/alefevre/programs/stable-clearaudio/slurm/logs/%j.err


mkdir -p /work/vita/alefevre/programs/stable-clearaudio/slurm/logs

export WANDB_API_KEY=<WANDB_API_KEY>

cd /work/vita/alefevre/programs/stable-clearaudio

source /work/vita/alefevre/programs/stable-clearaudio/venv/bin/activate

echo STARTING AT `date`

srun --gpu-bind=closest /bin/bash -c "
    python train.py \
      --model-config stable_audio_tools/configs/model_configs/audio_restoration/cold_diffusion_uncond.json \
      --dataset-config stable_audio_tools/configs/dataset_configs/cold_diffusion_configs/SCITAS_dataset_configs/training/intense_eq_ranges.json \
      --val-dataset-config stable_audio_tools/configs/dataset_configs/cold_diffusion_configs/SCITAS_dataset_configs/valid/intense_eq_ranges_valid.json \
      --val-every-n-epoch 1 \
      --checkpoint-every-n-epoch 1 \
      --early-stopping false \
      --save-top-k 1 \
      --name stable-clearaudio-cd-scitas \
      --save-dir stable_audio_tools/output/checkpoints \
      --strategy ddp_find_unused_parameters_true \
      --batch-size 128 \
      --val-batch-size 64
    "

echo FINISHED at `date`