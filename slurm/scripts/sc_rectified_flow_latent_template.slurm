#!/bin/bash
#SBATCH --job-name=sc-rectified_flow-latent
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --partition=h100
#SBATCH --output=/work/vita/alefevre/programs/stable-clearaudio/slurm/logs/%j.out
#SBATCH --error=/work/vita/alefevre/programs/stable-clearaudio/slurm/logs/%j.err


mkdir -p /work/vita/alefevre/programs/stable-clearaudio/slurm/logs

# Create a directory for local installations
mkdir -p ~/local/sox > /dev/null 2>&1
cd ~/local > /dev/null 2>&1

# Download and extract SoX source (silently)
wget -q https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz
tar -xzf sox-14.4.2.tar.gz > /dev/null 2>&1
cd sox-14.4.2 > /dev/null 2>&1

# Configure and install to your local directory (silently)
./configure --prefix=$HOME/local/sox > /dev/null 2>&1
make > /dev/null 2>&1
make install > /dev/null 2>&1

# Add to your PATH and LD_LIBRARY_PATH in your SLURM script
export PATH=$HOME/local/sox/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/sox/lib:$LD_LIBRARY_PATH

export WANDB_API_KEY=541eebad1b2e17e39790984bdb74a12fbc548828

cd /work/vita/alefevre/programs/stable-clearaudio

source /work/vita/alefevre/programs/stable-clearaudio/venv/bin/activate

echo STARTING AT `date`

srun --gpu-bind=closest /bin/bash -c "
    python train.py \
      --pretransform-ckpt-path stable_audio_tools/checkpoints/vae/vae_model.ckpt \
      --model-config stable_audio_tools/configs/model_configs/audio_restoration/rectified_flow_latent.json \
      --dataset-config stable_audio_tools/configs/dataset_configs/cold_diffusion_configs/SCITAS_dataset_configs/training/intense_eq_ranges.json \
      --val-dataset-config stable_audio_tools/configs/dataset_configs/cold_diffusion_configs/SCITAS_dataset_configs/valid/intense_eq_ranges_valid.json \
      --checkpoint-every-n-epoch 1 \
      --early-stopping false \
      --save-top-k -1 \
      --save-last-checkpoint true \
      --name sc-rectified_flow-latent \
      --save-dir stable_audio_tools/output/checkpoints \
      --batch-size 128
    "

echo FINISHED at `date`