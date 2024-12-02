#!/bin/bash

# Parameters
learning_rates=(1e-3)  # List of learning rates
latent_dims=(64 32 16 8 4 2)         # List of latent dimensions

# Loop through parameter combinations
for lr in "${learning_rates[@]}"; do
  for ld in "${latent_dims[@]}"; do
    # Define job name and script name
    job_name="fm_lr_${lr}_ld_${ld}"
    script_name="job_${job_name}.sh"

    # Write the SLURM script
    cat <<EOT > $script_name
#!/bin/bash
#SBATCH --job-name=$job_name   # Job name
#SBATCH --partition=gpu-volta  # GPU partition
#SBATCH --gres=gpu:1           # Request one GPU
#SBATCH --time=02:00:00        # Time limit (hh:mm:ss)
#SBATCH --ntasks=1             # Number of tasks
#SBATCH --cpus-per-task=4      # Number of CPU cores per task
#SBATCH --mem=32G              # Memory per node
#SBATCH --output=${job_name}_%j.log  # Standard output and error log

# Load necessary modules
module load cuda/11.8  # Adjust CUDA version if necessary

# Activate Conda environment
source /opt/ohpc/pub/compiler/anaconda3/2023.09-0/bin/activate
conda activate cs468

cd /home/research/tiffan/repo/foundation-model-schlieren
export PYTHONPATH=$PYTHONPATH:/home/research/tiffan/repo/foundation-model-schlieren

# Run the Python script
python src/models/autoencoders/fm_features/main.py \
  --learning_rate $lr \
  --weight_decay 1e-4
  --data_path data/features_16/exp_frames_672_432/dino_features_16.npy \
  --dataset_keyword dino_features_16 \
  --model_keyword unet_ae_5blk \
  --latent_dim $ld \
  --epochs 2000 \
  --batch_size 64 \
  --conv_channels 8,8,8,16,16 \
  --random_seed 42 \
  --save_checkpoint_every 100
EOT

    # Submit the job
    sbatch $script_name
  done
done
