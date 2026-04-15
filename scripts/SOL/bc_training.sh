#!/bin/bash
#SBATCH -G a100:1
#SBATCH -p general
#SBATCH -q class
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 0-03:30:00
#SBATCH --mem=20G
#SBATCH -o outputs/out/pedestrian_RL/bc_training_%j.out
#SBATCH -e outputs/err/pedestrian_RL/bc_training_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ctsai67@asu.edu

set -e

# Clone the repo first
# REPO_URL="https://github.com/andytsai104/pedestrian_RL.git"
# REPO_DIR="pedestrian_RL"
# BRANCH="main"

# if [ ! -d "$REPO_DIR/.git" ]; then
#     echo "Repo not found. Cloning..."
#     git clone -b "$BRANCH" "$REPO_URL"
# else
#     echo "Repo exists. Force updating..."
#     cd "$REPO_DIR"
#     git fetch origin
#     git reset --hard "origin/$BRANCH"
#     git clean -fd
# fi

# Initialize conda
source activate /scratch/ctsai67/envs/carla_env

# Go to project
cd ~/pedestrian_RL

# Debug info
echo "Python: $(which python)"
python --version
python -c "import sys; print(sys.executable)"

# Run training
python -m pedestrian_rl.training.bc_training
