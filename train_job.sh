#!/bin/bash
#SBATCH --job-name=ijustworkhere
#SBATCH --account=eecs448w24_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --mail-type=END

module purge

pip3 install --user torch
pip3 install --user sklearn
pips install --user pandas
pip3 install --user setuptools-rust
pip3 install --user transformers
pip3 install --user tqdm


python3 util/create_and_train.py
