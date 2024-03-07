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

pips install --user pandas
pip3 install --user nltk


python3 scripts/preprocess_text.py
