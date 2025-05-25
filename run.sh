#!/bin/bash

#SBATCH --job-name=debias_bow_tie_explaining_t

#SBATCH --partition=cpu-epyc-genoa

#SBATCH --nodelist=cpu16,cpu17,cpu18,cpu19,cpu20,cpu21,cpu22,cpu23

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=128

#SBATCH --mem=400GB

#SBATCH --output=Logs/debias_bow_tie_explaining_teacher_%j.out

#SBATCH --error=Logs/debias_bow_tie_explaining_teacher_%j.err

#SBATCH --qos=long

#SBATCH --time=24:00:00

module load miniconda

source /app/miniconda/24.1.2/etc/profile.d/conda.sh

conda activate TESTWAND

java --add-modules jdk.incubator.vector -Xmx4g -Xms2g -XX:+UnlockExperimentalVMOptions -XX:+EnableVectorSupport -jar target/MatricesHarness-WIF3011-1.0-SNAPSHOT.jar
