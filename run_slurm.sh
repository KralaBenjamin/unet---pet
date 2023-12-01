#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=benjamin.krala@hhi.fraunhofer.de
#SBATCH --output=test.out
#SBATCH --gpus=2
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

bash ./run.sh            # <--- main call exiting with return code 5 for FAILED
