#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=benjamin.krala@hhi.fraunhofer.de
#SBATCH --output=test.out

echo "starte"
bash ./run.sh            # <--- main call exiting with return code 5 for FAILED
echo "beende"