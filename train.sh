#!/bin/bash
#SBATCH --job-name=train              # Job name
##SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
##SBATCH --mail-user=email@ufl.edu     # Where to send mail	
#SBATCH --ntasks=4                    # Run on a single CPU
#SBATCH --mem-per-cpu=4000            # Job memory request
#SBATCH --gpus=gtx_1080_ti:1          # Job memory request
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --output=train_%j.log         # Standard output and error log


module load gcc/8.2.0 cuda/11.6.2 python/3.8.5 cudnn/8.0.5 cmake/3.19.8 eth_proxy
source ~/mp23_env/bin/activate

python train.py --config config.yaml

echo "Training started!"