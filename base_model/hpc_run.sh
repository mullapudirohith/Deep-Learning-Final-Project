#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx8000
#SBATCH --output=DL_model1.out
#SBATCH --mail-type=END
#SBATCH --mail-user=rcm8445@nyu.edu
#SBATCH --job-name=DL_model1

module purge

singularity exec --nv \
	    --overlay /scratch/rcm8445/pytorch-example/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; 
	    
	    pip install -r ../requirements.txt
	    pip install torch torchvision torchaudio
	    python3 base_model.py"