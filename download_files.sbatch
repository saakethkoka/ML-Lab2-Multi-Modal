#!/bin/bash
#SBATCH -J download        # Job name
#SBATCH -o download.out # Output file name
#SBATCH --time=4:00:00                 # Time limit
#SBATCH -n 10                      # Number of cores
#SBATCH --mem=64G
#SBATCH -G 1


# Load TensorFlow env
# module spider cuda-11.4.4-gcc-10.3.0-ctldo35
module purge
module load dev/1
module load cuda-11.4.4-gcc-10.3.0-ctldo35 cudnn-8.2.4.15-11.4-gcc-10.3.0-eluwegp
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
echo $CUDA_HOME
source ~/.venv/tensorflow/bin/activate

# Run Python script
python3 downloading_files.py