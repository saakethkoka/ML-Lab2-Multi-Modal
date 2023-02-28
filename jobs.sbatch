#!/bin/bash
#SBATCH -J captions        # Job name
#SBATCH -o captions.out # Output file name
#SBATCH --time=4:00:00                 # Time limit
#SBATCH --exclusive
#SBATCH -G 8


# Load TensorFlow env
# module spider cuda-11.4.4-gcc-10.3.0-ctldo35
module purge
module load dev/1
module load cuda-11.4.4-gcc-10.3.0-ctldo35 cudnn-8.2.4.15-11.4-gcc-10.3.0-eluwegp
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
echo $CUDA_HOME
source ~/.venv/tensorflow/bin/activate

# Run Python script
python3 main.py