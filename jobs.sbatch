#!/bin/bash
#SBATCH -J captions2        # Job name
#SBATCH -o captions2.out    # Output file name
#SBATCH --time=4:00:00     # Time limit
#SBATCH -c 10              # Number of cores
#SBATCH --mem=64G          # Memory limit
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
python3 flickr.py