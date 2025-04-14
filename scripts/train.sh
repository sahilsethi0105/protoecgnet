#!/bin/bash -l

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=40gb
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpuq
#SBATCH --time=00-23:59:59
#SBATCH --output=job_default.out

# Function to parse named arguments
parse_args() {
  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --job_name) job_name="$2"; shift ;;
      --epochs) epochs="$2"; shift ;;
      --batch_size) batch_size="$2"; shift ;;
      --lr) lr="$2"; shift ;;
      --l2) l2="$2"; shift ;;
      --dropout) dropout="$2"; shift ;;
      --checkpoint_dir) checkpoint_dir="$2"; shift ;;
      --log_dir) log_dir="$2"; shift ;;
      --save_top_k) save_top_k="$2"; shift ;;
      --patience) patience="$2"; shift ;;
      --resume_checkpoint) resume_checkpoint="$2"; shift ;;
      --pretrained_weights) pretrained_weights="$2"; shift ;;
      --training_stage) training_stage="$2"; shift ;;
      --dimension) dimension="$2"; shift ;;
      --backbone) backbone="$2"; shift ;;
      --single_class_prototype_per_class) single_class_prototype_per_class="$2"; shift ;;
      --joint_prototypes_per_border) joint_prototypes_per_border="$2"; shift ;;
      --sampling_rate) sampling_rate="$2"; shift ;;
      --label_set) label_set="$2"; shift ;;
      --test_model) test_model="$2"; shift ;;
      --save_weights) save_weights="$2"; shift ;;
      --seed) seed="$2"; shift ;;
      --scheduler_type) scheduler_type="$2"; shift ;;
      --num_workers) num_workers="$2"; shift ;;
      --custom_groups) custom_groups="$2"; shift ;;
      --proto_time_len) proto_time_len="$2"; shift ;;
      --proto_dim) proto_dim="$2"; shift ;;
      *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
  done
}

export TORCH_HOME=/gpfs/data/bbj-lab/users/sethis/torch_cache

# debugging flags
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# Parse the arguments
parse_args "$@"

cd /gpfs/data/bbj-lab/users/sethis/bbj_ecg

module purge
module load gcc/12.1.0
module load miniconda3/24.9.2
module list 2>&1

conda activate /gpfs/data/bbj-lab/users/sethis/ecg_env

#debugging
# which python  
# python --version  
# conda list 

# Run the Python script with parsed arguments
cd /gpfs/data/bbj-lab/users/sethis/bbj_ecg/src

CMD="srun --output=\"/gpfs/data/bbj-lab/users/sethis/bbj_ecg/${job_name}.txt\" --error=\"/gpfs/data/bbj-lab/users/sethis/bbj_ecg/${job_name}.err\" python3 main.py \
  --job_name \"$job_name\" \
  --epochs \"$epochs\" \
  --batch_size \"$batch_size\" \
  --lr \"$lr\" \
  --l2 \"$l2\" \
  --dropout \"$dropout\" \
  --checkpoint_dir \"$checkpoint_dir\" \
  --log_dir \"$log_dir\" \
  --save_top_k \"$save_top_k\" \
  --patience \"$patience\" \
  --training_stage \"$training_stage\" \
  --dimension \"$dimension\" \
  --backbone \"$backbone\" \
  --single_class_prototype_per_class \"$single_class_prototype_per_class\" \
  --joint_prototypes_per_border \"$joint_prototypes_per_border\" \
  --sampling_rate \"$sampling_rate\" \
  --label_set \"$label_set\" \
  --save_weights \"$save_weights\" \
  --scheduler_type \"$scheduler_type\" \
  --seed \"$seed\" \
  --num_workers \"$num_workers\""

# Only add pretrained_weights if it was provided
if [[ -n "$pretrained_weights" ]]; then
  CMD+=" --pretrained_weights \"$pretrained_weights\""
fi
if [[ -n "$custom_groups" ]]; then
  CMD+=" --custom_groups \"$custom_groups\""
fi
if [[ -n "$proto_time_len" ]]; then
  CMD+=" --proto_time_len \"$proto_time_len\""
fi
if [[ -n "$proto_dim" ]]; then
  CMD+=" --proto_dim \"$proto_dim\""
fi

# Run the command
eval $CMD > "/gpfs/data/bbj-lab/users/sethis/bbj_ecg/${job_name}_results.txt"

conda deactivate


#Command line test: 
#sbatch train.sh --job_name "train_resnet1d" --epochs 5 --batch_size 32        --lr 1e-3 --checkpoint_dir "/gpfs/data/bbj-lab/users/sethis/experiments/checkpoints"        --log_dir "/gpfs/data/bbj-lab/users/sethis/experiments/logs" --save_top_k 3        --patience 5 --resume_checkpoint True --training_stage "feature_extractor"        --dimension "1D" --backbone "resnet1d18" --single_class_prototype_per_class 10 --joint_prototypes_per_border 5        --sampling_rate 100 --label_set "superdiagnostic" --save_weights True --seed 42 --num_workers 0
