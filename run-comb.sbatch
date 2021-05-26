#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=168:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:rtx8000:1

#GREENE GREENE_GPU_MPS=YES

module purge

#script=$1
overlay_ext3=/home/jl8570/torch-env.ext3
if [[ $(hostname -s) =~ ^g ]]; then nv="--nv"; fi

singularity \
    exec $nv --overlay $overlay_ext3:ro \
    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif  \
    /bin/bash -c "
source /ext3/env.sh; 
python src/run_trainer.py \
    --data_dir data/USPTO_1k_TPL/Product/ \
    --output_dir models/USPTO_TPL/Product/ \
    --vocab data/vocab/mol_all_simpletokenizer.pt \
    --task_prefix Product: --num_epoch 100 --vocab_size 100 \
    --learning_rate 5e-4 --max_target_length 200 --log_steps 1500 \
    --max_source_length 400 --tokenizer simple --batch_size 32
python src/run_predict.py \
       --data_dir data/USPTO_1k_TPL/Product/ \
       --model_dir models/USPTO_TPL/Product/ \
       --num_beams 10 --batch_size 64 --max_target_length 200 \
       --task_prefix Product: --vocab_size 100 --tokenizer simple --max_source_length 400
python src/evaluate_predictions.py --prediction models/USPTO_TPL/Product/predictions.csv --type text
"  > prod_2.log 2>&1 &

sleep 60

singularity \
    exec $nv --overlay $overlay_ext3:ro \
    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif  \
    /bin/bash -c "
source /ext3/env.sh;
python src/run_trainer.py \
    --data_dir data/USPTO_1k_TPL/Retro/ \
    --output_dir models/USPTO_TPL/Retro/ \
    --vocab data/vocab/mol_all_simpletokenizer.pt \
    --task_prefix Retrosynthesis: --num_epoch 100 --vocab_size 100 \
    --learning_rate 5e-4 --max_target_length 400 --log_steps 1500 \
    --max_source_length 200 --tokenizer simple --batch_size 32
python src/run_predict.py \
       --data_dir data/USPTO_1k_TPL/Retro/ \
       --model_dir models/USPTO_TPL/Retro/ \
       --num_beams 10 --batch_size 64 --max_target_length 400 \
       --task_prefix Retrosynthesis: --vocab_size 100 --tokenizer simple --max_source_length 200
python src/evaluate_predictions.py --prediction models/USPTO_TPL/Multitask/Retro/predictions.csv --type text
"  > retro_2.log 2>&1 &

wait



