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
       --data_dir data/USPTO_1k_TPL/bidirectional/ \
       --output_dir models/USPTO_TPL/bidirectional/ \
       --vocab data/vocab/mol_all_simpletokenizer.pt \
       --task_prefix '' --num_epoch 100 --vocab_size 100 \
	   --learning_rate 1e-4 --max_target_length 400 --log_steps 5000 \
       --max_source_length 400 --tokenizer simple
python src/run_predict.py \
       --data_dir data/USPTO_1k_TPL/Product/ \
       --model_dir models/USPTO_TPL/bidirectional/ \
       --num_beams 10 --batch_size 64 --max_target_length 200 \
       --task_prefix Product: --vocab_size 100 --tokenizer simple --max_source_length 400
python src/evaluate_predictions.py --prediction models/USPTO_TPL/bidirectional/predictions.csv --type text
python src/run_predict.py \
       --data_dir data/USPTO_1k_TPL/Retro/ \
       --model_dir models/USPTO_TPL/bidirectional/ \
       --num_beams 10 --batch_size 64 --max_target_length 400 \
       --task_prefix Retrosynthesis: --vocab_size 100 --tokenizer simple --max_source_length 200
python src/evaluate_predictions.py --prediction models/USPTO_TPL/bidirectional/predictions.csv --type text
"  > 1_1.log 2>&1 &

sleep 60

singularity \
    exec $nv --overlay $overlay_ext3:ro \
    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif  \
    /bin/bash -c "
source /ext3/env.sh;
python src/run_trainer.py \
       --data_dir data/USPTO_1k_TPL/bidirectional/ \
       --output_dir models/USPTO_TPL/Multitask/bidirectional/ \
       --pretrain models/USPTO_TPL/Classification/ \
       --vocab data/vocab/mol_all_simpletokenizer.pt \
       --task_prefix '' --num_epoch 100 --vocab_size 100 \
       --learning_rate 1e-4 --max_target_length 400 --log_steps 5000 \
       --max_source_length 400 --tokenizer simple
python src/run_predict.py \
       --data_dir data/USPTO_1k_TPL/Product/ \
       --model_dir models/USPTO_TPL/Multitask/bidirectional/ \
       --num_beams 10 --batch_size 64 --max_target_length 200 \
       --task_prefix Product: --vocab_size 100 --tokenizer simple --max_source_length 400
python src/evaluate_predictions.py --prediction models/USPTO_TPL/Multitask/bidirectional/predictions.csv --type text
python src/run_predict.py \
       --data_dir data/USPTO_1k_TPL/Retro/ \
       --model_dir models/USPTO_TPL/Multitask/bidirectional/ \
       --num_beams 10 --batch_size 64 --max_target_length 400 \
       --task_prefix Retrosynthesis: --vocab_size 100 --tokenizer simple --max_source_length 200
python src/evaluate_predictions.py --prediction models/USPTO_TPL/Multitask/bidirectional/predictions.csv --type text
"  > 2_1.log 2>&1 &

wait



