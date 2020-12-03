#!/bin/bash
#
#SBATCH --job-name=CN_T5
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate torch-env
#python src/run_pretrain.py --data_dir data/pubchem/train_0/ --output_dir models/smiles_pretrain/new_vocab_0 --pretrain models/smiles_pretrain/train_9/ --vocab data/vocab/mol_all_20201109.pt --save_steps 200000 --save_total_limit 2
#python src/run_trainer.py --data_dir data/USPTO_yield/ --output_dir models/USPTO_yield/prefix/ --vocab data/vocab/mol_all_20201109.pt --task_prefix Yield: --max_target_length 5 --pretrain models/smiles_pretrain/train_9/
#python src/run_predict.py --data_dir data/USPTO_yield/ --model_dir models/USPTO_yield/prefix/ --task_prefix Yield: --max_target_length 5 --num_preds 1
python src/run_trainer.py --data_dir data/C_N_yield/MFF_FullCV_01/ --output_dir models/temp_test/ --vocab data/vocab/mol_all_20201109.pt --task_prefix Yield: --max_target_length 5 --num_epoch 200 --pretrain models/smiles_pretrain/train_9/ --save_steps 3000 --save_total_limit 10
#python src/run_predict.py --data_dir data/C_N_yield/MFF_FullCV_01/ --model_dir models/temp_test/checkpoint-3000/ --task_prefix Yield: --max_target_length 5 --num_preds 1
#python src/evaluate_predictions.py --prediction models/temp_test/checkpoint-3000/predictions.csv --type value
#python src/run_predict.py --data_dir data/C_N_yield/MFF_FullCV_01/ --model_dir models/temp_test/checkpoint-6000/ --task_prefix Yield: --max_target_length 5 --num_preds 1
#python src/evaluate_predictions.py --prediction models/temp_test/checkpoint-6000/predictions.csv --type value
#python src/run_predict.py --data_dir data/C_N_yield/MFF_FullCV_01/ --model_dir models/temp_test/checkpoint-9000/ --task_prefix Yield: --max_target_length 5 --num_preds 1
#python src/evaluate_predictions.py --prediction models/temp_test/checkpoint-9000/predictions.csv --type value
#python src/run_predict.py --data_dir data/C_N_yield/MFF_FullCV_01/ --model_dir models/temp_test/checkpoint-12000/ --task_prefix Yield: --max_target_length 5 --num_preds 1
#python src/evaluate_predictions.py --prediction models/temp_test/checkpoint-12000/predictions.csv --type value
#python src/run_predict.py --data_dir data/C_N_yield/MFF_FullCV_01/ --model_dir models/temp_test/checkpoint-15000/ --task_prefix Yield: --max_target_length 5 --num_preds 1
#python src/evaluate_predictions.py --prediction models/temp_test/checkpoint-15000/predictions.csv --type value
#python src/run_trainer.py --data_dir data/C_N_yield/MFF_FullCV_02/ --output_dir models/temp_test/ --vocab data/vocab/mol_all_20201109.pt --task_prefix Yield: --max_target_length 5 --num_epoch 100 --pretrain models/smiles_pretrain/train_9/
#python src/run_predict.py --data_dir data/C_N_yield/MFF_FullCV_02/ --model_dir models/temp_test/ --task_prefix Yield: --max_target_length 5 --num_preds 1
#python src/evaluate_predictions.py --prediction models/temp_test/predictions.csv --type value
#python src/run_trainer.py --data_dir data/C_N_yield/MFF_FullCV_03/ --output_dir models/temp_test/ --vocab data/vocab/mol_all_20201109.pt --task_prefix Yield: --max_target_length 5 --num_epoch 100 --pretrain models/smiles_pretrain/train_9/
#python src/run_predict.py --data_dir data/C_N_yield/MFF_FullCV_03/ --model_dir models/temp_test/ --task_prefix Yield: --max_target_length 5 --num_preds 1
#python src/evaluate_predictions.py --prediction models/temp_test/predictions.csv --type value
#python src/run_trainer.py --data_dir data/C_N_yield/MFF_FullCV_04/ --output_dir models/temp_test/ --vocab data/vocab/mol_all_20201109.pt --task_prefix Yield: --max_target_length 5 --num_epoch 100 --pretrain models/smiles_pretrain/train_9/
#python src/run_predict.py --data_dir data/C_N_yield/MFF_FullCV_04/ --model_dir models/temp_test/ --task_prefix Yield: --max_target_length 5 --num_preds 1
#python src/evaluate_predictions.py --prediction models/temp_test/predictions.csv --type value
#python src/run_trainer.py --data_dir data/C_N_yield/MFF_FullCV_05/ --output_dir models/temp_test/ --vocab data/vocab/mol_all_20201109.pt --task_prefix Yield: --max_target_length 5 --num_epoch 100 --pretrain models/smiles_pretrain/train_9/
#python src/run_predict.py --data_dir data/C_N_yield/MFF_FullCV_05/ --model_dir models/temp_test/ --task_prefix Yield: --max_target_length 5 --num_preds 1
