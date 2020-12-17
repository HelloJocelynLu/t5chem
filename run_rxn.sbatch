#!/bin/bash
#
#SBATCH --job-name=Xiaokang
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate torch-env
#python run_predict.py --data_dir STEREO_separated/ --model_dir STEREO_new_scratch/ --prediction STEREO_new_scratch/test_pred_beam10.txt --num_beams 10 > rxn_pred.out
#python src/run_trainer.py --data_dir data/STEREO_separated/ --output_dir models/STEREO_sep/prefix/ --pretrain models/smiles_pretrain/train_9/ --learning_rate 5e-4 --vocab data/vocab/mol_all_20201109.pt --task_prefix Product:
#python src/run_predict.py --data_dir data/STEREO_separated --model_dir models/STEREO_sep/prefix/ --prediction models/STEREO_sep/prefix/prediction_test.txt --num_beams 10 --batch_size 32
python src/run_trainer.py --data_dir data/ResLig/ --output_dir models/ResLig --pretrain models/STEREO_sep/prefix/ --vocab data/vocab/mol_all_20201109.pt --task_prefix Fill-Mask: --num_epoch 100
python src/run_predict.py --data_dir data/ResLig --model_dir models/ResLig --prediction models/ResLig/predictions.csv --num_beams 10 --batch_size 32
python src/evaluate_predictions.py --prediction models/ResLig/predictions.csv --type text
#python src/evaluate_predictions.py --prediction models/STEREO_sep/prefix/prediction_test.txt --type text
