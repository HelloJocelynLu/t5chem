#!/bin/bash
#
#SBATCH --job-name=STEREO
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate torch-env
#python run_predict.py --data_dir STEREO_separated/ --model_dir STEREO_new_scratch/ --prediction STEREO_new_scratch/test_pred_beam10.txt --num_beams 10 > rxn_pred.out
#python src/run_trainer.py --data_dir data/STEREO_separated/ --output_dir models/STEREO_sep/ --pretrain models/smiles_pretrain/rxn_pretrain/ --learning_rate 1e-4
python src/run_predict.py --data_dir data/STEREO_separated --vocab models/STEREO_sep/vocab.pt --model_dir models/STEREO_sep/best_cp-445000 --prediction models/STEREO_sep/prediction_test.txt --num_beams 10 --batch_size 16
