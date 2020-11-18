#!/bin/bash
#
#SBATCH --job-name=C-N_yield
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate torch-env
#python src/run_trainer_yield.py --data_files data/C_N_yield/MFF_FullCV_* --output_dir models/C_N_Yield/scratch/ --split_idx 2372 2767 --eval --vocab data/vocab/2020_08_09_vocab.pt --num_epoch 100
#python src/run_trainer_yield.py --data_files data/C_N_yield/MFF_Test1.txt --output_dir models/C_N_Yield/scratch/ --split_idx 3057 --vocab data/vocab/2020_08_09_vocab.pt
#python src/run_trainer_yield.py --data_files data/C_N_yield/MFF_Test[24].txt --output_dir models/C_N_Yield/scratch/ --split_idx 3055 --vocab data/vocab/2020_08_09_vocab.pt
#python src/run_trainer_yield.py --data_files data/C_N_yield/MFF_Test3.txt --output_dir models/C_N_Yield/scratch/ --split_idx 3058 --vocab data/vocab/2020_08_09_vocab.pt
#python src/run_pred_yield.py --data_files data/C_N_yield/MFF_FullCV_* --model_dir models/C_N_Yield/scratch/ --split_idx 2767 --load_best
#python src/run_pred_yield.py --data_files data/C_N_yield/MFF_Test1.txt --model_dir models/C_N_Yield/scratch/ --split_idx 3057
#python src/run_pred_yield.py --data_files data/C_N_yield/MFF_Test[24].txt --model_dir models/C_N_Yield/scratch/ --split_idx 3055
#python src/run_pred_yield.py --data_files data/C_N_yield/MFF_Test3.txt --model_dir models/C_N_Yield/scratch/ --split_idx 3058

#python src/run_trainer_yield.py --data_files data/C_N_yield/MFF_FullCV_* --output_dir models/C_N_Yield/pretrain_rxn/ --pretrain models/smiles_pretrain/rxn_pretrain/ --eval --split_idx 2372 2767 --vocab data/vocab/2020_08_09_vocab.pt --num_epoch 100 --mode sigmoid2
#python src/run_trainer_yield.py --data_files data/C_N_yield/MFF_Test1.txt --output_dir models/C_N_Yield/pretrain_rxn/ --pretrain models/smiles_pretrain/rxn_pretrain/ --split_idx 3057 --vocab data/vocab/2020_08_09_vocab.pt --mode sigmoid2
#python src/run_trainer_yield.py --data_files data/C_N_yield/MFF_Test[24].txt --output_dir models/C_N_Yield/pretrain_rxn/ --pretrain models/smiles_pretrain/rxn_pretrain/ --split_idx 3055 --vocab data/vocab/2020_08_09_vocab.pt --mode sigmoid2
#python src/run_trainer_yield.py --data_files data/C_N_yield/MFF_Test3.txt --output_dir models/C_N_Yield/pretrain_rxn/ --pretrain models/smiles_pretrain/rxn_pretrain/ --split_idx 3058 --vocab data/vocab/2020_08_09_vocab.pt --mode sigmoid2
#python src/run_pred_yield.py --data_files data/C_N_yield/MFF_FullCV_* --model_dir models/C_N_Yield/pretrain_rxn --split_idx 2767 --load_best --mode sigmoid2
#python src/run_pred_yield.py --data_files data/C_N_yield/MFF_Test1.txt --model_dir models/C_N_Yield/pretrain_rxn --split_idx 3057 --mode sigmoid2
#python src/run_pred_yield.py --data_files data/C_N_yield/MFF_Test[24].txt --model_dir models/C_N_Yield/pretrain_rxn --split_idx 3055 --mode sigmoid2
#python src/run_pred_yield.py --data_files data/C_N_yield/MFF_Test3.txt --model_dir models/C_N_Yield/pretrain_rxn --split_idx 3058 --mode sigmoid2

#python src/run_trainer_yield.py --data_files data/C_N_yield/out_sample/train.txt --output_dir models/C_N_Yield/pretrain_rxn/ --pretrain models/smiles_pretrain/rxn_pretrain/ --split_idx 0 --vocab data/vocab/2020_08_09_vocab.pt
#python src/run_pred_yield.py --data_files data/C_N_yield/out_sample/test* --model_dir models/C_N_Yield/pretrain_rxn --split_idx 0

#python src/run_trainer_yield.py --data_files data/USPTO_yield/Ni_rxns/train.txt --output_dir models/USPTO_yield/Ni_rxns/ --pretrain models/smiles_pretrain/rxn_pretrain/ --eval --split_idx 0 --val_file data/USPTO_yield/Ni_rxns/val.txt --vocab data/vocab/2020_08_09_vocab.pt --batch_size 4 --save_total_limit 0
python src/run_pred_yield.py --data_files data/USPTO_yield/Ni_rxns/train.txt --model_dir models/USPTO_yield/Ni_rxns/ --split_idx 1
python src/run_pred_yield.py --data_files data/USPTO_yield/Ni_rxns/val.txt --model_dir models/USPTO_yield/Ni_rxns/ --split_idx 1
python src/run_pred_yield.py --data_files data/USPTO_yield/Ni_rxns/test.txt --model_dir models/USPTO_yield/Ni_rxns/ --split_idx 1

