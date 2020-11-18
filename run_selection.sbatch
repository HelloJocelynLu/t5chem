#!/bin/bash
#
#SBATCH --job-name=selection
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate torch-env
python src/rxn_selector.py --data_file data/Ni_rxn/out_samples/train.txt --test_files data/Ni_rxn/out_samples/test* --output_dir models/Ni_rxn/rxn_pretrain/ --pretrain models/smiles_pretrain/rxn_pretrain/ --vocab data/vocab/2020_08_09_vocab.pt --mode linear2 >> models/Ni_rxn/selection/Ni-selection.log
#python src/rxn_selector.py --data_file data/Ni_rxn/out_samples/train.txt --test_files data/Ni_rxn/out_samples/test* --output_dir models/Ni_rxn/rxn_pretrain/ --pretrain models/smiles_pretrain/rxn_pretrain/ --vocab data/vocab/2020_08_09_vocab.pt > models/Ni_rxn/selection/Ni-selection.log
#python src/rxn_selector.py --data_file data/Ni_rxn/out_samples/train.txt --test_files data/Ni_rxn/out_samples/test* --output_dir models/Ni_rxn/scratch/ --vocab data/vocab/2020_08_09_vocab.pt >> models/Ni_rxn/selection/Ni-selection.log
#python src/rxn_selector.py --data_file data/Ni_rxn/out_samples/train.txt --test_files data/Ni_rxn/out_samples/test* --output_dir models/Ni_rxn/USPTO_pretrain/ --pretrain models/USPTO_yield/Ni_rxns/sigmoid/train/ --copy_all_w --vocab data/vocab/2020_08_09_vocab.pt >> models/Ni_rxn/selection/Ni-selection.log
#python run_selection_yield.py --data_files C_N_yield/MFF_FullCV_01.txt --output_dir CNYield/selection --pretrain USPTO_yield_model/sigmoid/test/ --copy_all_w --vocab 2020_08_09_vocab.pt --num_epoch 2500 >> selections.log
#python iterative_selector.py --data_file NI_cat/out_samples/train.txt --test_file NI_cat/out_samples/test3.txt --output_dir Ni_rxn/selection/iterative --pretrain smiles_pretrain/rxn_pretrain/ --vocab 2020_08_09_vocab.pt > iterative-selection.log
#python iterative_selector.py --data_file C_N_yield/out_sample/train.txt --test_file C_N_yield/out_sample/test4.txt --output_dir CNYield/selection/iterative --pretrain smiles_pretrain/rxn_pretrain/ --vocab 2020_08_09_vocab.pt >> iterative-selection.log
