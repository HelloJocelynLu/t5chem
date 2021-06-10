#!/bin/bash

#python src/run_pretrain.py --data_dir data/USPTO_50k_X19/ --output_dir models/USPTO_50k/multitask_pretrain --pretrain models/smiles_pretrain/rxn_pretrain/ --vocab data/vocab/USPTO_moltokenizer.pt --save_steps 200000 --save_total_limit 1 --num_epoch 100 --vocab_size 500
#python src/run_trainer.py --data_dir data/USPTO_50k_20M/ --output_dir models/USPTO_50k_20M/ --vocab data/vocab/mol_all_simpletokenizer.pt --task_prefix '' --num_epoch 50 --vocab_size 100 --learning_rate 1e-3 --max_target_length 250 --log_steps 3000 --max_source_length 200 --tokenizer simple --batch_size 32 --pretrain models/USPTO_50k_20M/checkpoint-900000/ --continue
#python src/run_trainer.py --data_dir data/USPTO_1k_TPL/Product/ --output_dir models/USPTO_TPL/Product/ --vocab data/vocab/mol_all_simpletokenizer.pt --task_prefix Product: --num_epoch 500 --vocab_size 100 --learning_rate 5e-4 --max_target_length 200 --log_steps 1500 --max_source_length 400 --tokenizer simple --batch_size 64
#python src/run_trainer.py --data_dir data/MIT_mixed/ --output_dir models/MIT_mixed/ --vocab data/vocab/mol_all_simpletokenizer.pt --task_prefix Product: --num_epoch 100 --vocab_size 100 --learning_rate 5e-4 --pretrain models/simple_smiles2/train_9/ --max_target_length 200 --log_steps 1500 --max_source_length 400 --tokenizer simple --batch_size 32
##python src/run_trainer.py --data_dir data/USPTO_1k_TPL_X9/bidirectional/ --output_dir models/USPTO_TPL/bidirectional_augm/ --vocab data/vocab/mol_all_simpletokenizer.pt --task_prefix '' --num_epoch 50 --vocab_size 100 --learning_rate 5e-4 --pretrain models/USPTO_TPL/bidirectional_augm/checkpoint-3400000/ --max_target_length 300 --log_steps 5000 --max_source_length 400 --tokenizer simple --batch_size 64 --continue
#python src/run_trainer.py --data_dir data/STEREO_separated/ --output_dir models/STEREO_sep/simple/ --vocab data/vocab/mol_all_simpletokenizer.pt --task_prefix Product: --num_epoch 30 --vocab_size 100 --learning_rate 5e-4 --pretrain models/simple_rxn/ --max_target_length 200 --log_steps 1500 --tokenizer simple --max_source_length 400
#python src/run_trainer.py --data_dir data/STEREO_separated/ --vocab data/vocab/selfies_900.pt --vocab_size 1000 --output_dir models/STEREO_sep/selfies --learning_rate 5e-4 --task_prefix Product: --num_epoch 30 --tokenizer selfies --pretrain models/selfies_pretrain/train_9/
#cp models/USPTO_50k/can/vocab.pt models/USPTO_50k/can/best_cp-32500/
python src/run_predict.py --data_dir data/USPTO_50k/ --model_dir models/USPTO_50k_20M/ --num_beams 10 --batch_size 32 --max_target_length 250 --task_prefix Retrosynthesis: --vocab_size 100 --tokenizer simple --max_source_length 200
python src/evaluate_predictions.py --prediction models/USPTO_50k_20M/predictions.csv --type text
##python src/run_predict.py --data_dir data/USPTO_1k_TPL/Product/ --model_dir models/USPTO_TPL/bidirectional_augm/ --num_beams 10 --batch_size 64 --max_target_length 200 --task_prefix Product: --vocab_size 100 --tokenizer simple --max_source_length 400
##python src/evaluate_predictions.py --prediction models/USPTO_TPL/bidirectional_augm/predictions.csv --type text

#python src/run_predict.py --data_dir data/USPTO_1k_TPL/Classification/ --model_dir models/USPTO_TPL/triple/ --num_beams 10 --batch_size 64 --max_target_length 5 --task_prefix 'Classification:' --vocab_size 100 --tokenizer simple --max_source_length 500
#python src/evaluate_predictions.py --prediction models/USPTO_TPL/bidirectional/predictions.csv --type text

##python src/run_predict.py --data_dir data/USPTO_1k_TPL/Retro/ --model_dir models/USPTO_TPL/bidirectional_augm/ --num_beams 10 --batch_size 64 --max_target_length 400 --task_prefix Retrosynthesis: --vocab_size 100 --tokenizer simple --max_source_length 200
##python src/evaluate_predictions.py --prediction models/USPTO_TPL/bidirectional_augm/predictions.csv --type text
#python src/run_predict.py --data_dir data/USPTO_1k_TPL/Retro/ --model_dir models/USPTO_TPL/triple/ --num_beams 10 --batch_size 64 --max_target_length 300 --task_prefix 'Retrosynthesis:' --vocab_size 100 --tokenizer simple --max_source_length 200
#python src/run_predict.py --data_dir data/MIT_mixed/ --model_dir models/MIT_mixed/ --num_beams 10 --batch_size 64 --max_target_length 200 --task_prefix Product: --vocab_size 100 --tokenizer simple --max_source_length 400
#python src/run_predict.py --data_dir data/USPTO_50k_X19/ --model_dir models/USPTO_50k_X19/ --num_beams 10 --max_target_length 250 --task_prefix Retrosynthesis: --vocab_size 100 --tokenizer simple --num_preds 10
#python src/run_trainer.py --data_dir data/ResLig/ --output_dir models/ResLig --pretrain models/STEREO_sep/prefix/ --vocab data/vocab/mol_all_20201109.pt --task_prefix Fill-Mask: --num_epoch 100
#python src/run_predict.py --data_dir data/ResLig --model_dir models/ResLig --prediction models/ResLig/predictions.csv --num_beams 10 --batch_size 32
#python src/evaluate_predictions.py --prediction models/USPTO_TPL/triple/predictions.csv --type text
#python src/evaluate_predictions.py --prediction models/USPTO_TPL/bidirectional/predictions.csv --type text
#python src/evaluate_predictions.py --prediction models/MIT_mixed/predictions.csv --type text
#python src/evaluate_score_rank.py --prediction models/USPTO_50k/
