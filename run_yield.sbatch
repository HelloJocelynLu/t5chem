#python src/run_trainer.py --data_dir data/USPTO_yield_X2/ --output_dir models/USPTO_yield/one_two_dig/ --vocab data/vocab/mol_all_20201109.pt --task_prefix Yield: --max_target_length 2 --pretrain models/USPTO_yield/one_dig/ --num_epoch 15 --save_total_limit 10
#python src/run_predict.py --data_dir data/USPTO_yield/ --model_dir models/USPTO_yield/one_two_dig/ --task_prefix Yield: --max_target_length 2 --num_preds 5 --prediction models/USPTO_yield/one_two_dig/predictions.csv
#python src/evaluate_predictions.py --prediction models/USPTO_yield/one_two_dig/predictions.csv --type value --average 1
#python src/evaluate_predictions.py --prediction models/USPTO_yield/one_two_dig/predictions.csv --type value --average 2
#python src/evaluate_predictions.py --prediction models/USPTO_yield/one_two_dig/predictions.csv --type value --average 3
#python src/run_trainer.py --data_dir data/USPTO_yield/ --output_dir models/USPTO_yield/constant_lr/ --vocab data/vocab/mol_all_20201109.pt --task_prefix Yield: --max_target_length 2 --pretrain models/USPTO_yield/two_dig/ --num_epoch 15 --save_total_limit 1
#python src/run_predict.py --data_dir data/USPTO_yield/ --model_dir models/USPTO_yield/constant_lr/ --task_prefix Yield: --max_target_length 2 --num_preds 3
#python src/evaluate_predictions.py --prediction models/USPTO_yield//predictions.csv --type value
#python src/run_pretrain.py --data_dir data/USPTO_50k_X19/ --output_dir models/rxn_pretrain/ --pretrain models/smiles_pretrain/new_vocab_9/ --vocab data/vocab/mol_all_20201109.pt --save_steps 200000 --save_total_limit 1
python src/run_pretrain.py --data_dir data/pubchem/train_0/ --output_dir models/selfies_pretrain/train_0/ --vocab data/vocab/selfies_900.pt --save_steps 200000 --save_total_limit 1 --tokenizer selfies
python src/run_pretrain.py --data_dir data/pubchem/train_1/ --output_dir models/selfies_pretrain/train_1/ --pretrain models/selfies_pretrain/train_0/ --vocab data/vocab/selfies_900.pt --save_steps 200000 --save_total_limit 1 --tokenizer selfies
python src/run_pretrain.py --data_dir data/pubchem/train_2/ --output_dir models/selfies_pretrain/train_2/ --pretrain models/selfies_pretrain/train_1/ --vocab data/vocab/selfies_900.pt --save_steps 200000 --save_total_limit 1 --tokenizer selfies
python src/run_pretrain.py --data_dir data/pubchem/train_3/ --output_dir models/selfies_pretrain/train_3/ --pretrain models/selfies_pretrain/train_2/ --vocab data/vocab/selfies_900.pt --save_steps 200000 --save_total_limit 1 --tokenizer selfies
python src/run_pretrain.py --data_dir data/pubchem/train_4/ --output_dir models/selfies_pretrain/train_4/ --pretrain models/selfies_pretrain/train_3/ --vocab data/vocab/selfies_900.pt --save_steps 200000 --save_total_limit 1 --tokenizer selfies
python src/run_pretrain.py --data_dir data/pubchem/train_5/ --output_dir models/selfies_pretrain/train_5/ --pretrain models/selfies_pretrain/train_4/ --vocab data/vocab/selfies_900.pt --save_steps 200000 --save_total_limit 1 --tokenizer selfies
python src/run_pretrain.py --data_dir data/pubchem/train_6/ --output_dir models/selfies_pretrain/train_6/ --pretrain models/selfies_pretrain/train_5/ --vocab data/vocab/selfies_900.pt --save_steps 200000 --save_total_limit 1 --tokenizer selfies
python src/run_pretrain.py --data_dir data/pubchem/train_7/ --output_dir models/selfies_pretrain/train_7/ --pretrain models/selfies_pretrain/train_6/ --vocab data/vocab/selfies_900.pt --save_steps 200000 --save_total_limit 1 --tokenizer selfies
python src/run_pretrain.py --data_dir data/pubchem/train_8/ --output_dir models/selfies_pretrain/train_8/ --pretrain models/selfies_pretrain/train_7/ --vocab data/vocab/selfies_900.pt --save_steps 200000 --save_total_limit 1 --tokenizer selfies
python src/run_pretrain.py --data_dir data/pubchem/train_9/ --output_dir models/selfies_pretrain/train_9/ --pretrain models/selfies_pretrain/train_8/ --vocab data/vocab/selfies_900.pt --save_steps 200000 --save_total_limit 1 --tokenizer selfies
#python src/run_pretrain.py --data_dir data/pubchem/train_3_X4/ --output_dir models/smiles_pretrain/augm_3/ --pretrain models/smiles_pretrain/augm_2 --vocab data/vocab/mol_all_20201109.pt --save_steps 200000 --save_total_limit 1
#python src/run_pretrain.py --data_dir data/pubchem/train_4_X4/ --output_dir models/smiles_pretrain/augm_4/ --pretrain models/smiles_pretrain/augm_3 --vocab data/vocab/mol_all_20201109.pt --save_steps 200000 --save_total_limit 1
#python src/run_pretrain.py --data_dir data/pubchem/train_5_X4/ --output_dir models/smiles_pretrain/augm_5/ --pretrain models/smiles_pretrain/augm_4 --vocab data/vocab/mol_all_20201109.pt --save_steps 200000 --save_total_limit 1
#python src/run_pretrain.py --data_dir data/pubchem/train_6_X4/ --output_dir models/smiles_pretrain/augm_6/ --pretrain models/smiles_pretrain/augm_5 --vocab data/vocab/mol_all_20201109.pt --save_steps 200000 --save_total_limit 1
#for (( c=1; c<=4; c++ ))
#do
#    cp models/temp_test/vocab.pt models/temp_test/checkpoint-$c/
#    rm -rf models/temp_test2/checkpoint*
#    python src/run_trainer.py --data_dir data/C_N_yield/MFF_Test${c}_X4/ --output_dir models/temp_test2/ --vocab data/vocab/mol_all_20201109.pt --task_prefix Yield: --max_target_length 2 --num_epoch 100 --pretrain models/smiles_pretrain/new_vocab_9/ --save_steps 3000 --save_total_limit 1
#python src/run_predict.py --data_dir data/C_N_yield/MFF_Test4/ --model_dir models/temp_test/ --task_prefix Yield: --max_target_length 2 --num_preds 5 --prediction models/temp_test/predictions.csv
#    python src/evaluate_predictions.py --prediction models/temp_test2/predictions.csv --type value
#done
#rm -rf models/temp_test2/checkpoint*
#python src/run_trainer.py --data_dir data/C_N_yield/MFF_FullCV_01_X4/ --output_dir models/temp_test2/ --vocab data/vocab/mol_all_20201109.pt --task_prefix Yield: --max_target_length 2 --num_epoch 200 --pretrain models/temp_test/ --save_steps 3000 --save_total_limit 1
#python src/run_predict.py --data_dir data/C_N_yield/MFF_FullCV_01/ --model_dir models/temp_test2/ --task_prefix Yield: --max_target_length 2 --num_preds 5
#python src/evaluate_predictions.py --prediction models/temp_test2/predictions.csv --type value --average 1
#python src/evaluate_predictions.py --prediction models/temp_test2/predictions.csv --type value --average 2
#python src/evaluate_predictions.py --prediction models/temp_test2/predictions.csv --type value --average 3
#python src/evaluate_predictions.py --prediction models/temp_test2/predictions.csv --type value --average 4
#python src/evaluate_predictions.py --prediction models/temp_test2/predictions.csv --type value --average 5
