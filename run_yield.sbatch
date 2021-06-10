#python src/run_trainer.py --new_lm_head --data_dir data/USPTO_1k_TPL/ --output_dir models/USPTO_TPL/Classification --vocab data/vocab/mol_all_simpletokenizer.pt --task_prefix Classification: --num_epoch 100 --vocab_size 100 --learning_rate 5e-4 --pretrain models/simple_rxn/ --max_target_length 250 --log_steps 1500 --tokenizer simple --batch_size 64 --target_classes 1000
#python src/run_trainer_KLD.py --data_dir data/USPTO_1k_TPL/ --output_dir models/USPTO_TPL/Classification --vocab data/vocab/mol_all_simpletokenizer.pt --task_prefix Classification: --num_epoch 100 --learning_rate 5e-4 --pretrain models/simple_rxn/ --log_steps 1500 --batch_size 64 --max_source_length 500 --loss_type CEE --num_classes 1000
python src/run_trainer_KLD.py --data_dir data/USPTO_1k_TPL/Yield/ --output_dir models/USPTO_TPL/Multitask/Yield/ --vocab data/vocab/mol_all_simpletokenizer.pt --task_prefix Yield: --num_epoch 100 --learning_rate 5e-4 --pretrain models/USPTO_TPL/Multitask/Classification/ --log_steps 1000 --batch_size 64 --max_source_length 500 --loss_type KLD
#python src/run_pretrain.py --data_dir data/USPTO_1k_TPL_X9/ --output_dir models/simple_rxn2/ --pretrain models/simple_smiles2/train_9/ --vocab data/vocab/mol_all_simpletokenizer.pt --save_steps 200000 --save_total_limit 1 --tokenizer simple --vocab_size 100 --max_length 500 --num_epoch 10
python src/run_predict_KLDiv.py --data_dir data/USPTO_1k_TPL/Yield/ --model_dir models/USPTO_TPL/Multitask/Yield/ --batch_size 64 --max_source_length 500 --task_prefix Yield: --tokenizer simple
#python src/run_predict_KLDiv.py --data_dir data/USPTO_1k_TPL/Yield/ --model_dir models/USPTO_TPL/Yield/MSE/best_cp-36000/ --batch_size 64 --max_source_length 500 --task_prefix Yield: --tokenizer simple
#python src/run_predict_KLDiv.py --data_dir data/USPTO_1k_TPL/Yield/ --model_dir models/USPTO_TPL/Yield/CEE20/ --batch_size 64 --max_source_length 500 --task_prefix Yield: --tokenizer simple
#python src/run_predict.py --new_lm_head --data_dir data/USPTO_1k_TPL/Yield/10class/ --model_dir models/USPTO_TPL/Yield/AsClass10/ --task_prefix Classification: --max_target_length 1 --num_preds 5 --tokenizer simple --vocab_size 100
#python src/run_predict.py --data_dir data/USPTO_yield/ --model_dir models/USPTO_yield/one_two_dig/ --task_prefix Yield: --max_target_length 2 --num_preds 5 --prediction models/USPTO_yield/one_two_dig/predictions.csv
#python src/evaluate_predictions.py --prediction models/USPTO_yield/one_two_dig/predictions.csv --type value --average 3
#python src/run_trainer.py --data_dir data/USPTO_1k_TPL/Yield/ --output_dir models/USPTO_TPL/Yield/ --vocab data/vocab/USPTO_simpletokenizer.pt --task_prefix Yield: --max_target_length 2 --pretrain models/simple_smiles/train_9/ --num_epoch 100 --tokenizer simple --max_source_length 600 --vocab_size 100
#python src/run_pretrain.py --data_dir data/pubchem/train_${c}/ --output_dir models/simple_smiles2/train_${c}/ --pretrain models/simple_smiles2/train_${d}/ --vocab data/vocab/mol_all_simpletokenizer.pt --save_steps 200000 --save_total_limit 1 --tokenizer simple --vocab_size 100
#python src/run_predict.py --data_dir data/USPTO_1k_TPL/Yield/ --model_dir models/USPTO_TPL/Yield/ --task_prefix Yield: --max_target_length 2 --num_preds 5 --tokenizer simple --vocab_size 100 --max_source_length 600
#for (( c=1; c<=9; c++ ))
#do
#    d=`expr ${c} - 1`
#    python src/run_pretrain.py --data_dir data/pubchem/train_${c}/ --output_dir models/simple_smiles2/train_${c}/ --pretrain models/simple_smiles2/train_${d}/ --vocab data/vocab/mol_all_simpletokenizer.pt --save_steps 200000 --save_total_limit 1 --tokenizer simple --vocab_size 100
#done
#for (( c=1; c<=4; c++ ))
#do
##    cp models/temp_test/vocab.pt models/temp_test/checkpoint-$c/
#    rm -rf models/temp_test/checkpoint*
#    python src/run_trainer_KLD.py --data_dir data/C_N_yield/MFF_Test${c}/ --output_dir models/temp_MFF/ --vocab data/vocab/mol_all_simpletokenizer.pt --task_prefix Yield: --num_epoch 100 --learning_rate 5e-4 --pretrain models/simple_rxn/ --batch_size 64 --max_source_length 500 --loss_type KLD
#    python src/run_predict_KLDiv.py --data_dir data/C_N_yield/MFF_Test${c}/ --model_dir models/temp_MFF/ --batch_size 64 --max_source_length 500 --task_prefix Yield: --tokenizer simple
#    python src/evaluate_predictions.py --prediction models/temp_MFF/predictions.csv --type value
#    rm -rf models/temp_MFF/
#    python src/run_trainer.py --data_dir data/C_N_yield/MFF_Test${c}/ --output_dir models/temp_test/ --vocab data/vocab/USPTO_simpletokenizer.pt --vocab_size 100 --task_prefix Yield: --max_target_length 2 --num_epoch 200 --pretrain models/simple_smiles/train_9/ --save_steps 3000 --save_total_limit 1 --tokenizer simple
#    python src/run_predict.py --data_dir data/C_N_yield/MFF_Test${c}/ --model_dir models/temp_test/ --task_prefix Yield: --max_target_length 2 --num_preds 5 --vocab_size 100 --prediction models/temp_test/predictions.csv --tokenizer simple
#    python src/evaluate_predictions.py --prediction models/temp_test/predictions.csv --type value --average 3
#done
#python src/evaluate_predictions.py --prediction models/USPTO_TPL/Classification/toy_set/predictions.csv --simple
#python src/evaluate_predictions.py --prediction models/USPTO_TPL/Yield/MSE/best_cp-36000/predictions.csv --type value
python src/evaluate_predictions.py --prediction models/USPTO_TPL/Multitask/Yield/predictions.csv --simple
