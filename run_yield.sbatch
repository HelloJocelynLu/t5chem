#python src/run_trainer.py --data_dir data/USPTO_yield_X2/ --output_dir models/USPTO_yield/cosine_lr/ --vocab data/vocab/mol_all_20201109.pt --task_prefix Yield: --max_target_length 2 --pretrain models/smiles_pretrain/new_vocab_9 --num_epoch 60 --save_total_limit 10 --lr_scheduler cosine
#python src/run_predict.py --data_dir data/USPTO_yield/ --model_dir models/USPTO_yield/cosine_lr/ --task_prefix Yield: --max_target_length 2 --num_preds 3
#python src/evaluate_predictions.py --prediction models/USPTO_yield/cosine_lr/predictions.csv --type value
#python src/run_trainer.py --data_dir data/USPTO_yield/ --output_dir models/USPTO_yield/constant_lr/ --vocab data/vocab/mol_all_20201109.pt --task_prefix Yield: --max_target_length 2 --pretrain models/USPTO_yield/two_dig/ --num_epoch 15 --save_total_limit 1
#python src/run_predict.py --data_dir data/USPTO_yield/ --model_dir models/USPTO_yield/constant_lr/ --task_prefix Yield: --max_target_length 2 --num_preds 3
python src/evaluate_predictions.py --prediction models/USPTO_yield//predictions.csv --type value
#python src/run_pretrain.py --data_dir data/pubchem/train_8/ --output_dir models/smiles_pretrain/new_vocab_8 --pretrain models/smiles_pretrain/new_vocab_7 --vocab data/vocab/mol_all_20201109.pt --save_steps 200000 --save_total_limit 2
#python src/run_pretrain.py --data_dir data/pubchem/train_9/ --output_dir models/smiles_pretrain/new_vocab_9 --pretrain models/smiles_pretrain/new_vocab_8 --vocab data/vocab/mol_all_20201109.pt --save_steps 200000 --save_total_limit 2
#for (( c=1; c<=4; c++ ))
#do
#    cp models/temp_test/vocab.pt models/temp_test/checkpoint-$c/
#    rm -rf models/temp_test2/checkpoint*
#    python src/run_trainer.py --data_dir data/C_N_yield/MFF_Test${c}_X4/ --output_dir models/temp_test2/ --vocab data/vocab/mol_all_20201109.pt --task_prefix Yield: --max_target_length 2 --num_epoch 100 --pretrain models/smiles_pretrain/new_vocab_9/ --save_steps 3000 --save_total_limit 1
#    python src/run_predict.py --data_dir data/C_N_yield/MFF_Test${c}/ --model_dir models/temp_test2/ --task_prefix Yield: --max_target_length 2 --num_preds 1 --prediction models/temp_test2/predictions.csv
#    python src/evaluate_predictions.py --prediction models/temp_test2/predictions.csv --type value
#done
#rm -rf models/temp_test/checkpoint*
#python src/run_trainer.py --data_dir data/C_N_yield/MFF_FullCV_01/ --output_dir models/temp_test/ --vocab data/vocab/mol_all_20201109.pt --task_prefix Yield: --max_target_length 2 --num_epoch 400 --pretrain models/USPTO_yield/prefix/best_cp-206000/ --save_steps 3000 --save_total_limit 10
#    python src/run_predict.py --data_dir data/C_N_yield/MFF_FullCV_10/ --model_dir models/temp_test/ --task_prefix Yield: --max_target_length 5 --num_preds 1
#    python src/evaluate_predictions.py --prediction models/temp_test/predictions.csv --type value
