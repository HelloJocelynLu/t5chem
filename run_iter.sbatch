#python src/run_trainer.py --data_dir data/USPTO_1k_TPL/bidirectional/toy_set/ --output_dir models/USPTO_TPL/iterativeAB/A1/ --vocab data/vocab/mol_all_simpletokenizer.pt --task_prefix '' --num_epoch 5 --vocab_size 100 --learning_rate 5e-4 --pretrain models/USPTO_TPL/bidirectional/ --max_target_length 300 --log_steps 500 --max_source_length 400 --tokenizer simple --batch_size 32
#for (( c=1; c<=4; c++ ))
#do
#    python src/run_predict.py --data_dir data/USPTO_1k_TPL/bidirectional/toy_set --model_dir models/USPTO_TPL/iterativeAB/A${c} --num_beams 10 --batch_size 64 --max_target_length 300 --task_prefix '' --vocab_size 100 --tokenizer simple --max_source_length 400
#    python src/evaluate_predictions.py --prediction models/USPTO_TPL/iterativeAB/A${c}/predictions.csv --type text
#    rm -rf models/USPTO_TPL/iterativeAB/A${c}/checkpoint-*
#    python src/run_trainer_KLD.py --data_dir data/USPTO_1k_TPL/Classification/toy_set/ --output_dir models/USPTO_TPL/iterativeAB/B${c} --vocab data/vocab/mol_all_simpletokenizer.pt --task_prefix Classification: --num_epoch 20 --learning_rate 5e-4 --pretrain models/USPTO_TPL/iterativeAB/A${c} --log_steps 500 --batch_size 32 --max_source_length 500 --loss_type CEE --num_classes 1000
#    python src/run_predict_KLDiv.py --data_dir data/USPTO_1k_TPL/Classification/toy_set --model_dir models/USPTO_TPL/iterativeAB/B${c} --batch_size 64 --max_source_length 500 --task_prefix Classification: --tokenizer simple
#    python src/evaluate_predictions.py --prediction models/USPTO_TPL/iterativeAB/B${c}/predictions.csv --simple
#    rm -rf models/USPTO_TPL/iterativeAB/B${c}/checkpoint-*
#    d=`expr ${c} + 1`
#    python src/run_trainer.py --data_dir data/USPTO_1k_TPL/bidirectional/toy_set/ --output_dir models/USPTO_TPL/iterativeAB/A${d}/ --vocab data/vocab/mol_all_simpletokenizer.pt --task_prefix '' --num_epoch 5 --vocab_size 100 --learning_rate 5e-4 --pretrain models/USPTO_TPL/iterativeAB/B${c} --max_target_length 300 --log_steps 500 --max_source_length 400 --tokenizer simple --batch_size 32
#done
python src/run_trainer_KLD.py --data_dir data/USPTO_1k_TPL/Classification/toy_set/ --output_dir models/USPTO_TPL/iterativeAB/B5 --vocab data/vocab/mol_all_simpletokenizer.pt --task_prefix Classification: --num_epoch 20 --learning_rate 5e-4 --pretrain models/USPTO_TPL/iterativeAB/A5 --log_steps 500 --batch_size 32 --max_source_length 500 --loss_type CEE --num_classes 1000 --freeze_pretrain
python src/run_predict_KLDiv.py --data_dir data/USPTO_1k_TPL/Classification/ --model_dir models/USPTO_TPL/iterativeAB/B5 --batch_size 64 --max_source_length 500 --task_prefix Classification: --tokenizer simple
python src/evaluate_predictions.py --prediction models/USPTO_TPL/iterativeAB/B5/predictions.csv --simple
rm -rf models/USPTO_TPL/iterativeAB/B5/checkpoint-*
