#python run_predict.py --data_dir STEREO_separated/ --model_dir STEREO_new_scratch/ --prediction STEREO_new_scratch/test_pred_beam10.txt --num_beams 10 > rxn_pred.out
#python src/run_trainer.py --data_dir data/USPTO_50k/ --output_dir models/USPTO_50k/no_pretrain/ --pretrain models/smiles_pretrain/new_vocab_9/ --learning_rate 5e-4 --vocab data/vocab/mol_all_20201109.pt --task_prefix Retrosynthesis: --num_epoch 500
python src/run_trainer.py --data_dir data/USPTO_50k/ --vocab temp.pt --output_dir models/USPTO_50k/nop_selfies/ --learning_rate 5e-4 --task_prefix Retrosynthesis: --num_epoch 1000 --tokenizer selfies
#cp models/USPTO_50k/can/vocab.pt models/USPTO_50k/can/best_cp-32500/
python src/run_predict.py --data_dir data/USPTO_50k/ --model_dir models/USPTO_50k/nop_selfies/ --num_beams 10 --batch_size 32 --max_target_length 250 --tokenizer selfies
#python src/run_trainer.py --data_dir data/ResLig/ --output_dir models/ResLig --pretrain models/STEREO_sep/prefix/ --vocab data/vocab/mol_all_20201109.pt --task_prefix Fill-Mask: --num_epoch 100
#python src/run_predict.py --data_dir data/ResLig --model_dir models/ResLig --prediction models/ResLig/predictions.csv --num_beams 10 --batch_size 32
python src/evaluate_predictions.py --prediction models/USPTO_50k/nop_selfies/predictions.csv --type text
#python src/evaluate_score_rank.py --prediction models/USPTO_50k/augmX20/checkpoint-4800000/predictionsX20.csv
