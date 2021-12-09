#t5chem train --data_dir data/C_N_yield/MFF_FullCV_01/ --output_dir models/C_N_yield/ --pretrain models/pretrain/simple/best_cp-10000/ --task_type regression
#t5chem predict --data_dir data/C_N_yield/MFF_FullCV_01/ --model_dir models/C_N_yield/

#t5chem train --data_dir data/C_N_yield/MFF_FullCV_02/ --output_dir models/C_N_yield/ --pretrain models/pretrain/simple/best_cp-10000/ --task_type regression
#t5chem predict --data_dir data/C_N_yield/MFF_FullCV_02/ --model_dir models/C_N_yield/

#t5chem train --data_dir data/C_N_yield/MFF_FullCV_03/ --output_dir models/C_N_yield/ --pretrain models/pretrain/simple/best_cp-10000/ --task_type regression
#t5chem predict --data_dir data/C_N_yield/MFF_FullCV_03/ --model_dir models/C_N_yield/

#t5chem train --data_dir data/C_N_yield/MFF_FullCV_04/ --output_dir models/C_N_yield/ --pretrain models/pretrain/simple/best_cp-10000/ --task_type regression
#t5chem predict --data_dir data/C_N_yield/MFF_FullCV_04/ --model_dir models/C_N_yield/

#t5chem train --data_dir data/C_N_yield/MFF_FullCV_05/ --output_dir models/C_N_yield/ --pretrain models/pretrain/simple/best_cp-10000/ --task_type regression
#t5chem predict --data_dir data/C_N_yield/MFF_FullCV_05/ --model_dir models/C_N_yield/

#t5chem train --data_dir data/C_N_yield/MFF_FullCV_06/ --output_dir models/C_N_yield/ --pretrain models/pretrain/simple/best_cp-10000/ --task_type regression
#t5chem predict --data_dir data/C_N_yield/MFF_FullCV_06/ --model_dir models/C_N_yield/

#t5chem train --data_dir data/C_N_yield/MFF_FullCV_07/ --output_dir models/C_N_yield/ --pretrain models/pretrain/simple/best_cp-10000/ --task_type regression
#t5chem predict --data_dir data/C_N_yield/MFF_FullCV_07/ --model_dir models/C_N_yield/

#t5chem train --data_dir data/C_N_yield/MFF_FullCV_08/ --output_dir models/C_N_yield/ --pretrain models/pretrain/simple/best_cp-10000/ --task_type regression
#t5chem predict --data_dir data/C_N_yield/MFF_FullCV_08/ --model_dir models/C_N_yield/

#t5chem train --data_dir data/C_N_yield/MFF_FullCV_09/ --output_dir models/C_N_yield/ --pretrain models/pretrain/simple/best_cp-10000/ --task_type regression
#t5chem predict --data_dir data/C_N_yield/MFF_FullCV_09/ --model_dir models/C_N_yield/

#t5chem train --data_dir data/C_N_yield/MFF_FullCV_10/ --output_dir models/C_N_yield/ --pretrain models/pretrain/simple/best_cp-10000/ --task_type regression
#t5chem predict --data_dir data/C_N_yield/MFF_FullCV_10/ --model_dir models/C_N_yield/

t5chem train --data_dir data/C_N_yield/MFF_Test1/ --output_dir models/C_N_yield/ --pretrain models/pretrain/simple/ --task_type regression --random_seed 42
t5chem predict --data_dir data/C_N_yield/MFF_Test1/ --model_dir models/C_N_yield/

t5chem train --data_dir data/C_N_yield/MFF_Test2/ --output_dir models/C_N_yield/ --pretrain models/pretrain/simple/ --task_type regression --random_seed 42
t5chem predict --data_dir data/C_N_yield/MFF_Test2/ --model_dir models/C_N_yield/

t5chem train --data_dir data/C_N_yield/MFF_Test3/ --output_dir models/C_N_yield/ --pretrain models/pretrain/simple/ --task_type regression --random_seed 42
t5chem predict --data_dir data/C_N_yield/MFF_Test3/ --model_dir models/C_N_yield/

t5chem train --data_dir data/C_N_yield/MFF_Test4/ --output_dir models/C_N_yield/ --pretrain models/pretrain/simple/ --task_type regression --random_seed 42
t5chem predict --data_dir data/C_N_yield/MFF_Test4/ --model_dir models/C_N_yield/