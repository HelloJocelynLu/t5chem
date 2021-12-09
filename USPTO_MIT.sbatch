t5chem train --data_dir ../t5chem_data/USPTO_MIT/MIT_mixed/ --output_dir models/USPTO_MIT_mixed/ --pretrain models/pretrain/simple --task_type product --num_epoch 30
t5chem predict --data_dir ../t5chem_data/USPTO_MIT/MIT_mixed/ --model_dir models/USPTO_MIT_mixed/
t5chem train --data_dir ../t5chem_data/USPTO_MIT/MIT_separated/ --output_dir models/USPTO_MIT_separated/ --pretrain models/pretrain/simple --task_type product --num_epoch 30
t5chem predict --data_dir ../t5chem_data/USPTO_MIT/MIT_separated/ --model_dir models/USPTO_MIT_separated/
