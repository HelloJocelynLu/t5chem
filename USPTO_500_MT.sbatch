t5chem train --data_dir ../t5chem_data/USPTO_MT_X5/mixed/ --output_dir models/USPTO_MT_augm/ --pretrain models/USPTO_MT_augm/ --task_type mixed --num_epoch 20
t5chem predict --data_dir ../t5chem_data/USPTO_MT/Product/ --model_dir models/USPTO_MT_augm/ --prefix Product:
t5chem predict --data_dir ../t5chem_data/USPTO_MT/Reactants/ --model_dir models/USPTO_MT_augm/ --prefix Reactants:
t5chem predict --data_dir ../t5chem_data/USPTO_MT/Reagents/ --model_dir models/USPTO_MT_augm/ --prefix Reagents: