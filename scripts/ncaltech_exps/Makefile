v2e-ncaltech101-bright-exp:
	python -W ignore train.py \
		--v2e_ncaltech_exp \
		--train_data_root ~/data/v2e_ncaltech101_bright \
		--train_ext .h5 \
		--valid_data_root ~/data/ncaltech \
		--valid_ext .bin \
		--test_data_root ~/data/ncaltech \
		--test_ext .bin \
		--data_list ../res/ncaltech_data_percentage_split.pkl \
		--num_voxel_bins 15 \
		--augmentation \
		--log_root ~/data/iccv_v2e_exps/v2e_ncaltech101_bright_exp \
		--num_epochs 100 \
		--use_pretrained \
		--batch_size 4 \
		--learning_rate 1e-4 \
		# --tb_start --tb_port 6006 \

v2e-ncaltech101-ideal-exp:
	python -W ignore train.py \
		--v2e_ncaltech_exp \
		--train_data_root ~/data/v2e_ncaltech101_ideal \
		--train_ext .h5 \
		--valid_data_root ~/data/ncaltech \
		--valid_ext .bin \
		--test_data_root ~/data/ncaltech \
		--test_ext .bin \
		--data_list ../res/ncaltech_data_percentage_split.pkl \
		--num_voxel_bins 15 \
		--augmentation \
		--log_root ~/data/iccv_v2e_exps/v2e_ncaltech101_ideal_exp \
		--num_epochs 100 \
		--use_pretrained \
		--batch_size 4 \
		--learning_rate 1e-4 \
		# --tb_start --tb_port 6006 \

v2e-ncaltech101-dark-exp:
	python -W ignore train.py \
		--v2e_ncaltech_exp \
		--train_data_root ~/data/v2e_ncaltech101_dark \
		--train_ext .h5 \
		--valid_data_root ~/data/ncaltech \
		--valid_ext .bin \
		--test_data_root ~/data/ncaltech \
		--test_ext .bin \
		--data_list ../res/ncaltech_data_percentage_split.pkl \
		--num_voxel_bins 15 \
		--augmentation \
		--log_root ~/data/iccv_v2e_exps/v2e_ncaltech101_dark_exp \
		--num_epochs 100 \
		--use_pretrained \
		--batch_size 4 \
		--learning_rate 1e-4 \
		# --tb_start --tb_port 6006 \

# data combined training
v2e-ncaltech101-combined-exp:
	python -W ignore train.py \
		--v2e_ncaltech_exp \
		--train_data_root ~/data/v2e_ncaltech101_all \
		--train_ext .h5 \
		--valid_data_root ~/data/ncaltech \
		--valid_ext .bin \
		--test_data_root ~/data/ncaltech \
		--test_ext .bin \
		--data_list ../res/ncaltech_data_percentage_split.pkl \
		--num_voxel_bins 15 \
		--augmentation \
		--log_root ~/data/iccv_v2e_exps/v2e_ncaltech101_combined_exp \
		--num_epochs 33 \
		--use_pretrained \
		--batch_size 4 \
		--learning_rate 1e-4 \
		--step_size 10 \
		# --tb_start --tb_port 6006 \

# data combined training: bright+dark
v2e-ncaltech101-bright-and-dark-exp:
	python -W ignore train.py \
		--v2e_ncaltech_exp \
		--train_data_root ~/data/v2e_ncaltech101_bnd \
		--train_ext .h5 \
		--valid_data_root ~/data/ncaltech \
		--valid_ext .bin \
		--test_data_root ~/data/ncaltech \
		--test_ext .bin \
		--data_list ../res/ncaltech_data_percentage_split.pkl \
		--num_voxel_bins 15 \
		--augmentation \
		--log_root ~/data/iccv_v2e_exps/v2e_ncaltech101_bright_and_dark_exp \
		--num_epochs 50 \
		--use_pretrained \
		--batch_size 4 \
		--learning_rate 1e-4 \
		--step_size 15 \
		# --tb_start --tb_port 6006 \

# data combined training: bright+additional bright
v2e-ncaltech101-bright-and-additional-bright-exp:
	python -W ignore train.py \
		--v2e_ncaltech_exp \
		--train_data_root ~/data/v2e_ncaltech101_addb \
		--train_ext .h5 \
		--valid_data_root ~/data/ncaltech \
		--valid_ext .bin \
		--test_data_root ~/data/ncaltech \
		--test_ext .bin \
		--data_list ../res/ncaltech_data_percentage_split.pkl \
		--num_voxel_bins 15 \
		--augmentation \
		--log_root ~/data/iccv_v2e_exps/v2e_ncaltech101_bright_and_additional_bright_exp\
		--num_epochs 50 \
		--use_pretrained \
		--batch_size 4 \
		--learning_rate 1e-4 \
		--step_size 15 \
		# --tb_start --tb_port 6006 \

# our own reference N-Caltech experiments using original events
original-ncaltech101-exp:
	python -W ignore train.py \
		--v2e_ncaltech_exp \
		--train_data_root ~/data/ncaltech \
		--train_ext .bin \
		--valid_data_root ~/data/ncaltech \
		--valid_ext .bin \
		--test_data_root ~/data/ncaltech \
		--test_ext .bin \
		--data_list ../res/ncaltech_data_percentage_split.pkl \
		--num_voxel_bins 15 \
		--augmentation \
		--log_root ~/data/iccv_v2e_exps/original_ncaltech101_exp \
		--num_epochs 100 \
		--use_pretrained \
		--batch_size 4 \
		--learning_rate 1e-4 \
		# --tb_start --tb_port 6006 \

# Finetuning experiments
# Train on V2E Combined, finetune on real
finetune-v2e-ncaltech101-combined-exp:
	python -W ignore train.py \
		--finetune_v2e_ncaltech_exp\
		--train_data_root ~/data/ncaltech \
		--train_ext .bin \
		--valid_data_root ~/data/ncaltech \
		--valid_ext .bin \
		--test_data_root ~/data/ncaltech \
		--test_ext .bin \
		--data_list ../res/ncaltech_data_percentage_split.pkl \
		--num_voxel_bins 15 \
		--augmentation \
		--log_root ~/data/iccv_v2e_exps/finetune_v2e_combined_to_real_exp \
		--saved_checkpoint_path ~/data/iccv_v2e_exps/v2e_ncaltech101_combined_exp/004/checkpoint.ckpt \
		--num_epochs 100 \
		--use_pretrained \
		--batch_size 4 \
		--learning_rate 1e-6 \
		# --tb_start --tb_port 6006 \

# EVALUATION
evaluate-on-v2e-ncaltech101:
	python -W ignore train.py \
		--evaluate \
		--v2e_ncaltech_exp \
		--train_data_root ~/data/ncaltech \
		--train_ext .bin \
		--valid_data_root ~/data/ncaltech \
		--valid_ext .bin \
		--test_data_root ~/data/ncaltech \
		--test_ext .bin \
		--data_list ../res/ncaltech_data_percentage_split.pkl \
		--num_voxel_bins 15 \
		--log_root ~/data/iccv_v2e_exps/original_ncaltech101_exp \
		--saved_checkpoint_path ~/data/iccv_v2e_exps/original_ncaltech101_exp/005/checkpoint.ckpt \
		--num_epochs 100 \
		--use_pretrained \
		--batch_size 4 \
		--learning_rate 1e-5 \
