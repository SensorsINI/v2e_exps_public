# train on V2E events, test on real events, 10 slices
v2e_mvsec_day_train:
	python ./train_ev_yolo.py \
		--img_size 416 \
		--num_epochs 100 \
		--batch_size 8 \
		--train_data_dir ${HOME}/data/v2e_mvsec_train_data \
		--val_data_dir ${HOME}/data/mvsec_data/val_data_final \
		--log_dir ${HOME}/data/iccv_v2e_exps/v2e_mvsec_day_train \
		--cut_stage 3 \
		--eval_stage 5 \
		--num_eval_outputs 1024 \
		--conv_input_dim 10 \
		--frl 1. \
		--fel 1. \
		--frl_gram 0. \
		--fel_gram 0. \
		--fel_context 0. \
		--tv 0.
# 1e5, 1e7
finetune_v2e_mvsec_day_train:
	python ./finetune_ev_yolo.py \
		--img_size 416 \
		--num_epochs 10 \
		--batch_size 8 \
		--train_data_dir ${HOME}/data/mvsec_data/train_data_final \
		--val_data_dir ${HOME}/data/mvsec_data/val_data_final \
		--log_dir ${HOME}/data/iccv_v2e_exps/finetune_v2e_mvsec_day_train \
		--checkpoint ${HOME}/data/iccv_v2e_exps/v2e_mvsec_day_train/004 \
		--cut_stage 3 \
		--eval_stage 5 \
		--num_eval_outputs 1024 \
		--conv_input_dim 10 \
		--frl 1. \
		--fel 1. \
		--frl_gram 0. \
		--fel_gram 0. \
		--fel_context 0. \
		--tv 0.
# 1e5, 1e7

finetune_v2e_mvsec_day_train_500_samples:
	python ./finetune_ev_yolo.py \
		--img_size 416 \
		--num_epochs 10 \
		--batch_size 8 \
		--train_data_dir ${HOME}/data/mvsec_data/train_data_final \
		--val_data_dir ${HOME}/data/mvsec_data/val_data_final \
		--sample_limit 500 \
		--log_dir ${HOME}/data/iccv_v2e_exps/finetune_v2e_mvsec_day_train_500_samples \
		--checkpoint ${HOME}/data/iccv_v2e_exps/v2e_mvsec_day_train/004 \
		--cut_stage 3 \
		--eval_stage 5 \
		--num_eval_outputs 1024 \
		--conv_input_dim 10 \
		--frl 1. \
		--fel 1. \
		--frl_gram 0. \
		--fel_gram 0. \
		--fel_context 0. \
		--tv 0.
# 1e5, 1e7

# finetune the model trained on v2e day events with night data
finetune_v2e_mvsec_night_train:
	python ./finetune_ev_yolo.py \
		--img_size 416 \
		--num_epochs 10 \
		--batch_size 8 \
		--train_data_dir ${HOME}/data/mvsec_data/mvsec_night_data \
		--val_data_dir ${HOME}/data/mvsec_data/val_night_data \
		--log_dir ${HOME}/data/iccv_v2e_exps/finetune_v2e_mvsec_night_train \
		--checkpoint ${HOME}/data/iccv_v2e_exps/v2e_mvsec_day_train/003 \
		--cut_stage 3 \
		--eval_stage 5 \
		--num_eval_outputs 1024 \
		--conv_input_dim 10 \
		--frl 1. \
		--fel 1. \
		--frl_gram 0. \
		--fel_gram 0. \
		--fel_context 0. \
		--tv 0.
# 1e5, 1e7

# train on real night, test on real night
v2e_mvsec_real_night_train:
	python ./train_ev_yolo.py \
		--img_size 416 \
		--num_epochs 100 \
		--batch_size 8 \
		--train_data_dir ${HOME}/data/mvsec_data/mvsec_night_data \
		--val_data_dir ${HOME}/data/mvsec_data/val_night_data \
		--log_dir ${HOME}/data/iccv_v2e_exps/v2e_mvsec_real_night_train \
		--cut_stage 3 \
		--eval_stage 5 \
		--num_eval_outputs 1024 \
		--conv_input_dim 10 \
		--frl 1. \
		--fel 1. \
		--frl_gram 0. \
		--fel_gram 0. \
		--fel_context 0. \
		--tv 0.

# train on V2E night, test on real night
v2e_mvsec_v2e_night_train:
	python ./train_ev_yolo.py \
		--img_size 416 \
		--num_epochs 30 \
		--batch_size 8 \
		--night \
		--train_data_dir ${HOME}/data/v2e_mvsec_train_data \
		--val_data_dir ${HOME}/data/mvsec_data/val_night_data \
		--log_dir ${HOME}/data/iccv_v2e_exps/v2e_mvsec_v2e_night_train \
		--cut_stage 3 \
		--eval_stage 5 \
		--num_eval_outputs 1024 \
		--conv_input_dim 10 \
		--frl 1. \
		--fel 1. \
		--frl_gram 0. \
		--fel_gram 0. \
		--fel_context 0. \
		--tv 0.
