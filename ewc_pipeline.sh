#!/bin/bash

# ------------- Training the Continual Learner up to the Last Task -------------

main_baselines_args=(
    # experiment name (dataset)
    --experiment split_cifar100
    # continual learning approach
    --approach ewc
    # last task index
    --lasttask 9
    # number of tasks
    --tasknum 10
    # number of epochs for training each task
    --nepochs 20
    # batch size
    --batch-size 16
    # stability-plasticity hyperparameter
    --lamb 500000
    # gradient clipping
    --clip 100.0
    # learning rate
    --lr 0.01
)

CUDA_VISIBLE_DEVICES=0 python main_baselines.py "${main_baselines_args[@]}"

# --------------------------- Model Inversion Attack ---------------------------

main_inv_args=(
    # location of the victim model
    --pretrained_model_add ewc_lamb_500000.0__model_type_resnet_dataset_split_cifar100_class_num_10_bs_16_lr_0.01_n_epochs_20__model_name_ResNet_task_num_9__seed_0_emb_fact_1_im_sz_32__last_task_9___.pkl
    # number of the inverted samples per tasks
    --num_samples 128
    # address for saving the inverted samples
    --save_dir cifar100_inverted_data_ewc
    # list of the previous tasks for inversion. E.g., --task_lst=1,2,3
    --task_lst 0,1,2,3,4,5,6,7,8
    # saving interval in the midst of the inversion (prints the loss)
    --save_every 1000
    # flag for using the batch norm, tv, and l2 regularizations in the inversion,
    # if not set, the inversion optimizes the cross entropy loss only
    --batch_reg
    # evaluate the pretrained model on the prev task, useful for debugging
    --init_acc
    # number of optimization steps for the inversion
    --n_iters 10000
)

CUDA_VISIBLE_DEVICES=0 python main_inv.py "${main_inv_args[@]}"

# ----------------------------- Poisoning Process ------------------------------

main_brainwash_args=(
    # extra description for saving the results, it will appear in pkl filenames
    --extra_desc reckless_test
    # location of the victim model
    --pretrained_model_add ewc_lamb_500000.0__model_type_resnet_dataset_split_cifar100_class_num_10_bs_16_lr_0.01_n_epochs_20__model_name_ResNet_task_num_9__seed_0_emb_fact_1_im_sz_32__last_task_9___.pkl
    # reckless or cautious attack
    --mode 'reckless'
    # task for evaluating brainwash effectiveness' (the loss is computed on all tasks)
    --target_task_for_eval 0
    # ell inf norm of the noise
    --delta 0.3
    # random seed
    --seed 0
    # evaluation interval
    --eval_every 10
    # folder for loading the inversion images
    --distill_folder cifar100_inverted_data_ewc
    # evaluate the pretrained model on the prev task, useful for debugging
    --init_acc
    # type of the noise norm, default is inf
    --noise_norm inf
    # learning rate for taking the pseudo step when training the continual learner with the poisoned data
    --cont_learner_lr 0.001
    # number of epochs for the noise training
    --n_epochs 5000
    # saving interval in the midst of the noise training (overwrites the previous noise)
    --save_every 100
)

CUDA_VISIBLE_DEVICES=0 python main_brainwash.py "${main_brainwash_args[@]}"

# ------------------------- Evaluation without PROACT --------------------------

# On poisoned data

main_baselines_args=(
    # experiment name (dataset)
    --experiment split_cifar100
    # continual learning approach
    --approach ewc
    # last task index
    --lasttask 9
    # number of tasks
    --tasknum 10
    # number of epochs for training the last task
    --nepochs 20
    # batch size
    --batch-size 16
    # learning rate
    --lr 0.01
    # gradient clipping
    --clip 100.0
    # stability-plasticity hyperparameter
    --lamb 500000
    # location of the noise
    --checkpoint noise_ewc_reckless_test__delta_0.3_dataset_split_cifar100_target_task_0_attacked_task_9_noise_optim_lr_0.005__n_iters_1_n_epochs_5000_seed_0_mode_reckless____min_acc_target_30.pkl
    # evaluate the pretrained model on the prev task, useful for debugging
    --init_acc
    # add noise to the samples
    --addnoise
)

CUDA_VISIBLE_DEVICES=0 python main_baselines.py "${main_baselines_args[@]}"

# On clean data

main_baselines_args=(
    # experiment name (dataset)
    --experiment split_cifar100
    # continual learning approach
    --approach ewc
    # last task index
    --lasttask 9
    # number of tasks
    --tasknum 10
    # number of epochs for training the last task
    --nepochs 20
    # batch size
    --batch-size 16
    # learning rate
    --lr 0.01
    # gradient clipping
    --clip 100.0
    # stability-plasticity hyperparameter
    --lamb 500000
    # location of the noise
    --checkpoint noise_ewc_reckless_test__delta_0.3_dataset_split_cifar100_target_task_0_attacked_task_9_noise_optim_lr_0.005__n_iters_1_n_epochs_5000_seed_0_mode_reckless____min_acc_target_30.pkl
    # evaluate the pretrained model on the prev task, useful for debugging
    --init_acc
)

CUDA_VISIBLE_DEVICES=0 python main_baselines.py "${main_baselines_args[@]}"

# Renaming results

mv acc_mat_noise_ewc_reckless_test__delta_0.3_dataset_split_cifar100_target_task_0_attacked_task_9_noise_optim_lr_0.005__n_iters_1_n_epochs_5000_seed_0_mode_reckless____min_acc_target_30_ours.npy ./acc_mat_noise_ewc.npy
mv acc_mat_noise_ewc_reckless_test__delta_0.3_dataset_split_cifar100_target_task_0_attacked_task_9_noise_optim_lr_0.005__n_iters_1_n_epochs_5000_seed_0_mode_reckless____min_acc_target_30_clean.npy ./acc_mat_ewc.npy

# -------------------------- Evaluation with PROACT ----------------------------

# On poisoned data

main_baselines_args=(
    # experiment name (dataset)
    --experiment split_cifar100
    # continual learning approach
    --approach ewc
    # last task index
    --lasttask 9
    # number of tasks
    --tasknum 10
    # number of epochs for training on the last task
    --nepochs 20
    # batch size
    --batch-size 16
    # learning rate
    --lr 0.01
    # gradient clipping
    --clip 100.0
    # stability-plasticity hyperparameter
    --lamb 500000
    # location of the noise
    --checkpoint ./noise_ewc_reckless_test__delta_0.3_dataset_split_cifar100_target_task_0_attacked_task_9_noise_optim_lr_0.005__n_iters_1_n_epochs_5000_seed_0_mode_reckless____min_acc_target_30.pkl
    # evaluate the pretrained model on the prev task, useful for debugging
    --init_acc
    # add noise to the samples
    --addnoise
    # folder where distillation data is saved
    --distill_folder cifar100_inverted_data_ewc
    # use defense mechanism
    --defend
    # use AGEM (Averaged Gradient Episodic Memory)
    --agem
    # sample size for the inverted data
    --inverted_sample_size 1152
    # batch size for the inverted data
    --inverted_batch_size 128
    # turn off regularization on projection
    --reg_turn_off_on_projection
    # use activation regularization
    --act_reg
    # weight for the activation regularization
    --lamb_act 0.1
    # output directory for results
    --output_dir ./
)

CUDA_VISIBLE_DEVICES=0 python main_baselines.py "${main_baselines_args[@]}"

mv acc_mat_ours.npy ./acc_mat_noise_ewc_defense.npy
mv task9_AGEM_projections.csv ./noise_ewc_defense_task9_AGEM_projections.csv