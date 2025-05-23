export CUDA_VISIBLE_DEVICES=0
python ft_qwen2_ultra.py \
    --model_path /home/zyx/qwen2vl/Qwen2-VL-2B-Instruct \
    --train_regression_data /mnt/e/datasets/galaxyWalker/train_add_feature.hdf5 \
    --train_classification_data /mnt/e/datasets/galaxyWalker/train_add_feature.hdf5 \
    --regression_eval_dir /mnt/e/datasets/galaxyWalker/ \
    --classification_eval_dir /mnt/e/datasets/galaxyWalker/classification \
    --eval_regression_tasks task1_redshift \
    --eval_classification_tasks smooth disk-edge-on spiral-arms bar bulge-size how-rounded edge-on-bulge spiral-winding spiral-arm-count merging\
    --image_dir /mnt/e/datasets/provabgs/images \
    --template_path template_ultra_qwen2vl_classification.json \
    --output_dir /mnt/e/outputs/galaxy_walker_stage2 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_steps 1000 \
    --save_steps 10000 \
    --samples_per_regression_task 5 \
    --samples_per_classification_task 50 \
    --num_train_epochs 3

