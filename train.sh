# Hardware
num_worker=3

# Input Size
input_height=240
input_width=240

# Data Path
# - Train
train_base_dir="./datasets/stanford_car_196/cars_train"
train_df_path="./datasets/stanford_car_196/train.csv"

# - Validation
# val_base_dir="./datasets/stanford_car_196/cars_train"
# val_df_path="./datasets/stanford_car_196/train.csv"

# Loss Function
criterion="label_smoothing_cross_entropy"

# Optimizer and its hyper-parameters
optimizer="adam"
learning_rate=1e-3

# Training Hyper-Parameter
epoch=2
batch_size=2

# Saving Paths
save_freq=1
ckpt_path="./checkpoints"
ckpt_prefix="ckpt_"

# Report
report="./reports"

python train.py \
        --num-worker $num_worker \
        --gpu \
        --pretrained \
        --input-height $input_height \
        --input-width $input_width \
        --train-base-dir $train_base_dir \
        --train-df-path $train_df_path \
        --criterion  $criterion \
        --optimizer $optimizer \
        --learning-rate $learning_rate \
        --epoch $epoch \
        --batch-size $batch_size \
        --save-freq $save_freq \
        --ckpt-path $ckpt_path \
        --ckpt-prefix $ckpt_prefix \
        --report $report