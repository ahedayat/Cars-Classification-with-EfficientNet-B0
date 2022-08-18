# Hardware
num_worker=3

# Input Size
input_height=240
input_width=240

# Data Path
# - test
test_base_dir="./datasets/stanford_car_196/cars_test"
test_df_path="./datasets/stanford_car_196/test.csv"

# Loss Function
criterion="label_smoothing_cross_entropy"


# Training Hyper-Parameter
batch_size=2

# Saving Paths
ckpt_load="./checkpoints/ckpt__final"

# Report
report="./reports"

python test.py \
        --num-worker $num_worker \
        --gpu \
        --input-height $input_height \
        --input-width $input_width \
        --test-base-dir $test_base_dir \
        --test-df-path $test_df_path \
        --criterion  $criterion \
        --batch-size $batch_size \
        --ckpt-load $ckpt_load \
        --report $report