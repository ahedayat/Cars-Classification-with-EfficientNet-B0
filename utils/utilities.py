"""
Utilities of Project
"""

import argparse


def get_args():
    """
    Argunments of:
        - `train.py`
        - `test.py`
    """
    parser = argparse.ArgumentParser(
        description='Arguemnt Parser of `Train` and `Evaluation` of deep neural network.')

    # Hardware
    parser.add_argument('-g', '--gpu', action='store_true', dest='gpu',
                        default=True, help='use cuda')
    parser.add_argument('-w', '--num-worker', dest='num_workers', default=1,
                        type=int, help='use cuda')

    # CNN Backbone
    parser.add_argument('--load-model', dest='load_model', default=None,
                        type=str, help='Path to the model you want to load it')
    parser.add_argument('--pretrained', action='store_true', dest='pretrained',
                        default=True, help='use pretrained model')

    # Input Size
    parser.add_argument('--input-height', dest='input_height', default=224, type=int,
                        help='Height of an input')
    parser.add_argument('--input-width', dest='input_width', default=224, type=int,
                        help='Width of an input')

    # Data Path
    # - Train
    parser.add_argument('--train-base-dir', dest='train_base_dir', default="./dataset/stanford_car_196/cars_train",
                        type=str, help='train dataset base directory')
    parser.add_argument('--train-df-path', dest='train_df_path', default="./dataset/stanford_car_196/train.csv",
                        type=str, help='train dataset Dataframe path')
    # - validation
    # parser.add_argument('--val-base-dir', dest='val_base_dir', default="./dataset/classification/test1000",
    #                     type=str, help='validation dataset base directory')
    # parser.add_argument('--val-df-path', dest='val_df_path', default="./dataset/classification/test.csv",
    #                     type=str, help='validation dataset Dataframe path')
    # - Test
    parser.add_argument('--test-base-dir', dest='test_base_dir', default="./dataset/classification/train1000",
                        type=str, help='Test dataset base directory')
    parser.add_argument('--test-df-path', dest='test_df_path',
                        type=str, help='Test dataset Dataframe path')

    # Loss Function
    parser.add_argument('--criterion', dest='criterion', default='cross_entropy',
                        type=str, help="loss functions: { 'cross_entropy'}")

    # Optimizer and its hyper-parameters
    parser.add_argument('-o', '--optimizer', dest='optimizer', default='adam',
                        type=str, help='optimization method: { adam, sgd }')
    parser.add_argument('-r', '--learning-rate', dest='lr', default=1e-3,
                        type=float, help='learning rate')

    # # Distance Function
    # parser.add_argument('-d', '--distance', dest='distance', default='cosine',
    #                     type=str, help="optimization method: { 'cosine', 'euclidean' }")

    # Training Hyper-Parameter
    parser.add_argument('-e', '--epoch', dest='epoch', default=5, type=int,
                        help='number of epochs')
    parser.add_argument('-b', '--batch-size', dest='batch_size', default=16,
                        type=int, help='batch size')

    # Saving Paths
    # - Check Points
    parser.add_argument('--save-freq', dest='save_freq', default=1,
                        type=int, help='Saving Frequency')
    parser.add_argument('--ckpt-path', dest='ckpt_path', default="./checkpoints/",
                        type=str, help='Saving check points path')
    parser.add_argument('--ckpt-prefix', dest='ckpt_prefix', default="ckpt_",
                        type=str, help='Check points is saved with this prefix')
    parser.add_argument('--ckpt-load', dest='ckpt_load',
                        type=str, help='Load check point')

    # - Report
    parser.add_argument('--report', dest='report', default="./reports",
                        type=str, help='Saving reports path')

    options = parser.parse_args()

    return options
