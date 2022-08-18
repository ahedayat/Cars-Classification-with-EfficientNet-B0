import os
import math
import random
import scipy.io
import pandas as pd
from tqdm import tqdm
import torchvision


def read_data_mat_file(annot_base_dir, annot_name, image_base_dir):
    """
        Making a data frame for Dataset annotations
    """

    data_dict = scipy.io.loadmat(os.path.join(annot_base_dir, annot_name))
    df = pd.DataFrame(columns=["bbox_x1", "bbox_y1",
                      "bbox_x2", "bbox_y2", "file_name", "gray_scale"])
    data_list = data_dict['annotations'][0, :].tolist()

    with tqdm(data_list) as t_data:
        for ix, data in enumerate(t_data):
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, label, file_name = data
            bbox_x1 = bbox_x1[0][0]
            bbox_y1 = bbox_y1[0][0]
            bbox_x2 = bbox_x2[0][0]
            bbox_y2 = bbox_y2[0][0]
            label = label[0][0]
            file_name = file_name[0]

            image = torchvision.io.read_image(
                os.path.join(image_base_dir, file_name))
            gray_scale = False
            if image.shape[0] == 1:
                gray_scale = True

            df = df.append({
                "bbox_x1": bbox_x1,
                "bbox_y1": bbox_y1,
                "bbox_x2": bbox_x2,
                "bbox_y2": bbox_y2,
                "label": label-1,
                "file_name": file_name,
                "gray_scale": gray_scale
            }, ignore_index=True)

    df["bbox_x1"] = df["bbox_x1"].astype(int)
    df["bbox_y1"] = df["bbox_y1"].astype(int)
    df["bbox_x2"] = df["bbox_x2"].astype(int)
    df["bbox_y2"] = df["bbox_y2"].astype(int)
    df["label"] = df["label"].astype(int)

    return df


def stratified_split_train_val(df, train_ratio=0.8):
    """
        split train set to `train` and `validation` with same distribution
    """
    df["train/val"] = None

    df_label = df[["label", "file_name"]].groupby(by="label").count()
    df_label.reset_index(inplace=True)

    with tqdm(df_label["label"].unique().tolist()) as t_labels:

        for label in t_labels:
            label_indecies = df[df["label"] == label].index.tolist()
            random.shuffle(label_indecies)

            num_label_data = len(label_indecies)
            num_label_train = math.ceil(num_label_data * train_ratio)

            train_label_indecies = label_indecies[:num_label_train]
            val_label_indecies = label_indecies[num_label_train:]

            df.loc[train_label_indecies, "train/val"] = "train"
            df.loc[val_label_indecies, "train/val"] = "val"

            t_labels.set_postfix(label=label)

    return df


def _main():
    # Train Dataset path
    train_annot_base_dir = "./mat_files"
    train_annot_name = "cars_train_annos.mat"
    train_image_base_dir = "./cars_train"
    train_ratio = 0.8
    train_saving_path = "."

    # Test Dataset path
    test_annot_base_dir = "./mat_files"
    test_annot_name = "cars_test_annos_withlabels.mat"
    test_image_base_dir = "./cars_test"
    test_saving_path = "."

    print("Train Data Processing: ")
    train_data = read_data_mat_file(
        annot_base_dir=train_annot_base_dir,
        annot_name=train_annot_name,
        image_base_dir=train_image_base_dir
    )

    # num_data = train_data.shape[0]
    # train_ratio = 0.8
    # num_train = int(train_ratio * num_data)
    # num_val = num_data - num_train

    # train_data = train_data.sample(frac=1)
    # train_data["train/val"] = ["train"] * num_train + ["val"] * num_val

    print("Spliting Training Data to 'Train' and 'Validation':")
    train_data = stratified_split_train_val(train_data, train_ratio=0.8)

    print("Test Data Processing: ")
    test_data = read_data_mat_file(
        annot_base_dir=test_annot_base_dir,
        annot_name=test_annot_name,
        image_base_dir=test_image_base_dir
    )

    train_data.to_csv(os.path.join(train_saving_path, "train.csv"))
    test_data.to_csv(os.path.join(test_saving_path, "test.csv"))


if __name__ == "__main__":
    _main()
