import os
import random
import torch
import torchvision
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class StanfordCar196(Dataset):
    """
    A dataloader for Classifier Network
    """

    def __init__(self, img_base_dir, annot_df_path, transformation=None, num_categories=None, mode="test"):
        """
        Parameters :
            - img_base_dir: path to base directory of images
            - df_path: path to the dataframe of annotations
            - transformation: torchvision.transforms
            - mode: mode of processing -> {`train`,`val`,`test`}
        """
        assert mode in ["train", "val",
                        "test"], "`mode` of dataloader must be one of this items: {`train`,`val`,`test`}"

        super().__init__()

        self.img_base_dir = img_base_dir
        self.transformation = transformation

        self.df = pd.read_csv(annot_df_path)
        self.df.drop(columns="Unnamed: 0", inplace=True)
        self.df = self.df[self.df["gray_scale"] == False]

        if mode in ["train", "val"]:
            self.df = self.df[self.df["train/val"] == mode]

        self.df["path"] = self.df["file_name"].apply(
            lambda x: os.path.join(self.img_base_dir, x))

        self.images = list(zip(self.df["path"], self.df["label"]))
        random.shuffle(self.images)

        if num_categories is None:
            self.num_categories = len(self.df["id"].unique())
        else:
            self.num_categories = num_categories

    def __getitem__(self, index):
        """
        In this function, an image and its one-hot label is returned.
        """

        img_path, img_cat = self.images[index]
        img_cat -= 1

        x = Image.open(img_path)
        if self.transformation is not None:
            x = self.transformation(x)

        # y = torch.zeros(self.num_categories)
        # y[img_cat] = 1
        # y = torch.tensor([img_cat])
        y = img_cat

        # print("path: {}, cat: {}, x.shape: {}, y.shape: {}".format(
        #     img_path.split("/")[-1], img_cat, x.shape, y.shape))

        return x, y

    def __len__(self):
        """
        `len(.)` function return number of data in dataset
        """
        return len(self.images)
