from torch.utils.data import Dataset, DataLoader
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class VideoGame:
    """VideoGame dataset."""
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)

    def __init__(self, csv, root_dir=""):
        """
        Initialize a VideoGame dataset.
        :param csv: a string
        :param root_dir: a string
        """
        df = pd.read_csv(csv,
                         sep=",",
                         skipinitialspace=True,
                         header=0,
                         names=("Rank", "Name", "Platform", "Year",
                                "Genre", "Publisher", "NA_Sales",
                                "EU_Sales", "JP_Sales", "Other_Sales",
                                "Global_Sales"))
        self.df = df.dropna()
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __str__(self):
        print(self.df)


def main():
    video_game_df = VideoGame("vgsales.csv")
    print(video_game_df)


if __name__ == '__main__':
    main()
