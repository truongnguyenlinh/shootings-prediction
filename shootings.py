import os
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import pie, axis, show

from statsmodels.tsa.seasonal import seasonal_decompose


class Shootings:
    """Shootings dataset."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 1000)

    def __init__(self, csv, root_dir=""):
        """
        Initialize a VideoGame dataset.
        :param csv: a string
        :param root_dir: a string
        """
        self.df = pd.read_csv(csv,
                              sep=",",
                              skipinitialspace=True,
                              header=0)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __str__(self):
        print(self.df)

    def gender_distribution(self):
        plt.style.use("ggplot")
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        config = {"kind": "pie", "autopct": "%1.1f%%", "startangle": 120}

        self.df["gender"].value_counts().plot(**config, ax=axes[0, 0])
        self.df["manner_of_death"].value_counts().plot(**config, ax=axes[0, 1])
        self.df["signs_of_mental_illness"].value_counts().plot(**config, ax=axes[0, 2])
        self.df["threat_level"].value_counts().plot(**config, ax=axes[1, 0])
        self.df["flee"].value_counts().plot(**config, ax=axes[1, 1])
        self.df["body_camera"].value_counts().plot(**config, ax=axes[1, 2])
        plt.show()

    def race_distribution(self):
        config = {"kind": "bar", "ylabel": "Number of Deaths",
                  "xlabel": "Race"}
        self.df["race"].value_counts().plot(**config)
        plt.show()

    def death_distribution(self):
        graph = sns.countplot(x="Race", data=self.df, hue="manner_of_death")
        graph.set_xlabel("Race")
        graph.set_ylabel("")

    def data_treatment(self):
        # Convert dtype string to datetime
        self.df["date"] = pd.to_datetime(self.df["date"], format="%Y-%m-%d")
        self.df = self.df.set_index("date")

        self.df['year'] = self.df.index.year
        self.df['month'] = self.df.index.month
        self.df['day'] = self.df.index.day
        print(self.df.head())


def main():
    url = "https://raw.githubusercontent.com/washingtonpost/data-police-shootings/master/fatal-police-shootings-data.csv"
    shootings_df = Shootings(url)
    # print(shootings_df.df.head())
    # shootings_df.race_distribution()
    shootings_df.data_treatment()


if __name__ == "__main__":
    main()
