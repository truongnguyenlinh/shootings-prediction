import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Shootings:
    """Shootings dataset."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 1000)

    def __init__(self, csv, root_dir=""):
        """
        Initialize a Shootings dataset.
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

    def column_distribution(self):
        """
        Plot pie charts which display death based on column type.
        """
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
        """
        Plot histogram which displays deaths based on race.
        """
        config = {"kind": "bar", "ylabel": "Number of Deaths",
                  "xlabel": "Race"}
        self.df["race"].value_counts().plot(**config)
        plt.show()

    def death_distribution(self):
        """
        Plot histogram which displays deaths based on race and manner of death.
        """
        graph = sns.countplot(x="race", data=self.df, hue="manner_of_death")
        graph.set_xlabel("Race")
        graph.set_ylabel("Number of Deaths")
        graph.set_title("Number of Deaths Based on Race")

    def data_treatment(self):
        """
        Treat dataset by binning and converting date data type.
        """
        # Bin age groups into 4 groups
        bins = [0, 18, 45, 60, 100]
        groups = ["Teenager", "Adult", "Old", "Very Old"]
        self.df["Age_Group"] = pd.cut(self.df["age"], bins, labels=groups)

        # Convert dtype string to datetime
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df["year"] = pd.DatetimeIndex(self.df["date"]).year
        self.df["month"] = pd.DatetimeIndex(self.df["date"]).month
        self.df["month_year"] = pd.to_datetime(self.df["date"]).dt.to_period("M")

        print(self.df.head())

    def time_series(self):
        plt.style.use('bmh')
        self.df["month_year"] = self.df.month_year.astype(str)
        line_chart = self.df.groupby(["month_year"]).agg("count")["id"].to_frame(name="count").reset_index()

        plt.figure(figsize=(20, 8))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=15)
        plt.plot(line_chart["month_year"], line_chart["count"])
        plt.title("Killings by Month")
        plt.xticks(ticks=line_chart["month_year"], rotation=90)
        plt.show()


def main():
    url = "https://raw.githubusercontent.com/washingtonpost/data-police-shootings/master/fatal-police-shootings-data.csv"
    shootings_df = Shootings(url)
    shootings_df.column_distribution()
    shootings_df.death_distribution()
    shootings_df.data_treatment()
    shootings_df.time_series()


if __name__ == "__main__":
    main()
