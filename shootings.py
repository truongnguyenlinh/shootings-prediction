import os
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import pie, axis, show
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import mapclassify as mc
# install geoplot via 'conda install geoplot -c conda-forge'

from statsmodels.tsa.seasonal import seasonal_decompose


class Shootings:
    """Shootings dataset."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 1000)

    def __init__(self, csv, root_dir=""):
        """
        Initialize a Shooting dataset.
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

    def usa_heatmap(self):
        states_df = self.df.state
        test = pd.DataFrame(states_df.value_counts(normalize=True).mul(100).round(1).astype(float))
        test = test.reset_index()
        test.columns = ["state", "percentage"]
        usa = gpd.read_file("map/states.shx")
        usa['percent'] = usa['STATE_ABBR'].map(lambda state: test.query("state == @state").iloc[0]['percentage'])
        scheme = mc.Quantiles(usa['percent'], k=5)

        ax = gplt.cartogram(
            usa,
            scale='percent', limits=(0.95, 1),
            projection=gcrs.AlbersEqualArea(central_longitude=-100, central_latitude=50),
            hue='percent', cmap='Blues', scheme=scheme,
            linewidth=0.9,
            legend=True, legend_kwargs={'loc': 0}, legend_var='hue',
            figsize=(8, 12)
        )
        gplt.polyplot(usa, facecolor='lightgray', edgecolor='None', ax=ax)

        plt.title("Shooting by state in percentage")
        plt.show()

def main():
    url = "https://raw.githubusercontent.com/washingtonpost/data-police-shootings/master/fatal-police-shootings-data.csv"
    shootings_df = Shootings(url)
    # print(shootings_df.df.head())
    # shootings_df.race_distribution()
    shootings_df.usa_heatmap()

if __name__ == "__main__":
    main()


