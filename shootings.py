import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import pie, axis, show
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import pmdarima as pm
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
        groups = ["Teenager", "Adult", "Older Adult", "Senior"]
        self.df["Age_Group"] = pd.cut(self.df["age"], bins, labels=groups)

        # Convert dtype string to datetime
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df["year"] = pd.DatetimeIndex(self.df["date"]).year
        self.df["month"] = pd.DatetimeIndex(self.df["date"]).month
        self.df["month_year"] = pd.to_datetime(self.df["date"]).dt.to_period("M")
        
        # Fill age column with mean, and drop rows where race is missing
        self.df["age"].fillna(value=self.df["age"].mean(), inplace=True)
        self.df["age"] = self.df["age"].astype(int)
        self.df.dropna(subset=["race"], inplace=True)

        # Add total_population column with data corresponding to race
        conditions = [self.df["race"] == "A", self.df["race"] == "W",
                      self.df["race"] == "H", self.df["race"] == "B",
                      self.df["race"] == "N", self.df["race"] == "O"]
        numbers = [14674252, 223553265, 50477594, 38929319, 2932248, 22579629]

        self.df["total_population"] = np.select(conditions, numbers, default=0)
        # print(self.df.head())

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

    def usa_heatmap(self):
        states_df = self.df.state
        test = pd.DataFrame(states_df.value_counts(normalize=True).mul(100).round(1).astype(float))
        test = test.reset_index()
        test.columns = ["state", "percentage"]
        usa = gpd.read_file("map/states.shx")
        usa['percent'] = usa['STATE_ABBR'].map(lambda state: test.query("state == @state").iloc[0]['percentage'])
        scheme = mc.FisherJenks(usa['percent'], k=7)
        print(scheme)
        ax = gplt.cartogram(
            usa,
            scale='percent', limits=(0.95,1),
            projection=gcrs.AlbersEqualArea(central_longitude=-100, central_latitude=39.5),
            hue='percent', cmap='Greens', scheme=scheme,
            linewidth=0.5,
            legend=True, legend_kwargs={'loc': 0},
            figsize=(8, 12)
        )
        gplt.polyplot(usa, facecolor='lightgray', edgecolor='None', ax=ax)

        plt.title("Shooting by state in percentage")
        plt.show()

    def race_death_proportion(self):
        races = ["A", "W", "H", "B", "N", "O"]
        killed_per_race = []
        prop_killed_per_race = []

        for i in races:
            i_killings = self.df["race"].loc[(self.df["race"] == i)].count()
            killed_per_race.append(i_killings)

        print(killed_per_race)
        for i in races:

            if i == "A":
                prop_i_killed = killed_per_race[0] / 14674252.0
                print(prop_i_killed)
            elif i == "W":
                prop_i_killed = killed_per_race[1] / 223553265.0
                print(prop_i_killed)
            elif i == "H":
                prop_i_killed = killed_per_race[2] / 50477594.0
                print(prop_i_killed)
            elif i == "B":
                prop_i_killed = killed_per_race[3] / 38929319.0
                print(prop_i_killed)
            elif i == "N":
                prop_i_killed = killed_per_race[4] / 2932248.0
                print(prop_i_killed)
            else:
                prop_i_killed = killed_per_race[5] / 22579629.0
                print(prop_i_killed)
            prop_killed_per_race.append(prop_i_killed)

        plt.figure(figsize=(14,6))
        plt.title("People killed as a proportion of their respective race",
                  fontsize=17)
        sns.barplot(x=races, y=prop_killed_per_race)
        plt.show()

    def add_prev(self, previous, this_count):
        return previous + this_count

    def arima_prediction(self):

        data_list = ["Teenager", "Adult", "Older Adult", "Senior"]
        data_list = ['A', 'W', 'H', 'B', 'N','O']
        for txt in data_list:
            data = self.df[self.df['race'] == txt]
            data = data.groupby("date").count().reset_index()
            data['count'] = data.groupby('date')['id'].transform('sum')
            data = data[['date','count']]
            for i in range(1, len(data)):
                data.loc[i, 'count'] += data.loc[i-1, 'count']

            print(data)
            data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%d")
            data = data.set_index("date")

            smodel = pm.auto_arima(data, start_p=1, start_q=1,
                             test='adf',
                             max_p=3, max_q=3, m=12,
                             start_P=0, seasonal=False,
                             d=None, D=1, trace=True,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True)


            n_periods = 12
            fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
            index_of_fc = pd.date_range(data.index[-1], periods=n_periods, freq='MS')

            # make series for plotting purpose
            fitted_series = pd.Series(fitted, index=index_of_fc)
            lower_series = pd.Series(confint[:, 0], index=index_of_fc)
            upper_series = pd.Series(confint[:, 1], index=index_of_fc)

            plt.plot(data, label=txt)
            plt.fill_between(lower_series.index,
                             lower_series,
                             upper_series,
                              alpha=.15)
            plt.plot(fitted_series, color="black")
        plt.title("ARIMA - Final Forecast of Police shooting deaths by race")
        plt.legend(loc="upper left")
        plt.show()


def main():
    url = "https://raw.githubusercontent.com/washingtonpost/data-police-shootings/master/fatal-police-shootings-data.csv"
    shootings_df = Shootings(url)
    # shootings_df.usa_heatmap()
    # shootings_df = Shootings(url)
    # shootings_df.column_distribution()
    # shootings_df.death_distribution()
    shootings_df.race_death_proportion()
    shootings_df.data_treatment()
    # shootings_df.time_series()

    shootings_df.arima_prediction()


if __name__ == "__main__":
    main()


