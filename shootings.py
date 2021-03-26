# install geoplot via 'conda install geoplot -c conda-forge'
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import pmdarima as pm
from sklearn import metrics
import mapclassify as mc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.arima_model import ARIMA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


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

        # Convert gender into binary values
        self.df["gender_bin"] = np.where(self.df["gender"] == "M", 1, 0)

        # Convert signs_of_mental_illness column into binary values
        self.df["signs_of_mental_illness_bin"] =\
            np.where(self.df["signs_of_mental_illness"] == True, 1, 0)

        # Convert body_camera column into binary values
        self.df["body_camera_bin"] =\
            np.where(self.df["body_camera"] == True, 1, 0)

        # Convert is_geocoding_exact column into binary values
        self.df["is_geocoding_exact_bin"] =\
            np.where(self.df["is_geocoding_exact"] == True, 1, 0)

        self.df["manner_of_death_bin"] = \
            np.where(self.df["manner_of_death"] == "shot", 1, 0)

        # Label encode threat level
        # attack:0, other:1, undetermined:2
        self.df["threat_level"] = self.df["threat_level"].astype("category")
        self.df["threat_level_cat"] = self.df["threat_level"].cat.codes

        # Label encode race column
        # A:0, B:1, H:2, N:3, O:4, W:5, None:6
        self.df["race"] = self.df["race"].astype("category")
        self.df["race_cat"] = self.df["race"].cat.codes

        self.df["total_population"] = np.select(conditions, numbers, default=0)

    def time_series(self):
        """
        Plot data in a time-series per month and year.
        """
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
        """
        Plot a geomap to highlight states which have highest number of deaths
        """
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
        """
        Plot a histogram to display deaths per race population
        """
        races = ["A", "W", "H", "B", "N", "O"]
        killed_per_race = []
        prop_killed_per_race = []

        for i in races:
            i_killings = self.df["race"].loc[(self.df["race"] == i)].count()
            killed_per_race.append(i_killings)

        killed_dict = {
            "A": 14674252.0,
            "W": 223553265.0,
            "H": 50477594.0,
            "B": 38929319.0,
            "N": 2932248.0,
            "O": 22579629.0
        }
        killed_order = {
            "A": 0,
            "W": 1,
            "H": 2,
            "B": 3,
            "N": 4,
            "O": 5
        }
        for i in races:
            prop_i_killed = killed_per_race[killed_order[i]] / killed_dict[i]
            prop_killed_per_race.append(prop_i_killed)

        plt.figure(figsize=(14,6))
        plt.title("People killed as a proportion of their respective race",
                  fontsize=17)
        sns.barplot(x=races, y=prop_killed_per_race)
        plt.show()

    def arima_prediction(self):
        """
        Complete an ARIMA prediction]
        """
        data_list = ['A', 'W', 'H', 'B', 'N','O']
        for txt in data_list:
            data = self.df[self.df['race'] == txt]
            # data = self.df
            data = data.groupby("month_year").count().reset_index()
            data['count'] = data.groupby('month_year')['name'].transform('sum')
            print(data.head())
            data = data[['month_year', 'count']]
            for i in range(1, len(data)):
                data.loc[i, 'count'] += data.loc[i - 1, 'count']

            data['month_year'] = data['month_year'].dt.to_timestamp('s').dt.strftime('%Y-%m')
            data['month_year'] = pd.to_datetime(data['month_year'], format="%Y-%m")
            data = data.set_index("month_year")
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
                             color="k", alpha=.15)
            plt.plot(fitted_series, color="black")
        plt.title("ARIMA - Final Forecast of Police shooting deaths by race")
        plt.legend(loc="upper left")
        plt.show()

    def ols_model(self):
        """
        Perform OLS regression on input against race
        """
        X = self.df[["id", "age", "gender_bin", "threat_level_cat",
                     "signs_of_mental_illness_bin", "manner_of_death_bin",
                     "body_camera_bin", "is_geocoding_exact_bin"]]
        y = self.df[["race_cat"]]

        X = sm.add_constant(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        model = sm.OLS(y_train, X_train).fit()
        predictions = model.predict(X_test)
        print(model.summary())
        print('Root Mean Squared Error:',
              np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    def back_test(self):
        """
        Perform back testing.
        """
        total_days_ahead = 1000

        X = self.df[["id", "age", "year", "month", "gender_bin",
                     "signs_of_mental_illness_bin", "body_camera_bin",
                     "is_geocoding_exact_bin", "manner_of_death_bin",
                     "threat_level_cat"]]
        y = self.df[['race_cat']]

        xscaler = MinMaxScaler()
        scaledX = xscaler.fit_transform(X)

        sel_chi2 = SelectKBest(chi2, k=3)  # Select 10 best features
        X_train_chi2 = sel_chi2.fit_transform(scaledX, y)
        selections = sel_chi2.get_support()

        # Build best feature list.
        keys = X.keys()
        chosen_features = []
        for i in range(0, len(selections)):
            if selections[i]:
                chosen_features.append(keys[i])

        subX = X[chosen_features]
        xscaler2 = MinMaxScaler()
        subXScaled = xscaler.fit_transform(subX)

        lenX = len(X)
        days_left = total_days_ahead

        # Get day ahead predictions. Each prediction uses the latest data.
        day_ahead_predictions = []
        while days_left >= 1:
            split_row = lenX - days_left

            # Split dataframe so training set has latest data.
            trainX = subX.iloc[:split_row, :]
            testX = subX.iloc[split_row:, :]
            trainY = y.iloc[:split_row]
            days_left -= 1

            # Scale train and test.
            scaledTrainX = xscaler.transform(trainX)
            scaledTestX = xscaler.transform(testX)

            # Build model with latest data and make predictions.
            model = LogisticRegression(solver='liblinear')
            model.fit(scaledTrainX, trainY)

            predictions = model.predict(scaledTestX)
            prediction = predictions[0]
            # Extract next-day prediction and add to list.
            day_ahead_predictions.append(prediction)

        print("Day ahead predictions: ")
        print(day_ahead_predictions)
        print("Actual values: ")
        split_row = len(y) - total_days_ahead
        testY = y.iloc[split_row:]
        print(testY)


def main():
    url = "https://raw.githubusercontent.com/washingtonpost/data-police-shootings/master/fatal-police-shootings-data.csv"
    shootings_df = Shootings(url)
    # shootings_df.usa_heatmap()
    # shootings_df = Shootings(url)
    # shootings_df.column_distribution()
    # shootings_df.death_distribution()
    # shootings_df.race_death_proportion()
    shootings_df.data_treatment()
    # shootings_df.time_series()
    # shootings_df.ols_model()

    # shootings_df.arima_prediction()
    shootings_df.back_test()


if __name__ == "__main__":
    main()


