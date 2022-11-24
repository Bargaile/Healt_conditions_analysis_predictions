import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from scipy.stats import loguniform
import statsmodels.stats.api as sms
import scipy.stats.distributions as dist

import optuna
from optuna.integration import LightGBMPruningCallback

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from imblearn.pipeline import Pipeline as ImPipeline
from imblearn.over_sampling import SMOTE 

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC, SVC, SVR

from sklearn.metrics import roc_curve, PrecisionRecallDisplay, RocCurveDisplay, precision_recall_curve
from sklearn.metrics import  plot_confusion_matrix, f1_score, log_loss, classification_report
from sklearn.model_selection import cross_validate, cross_val_score,  StratifiedKFold, KFold,  RandomizedSearchCV



# DATA CLEANING


def lower_case_all_values(df: pd.DataFrame) -> pd.DataFrame:
    """Takes all string values of the given pd.DataFrame and changes into lower case.

        params: pd.DataFrame: string values of which must be lowercased.
        return: the same pd.DataFrame, only with lowercased values.
        """

    df_new = df.applymap(lambda s: s.lower() if type(s) == str else s)
    return df_new


# VISUALIZATIONS


def plot_countplot(df: pd.DataFrame, feature1: str, feature2: str, title: str) -> None:
    """Takes as an input name of the pd.DataFrame and names of needed columns and plots a count plot.

        :param: df: the name of the pd.DataFrame to use;
                feature1: str - name of the columns to plot on X axis;
                feature2: str - name of the columns to plot as a 'hue' parameter (to do some cross counting);
                title: str -  final title (name) of the whole plot.
        """

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.countplot(x=feature1, hue=feature2, data=df, palette="viridis")

    if feature2 != None:
        ax.legend(loc="upper right", title=feature2)
        ax.bar_label(ax.containers[1])

    ax.set_title(title)
    ax.set(ylabel="")
    ax.set(xlabel=feature1)
    ax.bar_label(ax.containers[0])
    sns.despine(trim=True, left=True)
    plt.tight_layout()


def plot_line_plot_plotly(df: pd.DataFrame, x: str, y: str, z: str, title: str) -> None:
    """Takes as input name of the pd.DataFrame, names of columns ant plots a line plot.
        
        param: df:  the name of the pd.DataFrame to use;
               x: str - name of the column to plot on X axis;
               y: str - name of the column to plot on Y axis and to name the points on the line plot;
               z: str - name of the column to plot as hue (color) to the different lines;
               title: str - the title of the whole plot.
        """
    fig = px.line(df, x=x, y=y, color=z, text=y,)
    fig.update_traces(textposition="top left")
    fig.update_layout(legend_title="", title=title)
    fig.show()


def plot_stacked_barchart_plotly(
    df: pd.DataFrame, x: str, y: list, title: str, legend_title: str
) -> None:
    """Takes as an input name of the pd.Dataframe, needed columns, titles and plots a stacked bar chart of percentage
        
        param: df: the name of the pd.DataFrame to use;
               x: str - name of the column to plot on X axis;
               y: str - list of names (str) of the columns with percentage to plot on Y axis;
               title: str - title of the whole chart;
               legend_title: str - to rename the legend.
        """
    fig = px.bar(
        df,
        x=x,
        y=y,
        text_auto=True,
        color_discrete_map={y[0]: "#26828e", y[1]: "#35b779"},
        title=title,
    )
    fig.update_yaxes(title="")
    fig.update_xaxes(title="")
    fig.update_layout(legend_title=legend_title)
    fig.show()


def plot_box_stripplot(df: pd.DataFrame, x: str, y: str, title: str) -> None:
    """Takes as an input name of the pd.DataFrame, certain columns to plot on axis and plots boxplot+stripplot together.

        :param: df: the name of the pd.DataFrame to use;
               x: str - name of the column to plot on X axis;
               y: str - name of the column to plot on Y axis;
               title: str - title of the whole chart;
        """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_yscale("linear")
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    ax.set(xlabel="")
    sns.boxplot(x=x, y=y, data=df, palette="viridis")

    sns.stripplot(x=x, y=y, data=df, palette="GnBu", size=5, edgecolor="gray")

    sns.despine(trim=True, left=True)
    ax.set_title(title)


def plot_heatmap(df: pd.DataFrame, title: str) -> None:
    """Takes as an input pd.DataFrame and plot the heatmap with correlation coefficients between all features.

        :param: df: the name of the pd.DataFrame to use;
                title: str - title of the whole heatmap.
        """
    sns.set_theme(style="white")
    corr = df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    f, ax = plt.subplots(figsize=(10, 8))

    cmap = sns.dark_palette("seagreen", reverse=True, as_cmap=True)
    heatmap = sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot=True,
    )

    heatmap.set_title(
        title, fontdict={"fontsize": 16}, pad=12,
    )
    plt.xlabel("")
    plt.ylabel("")


def plot_categorical_countplots(df: pd.DataFrame, feature1: str) -> None:
    """ Takes as an input name of the pd.DataFrame (only categorical values) and one feature, which to use as a 'hue' factor.

        params: df: pd.DataFrame, consists only of categorical values in all columns.
                feature1: str - name of the last columns in the df, which will be used as a 'hue' factor in count plots.
        """
    plt.figure(figsize=(25, 35))
    for i in range(1, len(df.columns)):
        plt.subplot(int(len(df.columns) / 3) + 1, 3, i)
        ax = sns.countplot(
            x=df.columns[i - 1], hue=feature1, data=df, palette="viridis"
        )
        ax.xaxis.grid(True)
        ax.set_title(f"{df.columns[i-1]}")
        sns.despine(right=True, left=True)
    plt.show()


def multiple_violinplots(df: pd.DataFrame, feature1: str) -> None:
    """Plots multiple violin plots of 1 categorical feature and all other - numerical features.

        You need to make a new df only with needed columns: 1 categorical, other- numerical.
        params: df: pd.DatFrame, name of the data frame to use.
                Feature1: str, categorical feature in which to compare numerical values of all other features.
        """
    plt.figure(figsize=(15, 10))
    for i in range(1, len(df.columns)):
        plt.subplot(int(len(df.columns) / 3) + 1, 3, i)
        ax = sns.violinplot(
            x=feature1, y=df.columns[i - 1], data=df, palette="viridis", dodge=True
        )
        ax.xaxis.grid(True)
        ax.set_title(f"{df.columns[i-1]}")
        ax.set(xlabel=feature1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set(ylabel="")
        sns.despine(right=True, left=True)
    plt.show()


def plot_distplot(df: pd.DataFrame, feature1: str) -> None:
    """Plots the distplot of one given feature in the given data frame.
        params: df: pd.DataFrame to use;
                feature1: str - name of the column, which values to plot.
        """
    plt.figure(figsize=(5, 5))
    sns.distplot(df[feature1], color="#35b779")
    plt.title(f"Distribution of {feature1}")
    sns.despine(trim=True, left=True)


def plot_kde(df: pd.DataFrame, feature1: str, feature2: str) -> None:
    """ Plots KDE plot of two numerical features.
        params: df: usable pd.DataFrame;
                feature1: str - name of the numerical column, which values to plot;
                feature2: str - name of the categorical binary column of "yes" and "no", which values to use as a "hue";
        """
    plt.figure(figsize=(15, 5))
    plt.title(f"KDE Plot: {feature1} vs. {feature2}", fontsize=30, fontweight="bold")
    ax = sns.kdeplot(
        df[df[feature2] == "yes"][feature1],
        color="green",
        label=f"had {feature2}",
        lw=2,
        legend=True,
    )
    ax1 = sns.kdeplot(
        df[df[feature2] == "no"][feature1],
        color="blue",
        label=f"no {feature2}",
        lw=2,
        legend=True,
    )
    legend = ax.legend(loc="upper right")
    ax.yaxis.grid(True)
    sns.despine(right=True, left=True)
    plt.tight_layout()


# COUNTING


def make_crosstab_number(
    df: pd.DataFrame, feature1: str, feature2: str
) -> pd.DataFrame:
    """Takes as an input name of the pd.DataFrame and certain features to use, 
        outputs Pd.DataFrame with cross count of the values in these features.

        :param: df: the name of the pd.DataFrame to use;
                feature1: str - name of the first column which values to cross count;
                feature2: str - name of the second column which values to cross count;
        :return: pd.DataFrame with statistics of cross counted values from both used columns.
        """
    return pd.crosstab(df[feature1], df[feature2])


def make_crosstab_percent(
    df: pd.DataFrame, feature1: str, feature2: str
) -> pd.DataFrame:
    """Takes as an input name of the pd.DataFrame and certain features to use, 
        outputs Pd.DataFrame with cross count and turn into percent of the values in these features.

        :param: df: the name of the pd.DataFrame to use;
                feature1: str - name of the first column which values to cross count;
                feature2: str - name of the second column which values to cross count;
        :return: pd.DataFrame with percent of cross counted values from both used columns.
        """
    return pd.crosstab(df[feature1], df[feature2], normalize="index") * 100


# Inferential statistical analysis


class Diff_2_proportions:
    """
        Module Diff_2_proportions proceeds the needed table and all calculations
        for inferential statistical analysis of two proportions.

        Attribute:
        - df;
        - feature1: subgroups of people, proportion of which interests us (like gender);
        - feature2: feature by which we calculate the proportion and it's difference of the feature1 (like: stroke: yes);

        Methods of this class:
        - make_table;
        - total_proportion;
        - diff_of_proportions;
        - sample_size_needed;
        - std_error;
        - t_statistics;
        - p_value;
        - conf_interval_of_difference
        """

    def __init__(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        self._df = df
        self._feature1 = feature1
        self._feature2 = feature2

    def make_table(self) -> pd.DataFrame:
        """Creates a table - pd.DataFrame that helps to calculate the standard error."""

        self._table = self._df.groupby(self._feature1)[self._feature2].agg(
            [lambda z: np.mean(z == "yes"), "size"]
        )
        self._table.columns = ["proportion", "total_count"]
        print(f"Table of {self._feature2} per each group of {self._feature1}")
        return self._table

    def total_proportion(self) -> float:
        """Calculates the total proportion of feature2 together in all groups of feature1."""

        self._total_proportion = (self._df[self._feature2] == "yes").mean()
        print(f"Total proportion of {self._feature2} cases in the dataset:")
        return self._total_proportion

    def diff_of_proportions(self) -> float:
        """Calculates the difference in proportions"""

        self._diff = self._table.proportion.iloc[0] - self._table.proportion.iloc[1]
        print("Difference of two independent proportions:")
        return self._diff

    def sample_size_needed(self) -> None:
        """Calculates the required sample size to avoid p-hacking"""

        effect_size = sms.proportion_effectsize(
            self._table.proportion.iloc[0], self._table.proportion.iloc[1]
        )

        required_n = sms.NormalIndPower().solve_power(
            effect_size, power=0.8, alpha=0.05, ratio=1
        )
        required_n = ceil(required_n)
        print(f"Required sample size:{required_n}")

    def std_error(self):
        """Calculating standard error"""

        self._variance = self._total_proportion * (1 - self._total_proportion)
        self._standard_error = np.sqrt(
            self._variance
            * (
                1 / self._table.total_count.iloc[0]
                + 1 / self._table.total_count.iloc[1]
            )
        )
        return self._standard_error

    def t_statistics(self) -> float:
        """Calculate the test statistic"""

        hypothesized_estimate = 0
        self._test_stat = (self._diff - hypothesized_estimate) / self._standard_error
        print("Computed Test Statistic is:")
        return self._test_stat

    def p_value(self) -> float:
        """Calculate the  p-value, for 1 tail testing only"""

        self._pvalue = dist.norm.cdf(-np.abs(self._test_stat))
        print("Computed P-value is")
        return self._pvalue

    def conf_interval_of_difference(self):
        """Calculates the confidence interval of the difference of two proportions"""

        se_no = np.sqrt(
            self._table.proportion.iloc[0]
            * (1 - self._table.proportion.iloc[0])
            / self._table.total_count.iloc[0]
        )
        se_yes = np.sqrt(
            self._table.proportion.iloc[1]
            * (1 - self._table.proportion.iloc[1])
            / self._table.total_count.iloc[1]
        )

        se_diff = np.sqrt(se_no ** 2 + se_yes ** 2)

        self._lcb = self._diff - 2 * se_diff
        self._ucb = self._diff + 2 * se_diff
        print("CI in proportion of stroke cases among female and male:")
        return self._lcb, self._ucb


if __name__ == "__main__":
    Diff_2_proportions()


# MODELING


def base_line(
    X: pd.DataFrame, y: pd.DataFrame, preprocessor: np.array, resample: SMOTE
) -> pd.DataFrame:
    """
        Takes as an input X (all usable predictors) and y (outcome, dependent variable) pd.DataFrames.
        The function performs cross validation with different already selected models.
        Returns metrics and results of the models in pd.DataFrame format.

        :param: X - pd.DataFrame of predictors(independent features);
                y - pd.DataFrame of the outcome;
                preprocessor: ColumnTransformer with all needed scalers, transformers;
                resample: resampler from SMOTE() with different parameters.
        """

    balanced_accuracy = []
    roc_auc = []
    accuracy = []
    recall = []
    precision = []
    f1_score = []
    fit_time = []
    kfold = StratifiedKFold(n_splits=5)
    classifiers = [
        "Logistic regression",
        "Decision Tree",
        "Random Forest",
        "Linear SVC",
        "SVC",
        "KNN",
        "XGB classifier",
        "LGBM classifier",
    ]

    models = [
        LogisticRegression(),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100),
        LinearSVC(),
        SVC(),
        KNeighborsClassifier(),
        XGBClassifier(),
        LGBMClassifier(),
    ]

    for model in models:
        pipeline = ImPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("resample", resample),
                ("classifier", model),
            ]
        )
        result = cross_validate(
            pipeline,
            X,
            y,
            cv=kfold,
            scoring=(
                "balanced_accuracy",
                "accuracy",
                "f1_macro",
                "recall_macro",
                "precision_macro",
                "roc_auc",
            ),
        )
        fit_time.append(result["fit_time"].mean())
        balanced_accuracy.append(result["test_balanced_accuracy"].mean())
        accuracy.append(result["test_accuracy"].mean())
        recall.append(result["test_recall_macro"].mean())
        precision.append(result["test_precision_macro"].mean())
        f1_score.append(result["test_f1_macro"].mean())
        roc_auc.append(result["test_roc_auc"].mean())
    base_models = pd.DataFrame(
        {
            "Balanced accuracy": balanced_accuracy,
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "f1": f1_score,
            "Roc Auc": roc_auc,
            "Fit_time": fit_time,
        },
        index=classifiers,
    )
    base_models = base_models.style.background_gradient(cmap="Greens")
    return base_models


def LR_randomized_search(X: pd.DataFrame, y: pd.DataFrame, model: Pipeline):
    """Logistic regression hyper parameter searcher.
        
        Takes as an input X (independent variables(outcome) pd.DataFrame and a Pipeline with 
        preprocessors, transformers and certain model, fits the given data and searches for the best 
        hyper parameters.

        :param: X: pd.DataFrame with features;
                y: pd.DataFrame with outcome (dependent variable);
                model: sklearn.Pipeline with all needed transformers, preprocessors and chosen main model, in this
                example - Logistic Regression.
        """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    param_grid = [
        {
            "classifier__solver": ["newton-cg", "lbfgs", "liblinear"],
            "classifier__penalty": ["none", "l1", "l2", "elasticnet"],
            "classifier__C": loguniform(1e-5, 100),
            "classifier__class_weight": ["balanced", None],
            "classifier": [LogisticRegression()],
        },
    ]

    search = RandomizedSearchCV(
        model,
        param_grid,
        scoring="f1_macro",
        n_iter=20,
        n_jobs=-1,
        cv=cv,
        random_state=123,
    )

    result = search.fit(X, y)

    print(f"Best params:")
    print(result.best_params_)
    print("Best f1 score in randomized search:")
    print(result.best_score_)
    return result.best_params_


def plot_classifier_scores(
    model, X: pd.DataFrame, y: pd.DataFrame, predictions: np.array, target_labels: list
):
    """Plots the Confusion matrix and classification report from scikit-learn.
        
        :param: model - chosen model, modeled Pipeline from sklearn, on which data is trained.
                X - pd.DataFrame, X_train, X_validation, X_test data, which on to predict and plot the prediction 
                result.
                y - pd.DataFrame, the outcome, dependent variable: y_train. y_val, y_test, what to predict.
                predictions: y_hat, predictions from the model.
        """
    cmap = sns.dark_palette("seagreen", reverse=True, as_cmap=True)
    plot_confusion_matrix(model, X, y, cmap=cmap, display_labels=target_labels)
    plt.title("Confusion Matrix: ")
    plt.show()
    print(classification_report(y, predictions, target_names=target_labels))

    print()


def feature_names(
    module, numerical_features: list, binary_features: list, one_hot_features: list
) -> list:
    """
    Takes trained model.
    Extracts and returns feature name from preprocessor.
    """
    one_hot = list(
        module.named_steps["preprocessor"]
        .transformers_[1][1]
        .named_steps["encoder"]
        .get_feature_names(one_hot_features)
    )

    cat_all = numerical_features + one_hot + binary_features

    return cat_all


def plot_roc_auc_pr_auc(model: Pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """ Takes strained model and test set as an input, plots ROC AUC and PR AUC plots"""

    y_score = model.decision_function(X_test)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=model.classes_[1])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax1)
    prec, recall, _ = precision_recall_curve(
        y_test, y_score, pos_label=model.classes_[1]
    )
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot(ax=ax2)
    plt.show()


def SVC_randomized_search(X: pd.DataFrame, y: pd.DataFrame, model: Pipeline):
    """SVC hyper parameter searcher.
        
        Takes as an input X (independent variables(outcome) pd.DataFrame and a Pipeline with 
        preprocessors, transformers and certain model(SVC), fits the given data and searches for the best 
        hyper parameters.

        :param: X: pd.DataFrame with features;
                y: pd.DataFrame with outcome (dependent variable);
                model: sklearn.Pipeline with all needed transformers, preprocessors and chosen main model, in this
                example - SVC.
        """
    param_grid = [
        {
            "classifier__C": loguniform(1e-5, 100),
            "classifier__kernel": ["rbf"],
            "classifier__gamma": [10, 1, 0.1, 0.01],
            "classifier__class_weight": ["balanced", None],
            "classifier": [SVC()],
        },
    ]

    search = RandomizedSearchCV(
        model,
        param_grid,
        scoring="f1_macro",
        n_iter=20,
        n_jobs=-1,
        cv=5,
        random_state=123,
    )

    result = search.fit(X, y)

    print(f"Best params:")
    print(result.best_params_)
    print("Best f1 score in randomized search:")
    print(result.best_score_)
    return result.best_params_


def xgb_objective(
    trial, X: pd.DataFrame, y: pd.DataFrame, preprocessor, resample
) -> dict:
    """XGBoost hyper parameter searcher.

        Takes as an input X, y: pd.DataFrame and a Pipeline with 
        preprocessors, transformers and certain model, fits the given data and searches for the best 
        hyper parameters.

        :param: X: pd.DataFrame with features;
                y: pd.DataFrame with target (dependent variable);
                preprocessor: sklearn.Pipeline with all needed transformers, preprocessors;
                resample: type os used resampler from SMOTE().
        """

    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 100),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 100),
        "gama": trial.suggest_float("gama", 0, 20),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 100),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    cv_scores = np.empty(5)

    for idx, (train_idx, eval_idx) in enumerate(cv.split(X, y)):
        X_train, X_eval = X.iloc[train_idx], X.iloc[eval_idx]
        y_train, y_eval = y.iloc[train_idx], y.iloc[eval_idx]

        X_train = preprocessor.fit_transform(X_train)
        X_train, y_train = resample.fit_resample(X_train, y_train)
        X_eval = preprocessor.transform(X_eval)

        model = XGBClassifier(use_label_encoder=0, **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_eval, y_eval)],
            eval_metric="logloss",
            early_stopping_rounds=50,
            verbose=False,
        )
        preds = model.predict(X_eval)
        cv_scores[idx] = f1_score(y_eval, preds, average="macro")

    return np.mean(cv_scores)


def KNN_objective(
    trial,
    X: pd.DataFrame,
    y: pd.DataFrame,
    numeric_features: list,
    one_hot_features: list,
    binary_features: list,
    resample,
) -> float:
    """Takes as an input pd.DataFrames with features and outcome, gives the best score of f1 after training and
        cross validation.

        :params: trial : a process of evaluating an objective function;
                X: pd.DataFrame with independent features (predictors);
                y: pd.DataFrame with the outcome (what to predict);
                numeric_features: list - names of the features, which must be scaled with scaler (numerical columns);
        :returns: the score, this time - f1 - after fitting data with different hyper parameters to the model and
        cross validation.
        """
    # (a) List scalers to chose from
    scalers = trial.suggest_categorical("scalers", ["minmax", "standard", "robust"])

    # (b) Define your scalers
    if scalers == "minmax":
        scaler = MinMaxScaler()
    elif scalers == "standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    # (a) List all dimensionality reduction options
    dim_red = trial.suggest_categorical("dim_red", ["PCA", None])

    # (b) Define the PCA algorithm and its hyperparameters
    if dim_red == "PCA":
        # suggest an integer from 2 to 12 (as in total now I have 13 features)
        pca_n_components = trial.suggest_int("pca_n_components", 2, 10)
        dimen_red_algorithm = PCA(n_components=pca_n_components)
    # (c) No dimensionality reduction option
    else:
        dimen_red_algorithm = "passthrough"

    # -- Instantiate estimator model
    knn_n_neighbors = trial.suggest_int("knn_n_neighbors", 1, 20, 1)
    knn_metric = trial.suggest_categorical(
        "knn_metric", ["euclidean", "manhattan", "minkowski"]
    )
    knn_weights = trial.suggest_categorical("knn_weights", ["uniform", "distance"])

    estimator = KNeighborsClassifier(
        n_neighbors=knn_n_neighbors, metric=knn_metric, weights=knn_weights
    )

    binary_transformer = Pipeline(steps=[("encoder", OrdinalEncoder())])

    one_hot_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )

    # -- Make a pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", scaler, numeric_features),
            ("one_hot", one_hot_transformer, one_hot_features),
            ("binary", binary_transformer, binary_features),
        ],
        remainder="passthrough",
    )

    pipeline = ImPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("resample", resample),
            ("reduction", dimen_red_algorithm),
            ("estimator", estimator),
        ]
    )

    # -- Evaluate the score by cross-validation
    score = cross_val_score(pipeline, X, y, scoring="f1_macro")
    f1 = score.mean()
    return f1


def lgbm_objective(trial, X: pd.DataFrame, y: pd.DataFrame, preprocessor, resample) -> float:
    """ Takes as an input pd.DataFrames with features and outcome, gives the best score of f1 after training and
        cross validation.

        :params: trial : a process of evaluating an objective function;
                X: pd.DataFrame with independent features (predictors);
                y: pd.DataFrame with the outcome (what to predict);
                numeric_features: list - names of the features, which must be scaled with scaler (numerical columns);
        :returns: the score, this time - f1 - after fitting data with different hyper parameters to the model and
        cross validation.
    """
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train = preprocessor.fit_transform(X_train)
        X_train, y_train = resample.fit_resample(X_train, y_train)
        X_test = preprocessor.transform(X_test)

        model = LGBMClassifier(objective="binary", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="binary_logloss",
            early_stopping_rounds=100,
            callbacks=[LightGBMPruningCallback(trial, "binary_logloss")],
        )
        preds = model.predict_proba(X_test)
        cv_scores[idx] = log_loss(y_test, preds)

        return np.mean(cv_scores)


def base_line_regression(
    X: pd.DataFrame, y: pd.DataFrame, preprocessor: np.array
) -> pd.DataFrame:
    """
        Takes as an input X (all usable predictors) and y (outcome, dependent variable) pd.DataFrames.
        The the function performs cross validation with different already selected models.
        Returns metrics and results of the models in pd.DataFrame format.

        :param: X - pd.DataFrame of predictors(independent features);
                y - pd.DataFrame of the outcome;
                preprocessor: ColumnTransformer with all needed scalers, transformers.
        """
    mae = []
    mse = []
    rmse = []
    r2 = []
    fit_time = []
    regressors = [
        "Linear Regression",
        "Elastic Net",
        "Ridge Regression",
        "Lasso regression",
        "Decision Tree Regressor",
        "Random Forest Regressor",
        "SVR",
        "KNN Regressor",
        "XGB Regressor",
        "LGBM Regressor",
    ]

    models = [
        LinearRegression(),
        ElasticNet(),
        Ridge(),
        Lasso(),
        DecisionTreeRegressor(),
        RandomForestRegressor(n_estimators=100),
        SVR(),
        KNeighborsRegressor(),
        XGBRegressor(),
        LGBMRegressor(),
    ]
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    for model in models:
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", model),]
        )
        result = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=(
                "neg_mean_absolute_error",
                "neg_mean_squared_error",
                "neg_root_mean_squared_error",
                "r2",
            ),
        )
        fit_time.append(result["fit_time"].mean())
        mae.append(result["test_neg_mean_absolute_error"].mean() * -1)
        mse.append(result["test_neg_mean_squared_error"].mean() * -1)
        rmse.append(result["test_neg_root_mean_squared_error"].mean() * -1)
        r2.append(result["test_r2"].mean())
    base_models = pd.DataFrame(
        {
            "Mean_absolute_error": mae,
            "Mean_squared_error": mse,
            "Root_mean_squared_error": rmse,
            "R2": r2,
            "Fit time": fit_time,
        },
        index=regressors,
    )
    return base_models


def ridge_reg_objective(
    trial, X: pd.DataFrame, y: pd.DataFrame, preprocessor: ColumnTransformer
) -> float:
    """Takes as an input pd.DataFrames with features and outcome, gives the best scores after training and
        cross validation.

        :params: trial : a process of evaluating an objective function;
                X: pd.DataFrame with independent features (predictors);
                y: pd.DataFrame with the outcome (what to predict);
                preprocessor: sklearn ColumnsTransformer woth all needed preprocessors.
        :returns: the score: neg_root_mean_squared_error, multiplied by -1
        """
    alpha = trial.suggest_float("alpha", 0, 10)
    intercept = trial.suggest_categorical("fit_intercept", [True, False])
    tol = trial.suggest_float("tol", 0.001, 0.01, log=True)
    solver = trial.suggest_categorical(
        "solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
    )

    cv = KFold(n_splits=5, random_state=123, shuffle=True)

    model = Ridge(alpha=alpha, fit_intercept=intercept, tol=tol, solver=solver)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", model),])

    score = cross_val_score(
        pipeline, X, y, cv=cv, scoring="neg_root_mean_squared_error"
    ).mean()
    return score * -1


def base_line_multi(
    X: pd.DataFrame, y: np.array, preprocessor: np.array, resample: np.array
) -> pd.DataFrame:
    """
        Takes as an input X (all usable predictors) and y (outcome, dependent variable: multi-label) 
        pd.DataFrames.
        The the function performs cross validation with different already selected models.
        Returns metrics and results of the models in pd.DataFrame format.

        :param: X - pd.DataFrame of predictors(independent features);
                y - pd.DataFrame of the outcome;
                preprocessor: ColumnTransformer with all needed scalers, transformers;
                resample: SMOTE resampler for multiclass.
        """
    balanced_accuracy = []
    accuracy = []
    recall = []
    precision = []
    f1_score = []
    fit_time = []
    classifiers = [
        "Logistic regression",
        "Decision Tree",
        "Random Forest",
        "Linear SVC",
        "SVC",
        "KNN",
        "XGB classifier",
        "LGBM classifier",
    ]

    models = [
        LogisticRegression(multi_class="multinomial"),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100),
        LinearSVC(multi_class="crammer_singer"),
        SVC(),
        KNeighborsClassifier(),
        XGBClassifier(objective="multi:softmax"),
        LGBMClassifier(objective="multiclass"),
    ]

    for model in models:
        pipeline = ImPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("resample", resample),
                ("classifier", model),
            ]
        )
        result = cross_validate(
            pipeline,
            X,
            y,
            cv=3,
            scoring=(
                "balanced_accuracy",
                "accuracy",
                "f1_macro",
                "recall_macro",
                "precision_macro",
            ),
        )
        fit_time.append(result["fit_time"].mean())
        balanced_accuracy.append(result["test_balanced_accuracy"].mean())
        accuracy.append(result["test_accuracy"].mean())
        recall.append(result["test_recall_macro"].mean())
        precision.append(result["test_precision_macro"].mean())
        f1_score.append(result["test_f1_macro"].mean())
    base_models = pd.DataFrame(
        {
            "Balanced accuracy": balanced_accuracy,
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "f1": f1_score,
            "Fit_time": fit_time,
        },
        index=classifiers,
    )
    base_models = base_models.style.background_gradient(cmap="Greens")
    return base_models


def lgbm_multi_objective(trial, X, y, preprocessor, remaple) -> dict:
    """
    Takes x and y and dataframes
    Performs model training.
    Returns best parameters for particular multiclass model.
    """

    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        model = LGBMClassifier(objective="multiclass", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="multi_logloss",
            early_stopping_rounds=100,
            callbacks=[
                optuna.integration.LightGBMPruningCallback(trial, "multi_logloss")
            ],
        )
        preds = model.predict_proba(X_test)
        cv_scores[idx] = log_loss(y_test, preds)

        return np.mean(cv_scores)


def xgb_multioutput_randomized_search(
    model: Pipeline, X: pd.DataFrame, y: pd.DataFrame
):
    """XGBclassifier wrapped in MultioutputClassifier  hyper parameter searcher.
        
        Takes as an input X (independent variables(outcome) pd.DataFrame and a Pipeline with 
        preprocessors, transformers and certain model(XGB), fits the given data and searches for the best 
        hyper parameters.

        :param: X: pd.DataFrame with features;
                y: pd.DataFrame with outcome (dependent variable);
                model: sklearn.Pipeline with all needed transformers, preprocessors and chosen main model, in this
                example - MultioutputClassifier(XGBClassifier()).
        """
    xgbparams = {
        "classifier__estimator__n_estimators": range(0, 1000, 10),
        "classifier__estimator__learning_rate": [0.01, 0.05, 0.10, 0.15, 0.20],
        "classifier__estimator__num_leaves": range(20, 3000, 20),
        "classifier__estimator__max_depth": [3, 4, 5, 6, 8, 10, 12],
        "classifier__estimator__reg_alpha": loguniform(1e-3, 100),
        "classifier__estimator__reg_lambda": loguniform(1e-3, 100),
        "classifier__estimator__min_child_weight": [1, 3, 5, 7, 10],
        "classifier__estimator__gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
        "classifier__estimator__scale_pos_weight": range(1, 100, 1),
        "classifier__estimator__colsample_bytree": [0.3, 0.4, 0.5, 0.7],
    }
    search = RandomizedSearchCV(
        model, param_distributions=xgbparams, n_iter=5, n_jobs=-1, cv=5, verbose=3
    )
    result = search.fit(X, y)
    return result.best_params_


def forest_randomized_search_multioutput(
    X: pd.DataFrame, y: pd.DataFrame, model: Pipeline
):
    """Random Forest in the Multioutputlassifier hyper parameter searcher.
        
        Takes as an input X (independent variables(outcome) pd.DataFrame and a Pipeline with 
        preprocessors, transformers and certain model(SVC), fits the given data and searches for the best 
        hyper parameters.

        :param: X: pd.DataFrame with features;
                y: pd.DataFrame with outcome (dependent variable);
                model: sklearn.Pipeline with all needed transformers, preprocessors and chosen main model, in this
                example - SVC.
        """
    param_grid = [
        {
            "classifier__estimator__n_estimators": [
                int(x) for x in np.linspace(start=200, stop=2000, num=10)
            ],
            "classifier__estimator__max_depth": [
                int(x) for x in np.linspace(2, X.shape[1], num=1)
            ],
            "classifier__estimator__min_samples_split": [
                int(x) for x in np.linspace(2, X.shape[1], num=1)
            ],
            "classifier__estimator__min_samples_leaf": [1, 2, 4, 6, 8, 10],
            "classifier__estimator__max_features": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "classifier__estimator__class_weight": [
                "balanced",
                "balanced_subsample",
                None,
            ],
            "classifier": [MultiOutputClassifier(RandomForestClassifier())],
        },
    ]

    search = RandomizedSearchCV(
        model, param_grid, n_iter=20, n_jobs=-1, cv=5, random_state=123
    )

    result = search.fit(X, y)

    print(f"Best params:")
    print(result.best_params_)
    print("Best accuracy score in randomized search:")
    print(result.best_score_)
    return result.best_params_