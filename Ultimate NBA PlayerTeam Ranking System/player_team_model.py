"""
Isaac Xu & Spencer James Knapp
section AG
This program would create ML model for
player and team data and return decision
trees of ML models
"""

import graphviz
from sklearn.tree import export_graphviz
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


def players_ranking_model(player_data, file_name):
    """
    This function would take in a dataframe of players' score
    and make a machine learning model based on the input dataframe.
    This function would plot a plot of decision tree based on the
    testing of the ML model. The given file name would be the name
    of the returned plot
    """
    player_data = player_data[['total_score', 'avg_score', 'Rank']]

    # determin features and labels
    features = player_data.loc[:, (player_data.columns != 'Rank')]
    labels = player_data['Rank']

    # seperate training and testing features and labels
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2, random_state=2)

    # creating a model and train it
    model = DecisionTreeRegressor()
    model.fit(features_train, labels_train)

    # report training and testing result
    train_prediction = model.predict(features_train)
    print('player model testing error:',
          mean_squared_error(labels_train, train_prediction))
    test_prediction = model.predict(features_test)
    print('player model predicting error:',
          mean_squared_error(labels_test, test_prediction))

    # plot the decision tree of model when its testing
    plot_tree(model, features_test, labels_test, file_name)

    # since there hundreds of player in the df, we are creating
    # a short model to plot a tree that would be clearly demonstrate
    # in our final report
    short_model = DecisionTreeRegressor(max_depth=3)
    short_model.fit(features_train, labels_train)
    plot_tree(short_model, features_test, labels_test, file_name)


def teams_ranking_model(teams_data, file_name):
    """
    This function would take in a dataframe of teams' score
    and make a machine learning model based on the input dataframe.
    This function would plot a plot of decision tree based on the
    testing of the ML model. The given file name would be the name
    of the returned plot
    """
    # determin features and labels
    features = teams_data.loc[:, (teams_data.columns != 'Rank_Overall')
                              & (teams_data.columns != 'TEAM')]
    labels = teams_data['Rank_Overall']

    # seperate training and testing features and labels
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)

    # creating a model and train it
    model = DecisionTreeRegressor()
    model.fit(features_train, labels_train)

    # report training and testing result
    train_prediction = model.predict(features_train)
    print('team model testing error:',
          mean_squared_error(labels_train, train_prediction))
    test_prediction = model.predict(features_test)
    print('team model predicting error:',
          mean_squared_error(labels_test, test_prediction))

    # plot the decision tree of model when its testing
    plot_tree(model, features_test, labels_test, file_name)


def plot_tree(model, features, labels, file_name):
    """
    This function would take in a ML model and its features and label
    and plot a decision tree of that ML model and save it as a given
    file_name
    """
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=features.columns,
                               class_names=labels.unique(),
                               impurity=False,
                               filled=True, rounded=True,
                               special_characters=True)
    graphviz.Source(dot_data).render(file_name + '.gv', format='png')


def main():
    player_data = pd.read_csv('../data_upload/player_rank.csv')
    team_data = pd.read_csv('../data_upload/team_rank.csv')
    players_ranking_model(player_data, 'tree_of_players')
    teams_ranking_model(team_data, 'tree_of_teams')


if __name__ == '__main__':
    main()
