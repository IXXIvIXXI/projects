"""
Isaac Xu & Spencer James Knapp
section AG
This program would calculate player scores and help rank each player based on
their player score and merge csv files of players stats of different years and
team roster together and calulate each team score based on players in this
team and rank each team based on their scores
"""
import pandas as pd
import os
from get_csv import GetPlayerCsv


def player_scores(players, w1=.2, w2=.2, w3=.2, w4=.2, w5=.2):
    """
    Takes in dataframe of players and weights. Default value of weights is
    .2 unless set by user (weights need to sum to 1 - may need a control
    on this). Returns dataframe with players and computed scores,
    sorted highest score to lowest score.
    """
    players['Score'] = ((players['plus_minus'] * w1) +
                        (players['eFG_avg'] * w2) +
                        (players['winning_perct'] * w3) +
                        (players['total_games'] * w4) +
                        (players['pts'] * w5))
    players['year_count'] = 1

    for i in range(len(players)):
        if players.loc[i, 'total_games'] == 0:
            players.loc[i, 'year_count'] = 0

    return players.sort_values(by=['Score'], ascending=False)


def dataset_mergin(year1, year2, year3):
    """
    This function would take in three dataframes about players stats and
    scores in the most recent three years and merge them together.
    """
    combined = year1.merge(year2, left_on='player_name',
                           right_on='player_name', how='outer')
    combined2 = year3.merge(combined, left_on='player_name',
                            right_on='player_name', how='outer')
    return combined2


def get_average_score(combined, w1=.33, w2=.33, w3=.33):
    """
    This function would take in a combined dataframe of player's stats
    and scores in the most recent three years and calculate their average
    scores based on their score in each year. The default weight would be
    0.333 for each
    """
    combined['total_year_count'] = (combined['year_count'] +
                                    combined['year_count_x'] +
                                    combined['year_count_y'])
    combined['total_score'] = ((combined['Score'] * w1) +
                               (combined['Score_x'] * w2) +
                               (combined['Score_y'] * w3))
    combined['avg_score'] = \
        (combined['total_score'] / combined['total_year_count'])
    return combined.sort_values(by=['total_score'], ascending=False).dropna()


def player_ranks(scores):
    """
    Takes in dataframe of players with scores. Adds ranks based on scores
    (highest score = 1, lowest score = 518). Returns dataframe with players
    sorted according to rank column
    """
    scores.loc[:, 'Rank'] = scores.loc[:, 'total_score'].rank(ascending=False)
    scores['Rank'] = scores['Rank'].astype(int)
    return scores.dropna()


def combine_players_teams(players, teams):
    """
    Takes in players scores dataframe, reads in teams_players csv, and
    merges two datasets on player's full name. Returns combined dataset that
    now has all former player score data along with respective team (3 letter
    unique abbreviation)
    """
    # Read in teams_players dataset (from https://www.nbastuffer.com/2020-2021
    # -nba-player-stats/) & some data formatting in excel
    # Merge datasets for use
    teams_players = players.merge(teams, left_on='player_name',
                                  right_on='FULL NAME', how='right')
    # Drop players without associated team
    teams_players = teams_players[teams_players['TEAM'].notna()]
    return teams_players


def team_scores_combined(players_teams, n=10):
    """
    This function would take in a players scores and team dataframe and
    numbers of players counted in a team and return a new data frame with
    sum, average, and median of team scores. The default value of n is 10
    """
    filtered = players_teams[['total_score', 'TEAM']].dropna()
    n = 10
    top_10 = filtered.groupby('TEAM').head(n).reset_index()

    sums = top_10.groupby('TEAM')['total_score'].sum()
    avgs = top_10.groupby('TEAM')['total_score'].mean()
    meds = top_10.groupby('TEAM')['total_score'].median()

    # Return concatenated df with sum, mean, and median scores for each team
    combined_teams = pd.concat([sums, avgs, meds], axis=1)
    # Set column titles
    combined_teams.columns = ['Sum', 'Average', 'Median']
    return combined_teams


def team_ranks(scores_teams):
    """
    This function would take in a dataframe with team scores and rank
    them in ascending order
    """
    scores_teams['Rank_Sum'] = scores_teams['Sum'].rank(ascending=False)
    scores_teams['Rank_Avg'] = scores_teams['Average'].rank(ascending=False)
    scores_teams['Rank_Med'] = scores_teams['Median'].rank(ascending=False)

    for team in scores_teams:
        scores_teams['Rank_Avg_All'] = ((scores_teams['Rank_Sum'] +
                                        scores_teams['Rank_Avg'] +
                                        scores_teams['Rank_Med']) / 3)
        scores_teams['Rank_Overall'] = \
            scores_teams['Rank_Avg_All'].rank(ascending=True)

    # Rank and sort teams based on Sum of total_score for teams
    scores_teams['Rank_Overall'] = scores_teams['Rank_Overall'].astype(int)
    sorted_ranks = \
        scores_teams.sort_values(by=['Rank_Avg_All'], ascending=True)
    return sorted_ranks


def download(df, file_name):
    """
    This function would take in a dataframe and file_name
    and download the given df in csv with the given filename
    """
    df.to_csv(file_name + '.csv', encoding='utf-8', index=False)


def main():
    # file exist checker
    file_names = os.listdir('../data_upload')
    files_list = ['player_2018.csv', 'player_2019.csv', 'player_2020.csv']
    for file in files_list:
        if file not in file_names:
            year = file[7:11]
            player_csv = GetPlayerCsv()
            player_csv.get_player_csv(year)

    roster = pd.read_csv('../data_upload/teams_players.csv')
    year_2020 = pd.read_csv('../data_upload/player_2020.csv')
    year_2019 = pd.read_csv('../data_upload/player_2019.csv')
    year_2018 = pd.read_csv('../data_upload/player_2018.csv')

    # get player scores of each year
    player_score_2020 = player_scores(year_2020,
                                      w1=.25, w2=.25, w3=.05, w4=.05, w5=.4)
    player_score_2019 = player_scores(year_2019,
                                      w1=.25, w2=.25, w3=.05, w4=.05, w5=.4)
    player_score_2018 = player_scores(year_2018,
                                      w1=.25, w2=.25, w3=.05, w4=.05, w5=.4)

    # merge three created dataframes
    combined = dataset_mergin(player_score_2020, player_score_2019,
                              player_score_2018)

    # get avgerage score for each player
    player_avg = get_average_score(combined, w1=.05, w2=.2, w3=.75)

    # rank players
    player_rank = player_ranks(player_avg)

    # combine player_rank with team roster
    players_teams = combine_players_teams(player_rank, roster)

    # calculate team score for each team based on players
    team_scores = team_scores_combined(players_teams)

    # rank team
    team_rank = team_ranks(team_scores)

    # download files
    download(player_rank, "player_rank")
    download(team_rank, "team_rank")


if __name__ == '__main__':
    main()
