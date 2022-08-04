"""
Isaac Xu & Spencer James Knapp
section AG
This program would help build a dataframe based on
stats acquired from PlayerStats and save the created
dataframe in a csv file
"""
import pandas as pd
from nba_api.stats.static import players
from player_stats import PlayerStats


class GetPlayerCsv:
    """
    This class would create dataframes of stats of
    currently active players in 2020, 2018, and 2019
    and return the client in csv form
    """

    def __init__(self):
        """
        This is the initializer of GetPlayerCsv class
        which would create a new GetPlayerCsv which
        would be able to provide client of csv files of
        player's stats in 2020, 2019, and 2018
        """
        self._player_list = players.get_players()
        self._active_players = [player for player in self._player_list
                                if player['is_active'] is True]

    def get_player_csv(self, year):
        """
        This function would take in a list of dictionaries of
        active players and a given year and it would return a
        data frame of each active player's stats in that year.
        If the given year is not 2020, 2019, or 2018, the function
        would return a reminder to the clinent
        """
        if year not in ['2020', '2019', '2018']:
            return 'Please input the year within [2020, 2019, 2018]'

        name_list = []
        id_list = []
        plus_minus_list = []
        eFG_list = []
        winning_perct_list = []
        total_games = []
        pts_list = []

        for player in self._active_players:
            name = player['full_name']
            name_list.append(name)

            player_id = player['id']
            id_list.append(player_id)

            player = PlayerStats(player_id, year)
            plus_minus_list.append(player.get_plus_minus())
            eFG_list.append(player.get_eFG_avg())
            winning_perct_list.append(player.get_winning_perct())
            total_games.append(player.get_total_games())
            pts_list.append(player.get_player_pts())

        data = {'player_name':  name_list,
                'player_id': id_list,
                'plus_minus': plus_minus_list,
                'eFG_avg': eFG_list,
                'winning_perct': winning_perct_list,
                'total_games': total_games,
                'pts': pts_list
                }

        df = pd.DataFrame(data, columns=['player_name',
                                         'player_id', 'plus_minus',
                                         'eFG_avg', 'winning_perct',
                                         'total_games', 'pts'])
        file_name = 'player_data_' + year
        return df.to_csv(file_name, encoding='utf-8', index=False)
