"""
Isaac Xu & Spencer James Knapp
section AG
This program implements a PlayerStats
class which would be helpful for calculating
player and team scores based on stats given
by this class
"""
from nba_api.stats.endpoints import playergamelog


class PlayerStats:
    """
    The PlayerStats class would take in
    a player id and year and it would return
    that player's total games, average plus
    minus, eFG, or pts, or winning percentage
    of the given year
    """

    def __init__(self, player_id, year):
        """
        This is the initializer of PlayerStats.
        It would take in plaeyr id and year to
        create a new PlayerStats to represent
        the player at the given year
        """
        self._year = year
        self._player_id = player_id
        self._gamelog_player_df = \
            playergamelog.PlayerGameLog(player_id,
                                        season=self._year).get_data_frames()[0]

    def get_total_games(self):
        """
        This function would return the amount of
        games the given player played in the given
        year
        """
        return len(self._gamelog_player_df)

    def get_plus_minus(self):
        """
        This funcrion would return the average plus
        minus of the given player in the given year
        """
        return self._gamelog_player_df['PLUS_MINUS'].mean()

    def get_eFG_avg(self):
        """
        This function would return the average
        effective field gaal percentage of the
        given player in the given year. If the
        given player didn't player in the given
        year at all, the function would return
        0 directly
        """
        FGM_avg = self._gamelog_player_df['FGM'].mean()
        FGA_avg = self._gamelog_player_df['FGA'].mean()
        FG3M_avg = self._gamelog_player_df['FG3M'].mean()
        if FGA_avg == 0:
            return 0
        eFG_avg = ((FGM_avg + 0.5 * FG3M_avg) / FGA_avg) * 100
        return eFG_avg

    def get_winning_perct(self):
        """
        This function would return the winning percentage
        of the given player in the given year. If the given
        player didn't play in that year at all, the function
        would return 0 directly
        """
        w_games = 0
        l_games = 0

        for w_or_l in self._gamelog_player_df['WL']:
            if w_or_l == 'W':
                w_games += 1
            else:
                l_games += 1

        if (w_games == 0) and (l_games == 0):
            return 0

        winning_perct = (w_games / (l_games + w_games)) * 100
        return winning_perct

    def get_player_pts(self):
        """
        This function would return the avgerage points
        per game of the given player in the given year
        """
        return self._gamelog_player_df['PTS'].mean()
