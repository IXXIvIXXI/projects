# README
* Isaac Xu & Spencer James Knapp
* section AG

# Client Code:
	There are two files for client code, “data_merging_ranking.py” and “player_team_model.py”
* __Data_merging_ranking.py__:
The program in this file would merge player stats and team roster dataframes together and return ranked player and team dataframes to clients. At the beginning of the main method, there is a block of code which applies an os module to check if player data files are in the directory or not.
			
	* file exist checker
	```
    file_names = os.listdir('../data_upload')
    files_list = ['player_2018.csv', 'player_2019.csv', 'player_2020.csv']
    for file in files_list:
        if file not in file_names:
            year = file[7:11]
            player_csv = GetPlayerCsv()
            player_csv.get_player_csv(year)
	```
	* Since we have already got dataframes, the program would pass this checker and execute code below. If we don’t have these player stats csv files the program would create csv files we need by using functions from player_stats and get_csv classes.

* __player_team_model.py__:
Program in this file applies graphviz to visualize decision trees of our models. Since graphviz was not installed in our IDE originally, we need to use “pip install graphviz” to download and install graphviz. However, the installed graphviz would not be in the same path as our files. We also need to use “conda install graphviz” to make it in the same path as our files and after finishing these setups, the graphviz should work for our code.
 
# Class Code:
	There are two files for class code, “player_stats.py” and “get_csv.py”
* __player_stats.py__:
Our data are mostly acquired from nba_api. In order to make this class runnable, we need to install nba_api by using “pip install nba_api”.

* __get_csv.py__:
Get_csv would use functions from player_stats to create a dataframe about player stats in the given year. Since our project only includes the year 2020, 2019, and 2018. The get_player_csv function in player_stats would return a string, ‘Please input the year within [2020, 2019, 2018]’, to remind clients to input the year within this range.


# File Path:
Our project reads many csv files from a directory in order to do data merging and ranking or ML training and testing for the given file. Correct file path matters because it would make sure pandas could read in the given file. In our project, we use file path format as ‘../data_upload/’ and we would submit our files in the folder that would make sure this file path format would also work when you test our code.
