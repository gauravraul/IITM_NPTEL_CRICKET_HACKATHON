import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import itertools
import copy
import time
import math
from collections import Counter
import xgboost as xg


class MyModel:
    def __init__(self) -> None:
        self.start = time.time()

        self.ridge_regressor = Ridge(alpha=10)

        self.venue_keyword_search = {
            "Eden": "Kolkata",
            "Wankhede": "Wankhede",
            "Brabourne": "Brabourne",
            "Patil": "Patil",
            "Maharashtra": "Maharashtra",
            "Jaitley": "Delhi",
            "Kotla": "Delhi",
            "Chidambaram": "Chennai",
            "Punjab": "Punjab",
            "Chinnaswamy": "Bangalore",
            "Holkar": "Indore",
            "Mansingh": "Rajasthan",
            "Green": "Kanpur",
            "Saurashtra": "Saurashtra",
            "Himachal": "Dharmasala",
            "Nehru": "Guwahati",
            "Gandhi": "Uppal",
            "Modi": "Ahmedabad",
        }

        self.venue_dict = {
            "Eden Gardens, Kolkata": "Kolkata",
            "Eden Gardens": "Kolkata",
            "Wankhede Stadium, Mumbai": "Wankhede",
            "Wankhede Stadium": "Wankhede",
            "Brabourne Stadium, Mumbai": "Brabourne",
            "Brabourne Stadium": "Brabourne",
            "Dr DY Patil Sports Academy, Mumbai": "Patil",
            "Dr DY Patil Sports Academy": "Patil",
            "Maharashtra Cricket Association Stadium, Pune": "Maharashtra",
            "Maharashtra Cricket Association Stadium": "Maharashtra",
            "Arun Jaitley Stadium, Delhi": "Delhi",
            "Arun Jaitley Stadium": "Delhi",
            "Feroz Shah Kotla": "Delhi",
            "MA Chidambaram Stadium, Chepauk, Chennai": "Chennai",
            "MA Chidambaram Stadium": "Chennai",
            "MA Chidambaram Stadium, Chepauk": "Chennai",
            "Punjab Cricket Association IS Bindra Stadium": "Punjab",
            "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab",
            "Punjab Cricket Association Stadium, Mohali": "Punjab",
            "M Chinnaswamy Stadium": "Bangalore",
            "M.Chinnaswamy Stadium": "Bangalore",
            "Sawai Mansingh Stadium": "Rajasthan",
            "Holkar Cricket Stadium": "Indore",
            "Green Park": "Kanpur",
            "Saurashtra Cricket Association Stadium": "Saurashtra",
            "Himachal Pradesh Cricket Association Stadium": "Dharmasala",
            "Nehru Stadium": "Guwahati",
            "Rajiv Gandhi International Stadium": "Uppal",
            "Narendra Modi Stadium, Ahmedabad": "Ahmedabad",
        }

        self.required_venue_list = [
            "Kolkata",
            "Wankhede",
            "Brabourne",
            "Patil",
            "Maharashtra",
            "Delhi",
            "Chennai",
            "Uppal",
            "Punjab",
            "Bangalore",
            "Rajasthan",
            "Ahmedabad",
            "Guwahati",
        ]
        self.required_team_list = [
            "Rajasthan Royals",
            "Gujarat Titans",
            "Royal Challengers Bangalore",
            "Lucknow Super Giants",
            "Sunrisers Hyderabad",
            "Punjab Kings",
            "Delhi Capitals",
            "Mumbai Indians",
            "Chennai Super Kings",
            "Kolkata Knight Riders",
        ]

    def fit(self, files_list):
        ball_by_ball_scores_data = files_list[0]
        match_results_data = files_list[1]

        venue_df = match_results_data[["ID", "Venue", "Date", "Team1", "Team2"]]
        ball_by_ball_scores_data = pd.merge(ball_by_ball_scores_data, venue_df, on="ID")
        ball_by_ball_scores_data["Venue"].replace(self.venue_dict, inplace=True)

        ball_by_ball_scores_data["BowlingTeam"] = ball_by_ball_scores_data.apply(
            lambda row: row["Team2"]
            if row["Team1"] == row["BattingTeam"]
            else row["Team1"],
            axis=1,
        )

        ball_by_ball_scores_data = ball_by_ball_scores_data[
            ball_by_ball_scores_data["Venue"].isin(self.required_venue_list)
            & ball_by_ball_scores_data["BattingTeam"].isin(self.required_team_list)
            & ball_by_ball_scores_data["BowlingTeam"].isin(self.required_team_list)
        ]

        ball_by_ball_scores_data = ball_by_ball_scores_data[
            (ball_by_ball_scores_data["overs"] < 6)
            & (ball_by_ball_scores_data["innings"] < 3)
        ]

        bat_runs = (
            ball_by_ball_scores_data.groupby(["ID", "batter"])
            .agg({"batsman_run": "sum", "ballnumber": "count"})
            .reset_index()
        )

        bat_runs.rename(columns={"ballnumber": "balls_faced"}, inplace=True)
        bat_runs["strike_rate"] = round(
            (bat_runs["batsman_run"] / bat_runs["balls_faced"]) * 100, 2
        )
        mean_bat_stats = (
            bat_runs.groupby(["batter"])[["batsman_run", "strike_rate"]]
            .median()
            .reset_index()
        )
        mean_bat_stats.rename(
            columns={"batsman_run": "avg_runs", "strike_rate": "avg_sr"}, inplace=True
        )

        max_bat_stats = (
            bat_runs.groupby(["batter"])[["batsman_run", "strike_rate"]]
            .max()
            .reset_index()
        )
        max_bat_stats.rename(
            columns={"batsman_run": "max_runs", "strike_rate": "max_sr"}, inplace=True
        )

        ball_eco = (
            ball_by_ball_scores_data.groupby(["ID", "bowler"])
            .agg({"total_run": "sum", "ballnumber": "count"})
            .reset_index()
        )
        ball_eco.rename(
            columns={"ballnumber": "balls_bowled", "total_run": "runs_conceded"},
            inplace=True,
        )
        ball_eco["overs_bowled"] = ball_eco["balls_bowled"] // 6
        ball_eco["economy"] = round(
            (ball_eco["runs_conceded"] / ball_eco["overs_bowled"]), 2
        )

        mean_bowl_stats = (
            ball_eco.groupby(["bowler"])
            .agg({"runs_conceded": "median", "economy": "median"})
            .reset_index()
        )

        all_bat_stats = pd.merge(mean_bat_stats, max_bat_stats, on="batter")

        grp_stats = all_bat_stats.apply(lambda row: group_players(self, row), axis=1)
        bowler_grp_stats = mean_bowl_stats.apply(
            lambda row: grp_bowlers(self, row), axis=1
        )

        self.player_grp_avg = pd.Series(
            grp_stats.grp_category.values, index=grp_stats.batter
        ).to_dict()
        self.player_grp_max = pd.Series(
            grp_stats.max_category.values, index=grp_stats.batter
        ).to_dict()
        self.bowler_grp_avg = pd.Series(
            bowler_grp_stats.bowler_category.values, index=bowler_grp_stats.bowler
        ).to_dict()

        ball_by_ball_scores_data = pd.merge(
            ball_by_ball_scores_data, grp_stats, on="batter"
        )
        ball_by_ball_scores_data = pd.merge(
            ball_by_ball_scores_data, bowler_grp_stats, on="bowler"
        )

        batter_list = (
            ball_by_ball_scores_data.groupby(
                ["Date", "Venue", "innings", "BattingTeam"]
            )
            .agg({"batter": lambda x: list(set(x)), "total_run": "sum"})
            .reset_index()
        )

        bowler_list = (
            ball_by_ball_scores_data.groupby(
                ["Date", "Venue", "innings", "BowlingTeam"]
            )
            .agg({"bowler": lambda x: list(set(x)), "total_run": "sum"})
            .reset_index()
        )

        test = batter_list.apply(lambda row: player_grp_func(self, row), axis=1)
        test_bowler = bowler_list.apply(lambda row: bowler_grp_func(self, row), axis=1)

        test.sort_values(["Date", "innings"])
        test_bowler.sort_values(["Date", "innings"])

        test_join = pd.merge(test, test_bowler, on=["Date", "Venue", "innings"])
        test_join.rename(columns={"total_run_x": "total_run"}, inplace=True)

        self.X = test_join[
            [
                "Venue",
                "innings",
                "BattingTeam",
                "lt_r5_sr135",
                "lt_r5_sr135a",
                "bt_r5t10_sr135",
                "bt_r5t10_sr135a",
                "bt_r10t15_sr135",
                "bt_r10t15_sr135a",
                "gt_r15_sr135",
                "gt_r15_sr135a",
                "debutant",
                # "max_lt_10r",
                # "max_bt_10t20r",
                # "max_bt_20t30r",
                # "max_bt_30t40r",
                # "max_ab_40r",
                # "new",
                "batter_count",
                "BowlingTeam",
                "lt_rc5",
                "rc_bt_5t10",
                "rc_bt_10t15_eco_lt_7p5",
                "rc_bt_10t15_eco_gt_7p5",
                "rc_bt_15t20_eco_lt_10",
                "rc_bt_15t20_eco_gt_10",
                "rc_gt_20",
                "new_bolwer",
                "bowler_count",
            ]
        ]
        y = test[["total_run"]]

        self.X = pd.get_dummies(
            self.X, columns=["innings", "Venue", "BattingTeam", "BowlingTeam"]
        )
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y, test_size=0.20, shuffle=False
        )

        self.ridge_regressor.fit(self.X, y)

    def predict(self, test_data):
        test_data.rename(
            columns={
                "venue": "Venue",
                "batting_team": "BattingTeam",
                "bowling_team": "BowlingTeam",
                "batsmen": "batter",
                "bowlers": "bowler",
            },
            inplace=True,
        )

        temp_data = test_data.apply(lambda row: player_grp_func_test(self, row), axis=1)
        temp_data = temp_data.apply(lambda row: bowler_grp_func_test(self, row), axis=1)

        temp_df = temp_data[
            [
                "Venue",
                "innings",
                "BattingTeam",
                "lt_r5_sr135",
                "lt_r5_sr135a",
                "bt_r5t10_sr135",
                "bt_r5t10_sr135a",
                "bt_r10t15_sr135",
                "bt_r10t15_sr135a",
                "gt_r15_sr135",
                "gt_r15_sr135a",
                "debutant",
                # "max_lt_10r",
                # "max_bt_10t20r",
                # "max_bt_20t30r",
                # "max_bt_30t40r",
                # "max_ab_40r",
                # "new",
                "batter_count",
                "BowlingTeam",
                "lt_rc5",
                "rc_bt_5t10",
                "rc_bt_10t15_eco_lt_7p5",
                "rc_bt_10t15_eco_gt_7p5",
                "rc_bt_15t20_eco_lt_10",
                "rc_bt_15t20_eco_gt_10",
                "rc_gt_20",
                "new_bolwer",
                "bowler_count",
            ]
        ]

        test_df = temp_df.apply(lambda row: process_test_venue(self, row), axis=1)

        test_df = pd.get_dummies(
            test_df, columns=["innings", "Venue", "BattingTeam", "BowlingTeam"]
        ).reindex(columns=self.X.columns, fill_value=False)
        test_df = test_df.replace(False, 0)
        test_df = test_df.replace(True, 1)

        test_df_pred = self.ridge_regressor.predict(test_df)

        res = test_df_pred.tolist()
        flat_list = list(itertools.chain(*res))

        print("Time Taken (seconds) = ", time.time() - self.start)

        return flat_list


def process_test_venue(self, row):
    venue = row["Venue"]
    for k, v in self.venue_keyword_search.items():
        if k in venue:
            row["Venue"] = v
    return row


def group_players(self, row):
    r = row["avg_runs"]
    sr = row["avg_sr"]
    max_runs = row["max_runs"]
    max_sr = row["max_sr"]

    if (r < 5) and (sr < 135):
        row["grp_category"] = "lt_r5_sr135"
    elif (r < 5) and (sr >= 135):
        row["grp_category"] = "lt_r5_sr135a"
    elif (r >= 5) and (r < 10) and (sr < 135):
        row["grp_category"] = "bt_r5t10_sr135"
    elif (r >= 5) and (r < 10) and (sr >= 135):
        row["grp_category"] = "bt_r5t10_sr135a"
    elif (r >= 10) and (r < 15) and (sr < 135):
        row["grp_category"] = "bt_r10t15_sr135"
    elif (r >= 10) and (r < 15) and (sr >= 135):
        row["grp_category"] = "bt_r10t15_sr135a"
    elif (r >= 15) and (sr < 135):
        row["grp_category"] = "gt_r15_sr135"
    elif (r >= 15) and (sr >= 135):
        row["grp_category"] = "gt_r15_sr135a"
    else:
        row["grp_category"] = "new"

    if max_runs < 10:
        row["max_category"] = "max_lt_10r"
    elif (max_runs >= 10) and (max_runs < 20):
        row["max_category"] = "max_bt_10t20r"
    elif (max_runs >= 20) and (max_runs < 30):
        row["max_category"] = "max_bt_20t30r"
    elif (max_runs >= 30) and (max_runs < 40):
        row["max_category"] = "max_bt_30t40r"
    elif max_runs >= 40:
        row["max_category"] = "max_ab_40r"
    else:
        row["max_category"] = "new"

    return row


def player_grp_func(self, row):
    batter_list = row["batter"]

    temp_dict = {}
    for i in batter_list:
        cat = self.player_grp_avg.get(i, "debutant")
        if temp_dict.get(cat) is None:
            temp_dict[cat] = 1
        else:
            temp = temp_dict.get(cat)
            temp_dict[cat] += 1

    row["lt_r5_sr135"] = temp_dict.get("lt_r5_sr135", 0)
    row["lt_r5_sr135a"] = temp_dict.get("lt_r5_sr135a", 0)
    row["bt_r5t10_sr135"] = temp_dict.get("bt_r5t10_sr135", 0)
    row["bt_r5t10_sr135a"] = temp_dict.get("bt_r5t10_sr135a", 0)
    row["bt_r10t15_sr135"] = temp_dict.get("bt_r10t15_sr135", 0)
    row["bt_r10t15_sr135a"] = temp_dict.get("bt_r10t15_sr135a", 0)
    row["gt_r15_sr135"] = temp_dict.get("gt_r15_sr135", 0)
    row["gt_r15_sr135a"] = temp_dict.get("gt_r15_sr135a", 0)
    row["debutant"] = temp_dict.get("debutant", 0)
    row["batter_count"] = len(batter_list)

    temp_dict_max = {}
    for i in batter_list:
        cat_max = self.player_grp_max.get(i, "new")
        if temp_dict_max.get(cat_max) is None:
            temp_dict_max[cat_max] = 1
        else:
            temp = temp_dict_max.get(cat_max)
            temp_dict_max[cat_max] += 1

    row["max_lt_10r"] = temp_dict_max.get("max_lt_10r", 0)
    row["max_bt_10t20r"] = temp_dict_max.get("max_bt_10t20r", 0)
    row["max_bt_20t30r"] = temp_dict_max.get("max_bt_20t30r", 0)
    row["max_bt_30t40r"] = temp_dict_max.get("max_bt_30t40r", 0)
    row["max_ab_40r"] = temp_dict_max.get("max_ab_40r", 0)
    row["new"] = temp_dict_max.get("new", 0)

    return row


def player_grp_func_test(self, row):
    temp = row["batter"]
    batter_list = temp.split(",")
    temp_dict = {}
    for i in batter_list:
        cat = self.player_grp_avg.get(i, "debutant")
        if temp_dict.get(cat) is None:
            temp_dict[cat] = 1
        else:
            temp = temp_dict.get(cat)
            temp_dict[cat] += 1

    row["lt_r5_sr135"] = temp_dict.get("lt_r5_sr135", 0)
    row["lt_r5_sr135a"] = temp_dict.get("lt_r5_sr135a", 0)
    row["bt_r5t10_sr135"] = temp_dict.get("bt_r5t10_sr135", 0)
    row["bt_r5t10_sr135a"] = temp_dict.get("bt_r5t10_sr135a", 0)
    row["bt_r10t15_sr135"] = temp_dict.get("bt_r10t15_sr135", 0)
    row["bt_r10t15_sr135a"] = temp_dict.get("bt_r10t15_sr135a", 0)
    row["gt_r15_sr135"] = temp_dict.get("gt_r15_sr135", 0)
    row["gt_r15_sr135a"] = temp_dict.get("gt_r15_sr135a", 0)
    row["debutant"] = temp_dict.get("debutant", 0)
    row["batter_count"] = len(batter_list)

    temp_dict_max = {}
    for i in batter_list:
        cat_max = self.player_grp_max.get(i, "new")
        if temp_dict_max.get(cat_max) is None:
            temp_dict_max[cat_max] = 1
        else:
            temp = temp_dict_max.get(cat_max)
            temp_dict_max[cat_max] += 1

    row["max_lt_10r"] = temp_dict_max.get("max_lt_10r", 0)
    row["max_bt_10t20r"] = temp_dict_max.get("max_bt_10t20r", 0)
    row["max_bt_20t30r"] = temp_dict_max.get("max_bt_20t30r", 0)
    row["max_bt_30t40r"] = temp_dict_max.get("max_bt_30t40r", 0)
    row["max_ab_40r"] = temp_dict_max.get("max_ab_40r", 0)
    row["new"] = temp_dict_max.get("new", 0)

    return row


def grp_bowlers(self, row):
    rc = row["runs_conceded"]
    eco = row["economy"]

    if rc < 5:
        row["bowler_category"] = "lt_rc5"
    elif (rc >= 5) and (rc < 10):
        row["bowler_category"] = "rc_bt_5t10"
    elif (rc >= 10) and (rc < 15) and (eco < 7.5):
        row["bowler_category"] = "rc_bt_10t15_eco_lt_7p5"
    elif (rc >= 10) and (rc < 15) and (eco >= 7.5):
        row["bowler_category"] = "rc_bt_10t15_eco_gt_7p5"
    elif (rc >= 15) and (rc < 20) and (eco < 10):
        row["bowler_category"] = "rc_bt_15t20_eco_lt_10"
    elif (rc >= 15) and (rc < 20) and (eco >= 10):
        row["bowler_category"] = "rc_bt_15t20_eco_gt_10"
    elif rc >= 20:
        row["bowler_category"] = "rc_gt_20"
    else:
        row["bowler_category"] = "new_bolwer"

    return row


def bowler_grp_func(self, row):
    bowler_list = row["bowler"]
    temp_dict = {}
    for i in bowler_list:
        cat = self.bowler_grp_avg.get(i, "new_bowler")
        if temp_dict.get(cat) is None:
            temp_dict[cat] = 1
        else:
            temp = temp_dict.get(cat)
            temp_dict[cat] += 1

    row["lt_rc5"] = temp_dict.get("lt_rc5", 0)
    row["rc_bt_5t10"] = temp_dict.get("rc_bt_5t10", 0)
    row["rc_bt_10t15_eco_lt_7p5"] = temp_dict.get("rc_bt_10t15_eco_lt_7p5", 0)
    row["rc_bt_10t15_eco_gt_7p5"] = temp_dict.get("rc_bt_10t15_eco_gt_7p5", 0)
    row["rc_bt_15t20_eco_lt_10"] = temp_dict.get("rc_bt_15t20_eco_lt_10", 0)
    row["rc_bt_15t20_eco_gt_10"] = temp_dict.get("rc_bt_15t20_eco_gt_10", 0)
    row["rc_gt_20"] = temp_dict.get("rc_gt_20", 0)
    row["new_bolwer"] = temp_dict.get("new_bolwer", 0)
    row["bowler_count"] = len(bowler_list)

    return row


def bowler_grp_func_test(self, row):
    temp = row["bowler"]
    bowler_list = temp.split(",")

    temp_dict = {}
    for i in bowler_list:
        cat = self.bowler_grp_avg.get(i, "new_bowler")
        if temp_dict.get(cat) is None:
            temp_dict[cat] = 1
        else:
            temp = temp_dict.get(cat)
            temp_dict[cat] += 1

    row["lt_rc5"] = temp_dict.get("lt_rc5", 0)
    row["rc_bt_5t10"] = temp_dict.get("rc_bt_5t10", 0)
    row["rc_bt_10t15_eco_lt_7p5"] = temp_dict.get("rc_bt_10t15_eco_lt_7p5", 0)
    row["rc_bt_10t15_eco_gt_7p5"] = temp_dict.get("rc_bt_10t15_eco_gt_7p5", 0)
    row["rc_bt_15t20_eco_lt_10"] = temp_dict.get("rc_bt_15t20_eco_lt_10", 0)
    row["rc_bt_15t20_eco_gt_10"] = temp_dict.get("rc_bt_15t20_eco_gt_10", 0)
    row["rc_gt_20"] = temp_dict.get("rc_gt_20", 0)
    row["new_bolwer"] = temp_dict.get("new_bolwer", 0)
    row["bowler_count"] = len(bowler_list)

    return row
