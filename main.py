import streamlit as st

st.set_page_config(layout="wide")


col1, col2, col3 = st.columns(3)
col2.markdown("""
              # World Cup 2022 Predictor

Developed using machine learning, the model tries to predict the results of world cup matches. The model, which was originally developed to predict only the teams that made it to the round of 16, has been updated to make predictions in the group stages as well. The model produces an outcome based on the probability that both parties win. If these odds are close, it recommends playing a double chance. In the round of 16, it will only predict the team that has reached the next round.

The theoretical success rate of the model is 69% ("accuracy" with machine learning lexicon). Reaching this rate at the end of the tournament will make the model successful.

The page will be updated daily.

              """)



group_successed_matches = 32.0
group_total_matches = 48.0

knockout_successed_matches = 6.0
knockout_total_matches = 6.0

total_successed_matches = group_successed_matches + knockout_successed_matches
total_matches = group_total_matches + knockout_total_matches


col2.header("All Predicted Matches")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col3.metric("Predicted Number Of Matches", value=int(total_matches))
succes_rate_str = str(int(total_successed_matches/total_matches*100))+'%'
succes_rate = str(int(total_successed_matches)) + "/" + str(int(total_matches))
col4.metric("Success Rate", value=succes_rate_str, delta=succes_rate)

col1, col2, col3, col4 = st.columns(4)
col1.header("Group Matches")
col2.header("⚽️")
col1.metric("Predicted Number Of Matches", value=int(group_total_matches))
succes_rate_str = str(int(group_successed_matches/group_total_matches*100))+'%'
succes_rate = str(int(group_successed_matches)) + "/" + str(int(group_total_matches))
col2.metric("Success Rate", value=succes_rate_str, delta=succes_rate)

col3.header("Knockout")
col4.header("Stage")
col3.metric("Predicted Number Of Matches", value=int(knockout_total_matches))
succes_rate_str = str(int(knockout_successed_matches/knockout_total_matches*100))+'%'
succes_rate = str(int(knockout_successed_matches)) + "/" + str(int(knockout_total_matches))
col4.metric("Success Rate", value=succes_rate_str, delta=succes_rate)

import pandas as pd
import numpy as np
group_matches = {'Team 1': ["Qatar",
                "England",
                "Senegal",
                "USA",
                "Argentina",
                "Denmark",
                "Mexico",
                "France",
                "Morocco",
                "Germany",
                "Spain",
                "Belgium",
                "Switzerland",
                "Uruguay",
                "Portugal",
                "Brazil",
                "Wales",
                "Qatar",
                "Holland",
                "England",
                "Tunisia",
                "Poland",
                "France",
                "Argentina",
                "Japan",
                "Belgium",
                "Croatia",
                "Spain",
                "Cameroon",
                "South Korea",
                "Brazil",
                "Portugal",
                "Ecuador",
                "Holland",
                "Iran",
                "Wales",
                "Australia",
                "Tunisia",
                "Saudi Arabia",
                "Poland",
                "Croatia",
                "Canada",
                "Costa Rica",
                "Japan",
                "Ghana",
                "South Korea",
                "Cameroon",
                "Serbia"
                ], 
     'Team 2': ["Ecuador",
                "Iran",
                "Holland",
                "Wales",
                "Saudi Arabia",
                "Tunisia",
                "Poland",
                "Australia",
                "Croatia",
                "Japan",
                "Costa Rica",
                "Canada",
                "Cameroon",
                "South Korea",
                "Ghana",
                "Serbia",
                "Iran",
                "Senegal",
                "Ecuador",
                "USA",
                "Australia",
                "Saudi Arabia",
                "Denmark",
                "Mexico",
                "Costa Rica",
                "Morocco",
                "Canada",
                "Germany",
                "Serbia",
                "Ghana",
                "Switzerland",
                "Uruguay",
                "Senegal",
                "Qatar",
                "USA",
                "England",
                "Denmark",
                "France",
                "Mexico",
                "Argentina",
                "Belgium",
                "Morocco",
                "Germany",
                "Spain",
                "Uruguay",
                "Portugal",
                "Brazil",
                "Switzerland"
                ],
     'Prediction': ["X2",
                    "1",
                    "2",
                    "1X",
                    "1",
                    "1",
                    "1X",
                    "1",
                    "X2",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "2",
                    "1",
                    "1",
                    "2X",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1X",
                    "2",
                    "1",
                    "1",
                    "1",
                    "2",
                    "1",
                    "X2",
                    "2",
                    "2",
                    "2",
                    "2",
                    "2",
                    "X2",
                    "X2",
                    "2",
                    "2",
                    "2",
                    "2",
                    "2",
                    "X2"
                    ],
     'Odd': ["1.35",
             "1.35",
             "1.62",
             "1.40",
             "1.18",
             "1.47",
             "1.45",
             "1.24",
             "1.21",
             "1.52",
             "1.17",
             "1.47",
             "1.72",
             "1.82",
             "1.45",
             "1.46",
             "2.20",
             "1.65",
             "1.74",
             "1.57",
             "1.74",
             "1.77",
             "1.80",
             "1.60",
             "1.48",
             "2.04",
             "2.15",
             "1.43",
             "1.74",
             "2.65",
             "1.40",
             "1.95",
             "2.75",
             "1.18",
             "1.23",
             "1.57",
             "1.46",
             "1.35",
             "1.70",
             "1.52",
             "1.38",
             "1.45",
             "1.11",
             "1.30",
             "1.85",
             "1.62",
             "1.23",
             "1.60"],
             
     'Success': ["✅",
                 "✅",
                 "✅",
                 "✅",
                 "❌",
                 "❌",
                 "✅",
                 "✅",
                 "✅",
                 "❌",
                 "✅",
                 "✅",
                 "✅",
                 "❌",
                 "✅",
                 "✅",
                 "❌",
                 "✅",
                 "❌",
                 "❌",
                 "✅",
                 "✅",
                 "✅",
                 "✅",
                 "❌",
                 "❌",
                 "✅",
                 "✅",
                 "❌",
                 "❌",
                 "✅",
                 "✅",
                 "✅",
                 "✅",
                 "✅",
                 "✅",
                 "❌",
                 "❌",
                 "✅",
                 "✅",
                 "✅",
                 "✅",
                 "✅",
                 "❌",
                 "✅",
                 "❌",
                 "❌",
                 "✅",

                 ]}

knockout_matches = {'Stage': [ "Round of 16",
                               "Round of 16",
                               "Round of 16",
                               "Round of 16",
                               "Round of 16",
                               "Round of 16",
                               "Round of 16",
                               "Round of 16"
                               ], 
                    'Team 1': ["Holland",
                               "Argentina",
                               "England",
                               "France",
                               "Japan",
                               "Brazil",
                               "Morocco",
                               "Portugal"
                               ], 
                    'Team 2': ["USA",
                               "Australia",
                               "Senegal",
                               "Poland",
                               "Croatia",
                               "Korea Republic",
                               "Spain",
                               "Switzerland"
                               ],
                'Prediction': ["1",
                               "1",
                               "1",
                               "1",
                               "2",
                               "1",
                               "2",
                               "1"
                               ],
                'Odd': ["1.25",
                        "1.10",
                        "1.25",
                        "1.12",
                        "1.55",
                        "1.12",
                        "1.25",
                        "1.45"
                        ],
             
                'Success': ["✅",
                            "✅",
                            "✅",
                            "✅",
                            "✅",
                            "✅",
                            "❓",
                            "❓"
                            ]}


group_matches_df = pd.DataFrame(data=group_matches)
group_matches_df.index = np.arange(1, len(group_matches_df) + 1)


knockout_matches_df = pd.DataFrame(data=knockout_matches)
knockout_matches_df.index = np.arange(1, len(knockout_matches_df) + 1)

# Function 
def color_df(val):
    if val == "✅":
        color = 'green'
    else :
        color = '#ff7e7e'
    return f'background-color: {color}'

col1, col2 = st.columns(2)

col1.dataframe(group_matches_df.style.applymap(color_df, subset=['Success']),
                  800, 1730)

col2.dataframe(knockout_matches_df.style.applymap(color_df, subset=['Success']),
                  800, 1730)



# streamlit run /Users/berkanyuce/Documents/GitHub/World-Cup-2022/main.py

