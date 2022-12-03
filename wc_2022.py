import pandas as pd
import numpy as np

DATADIR = 'https://raw.githubusercontent.com/berkanyuce/World-Cup-2022/blob/main/'


results = pd.read_csv("https://raw.githubusercontent.com/berkanyuce/World-Cup-2022/main/results.csv")
results = results.drop(['Unnamed: 0'], axis=1)


#####
# Machine Learning
#####

#split dataset in features and target variable
feature_cols = ['def', 'mid', 'att', 'ovr','rank']
X = results[feature_cols] # Features
y = results.winner # Target variable
y=y.astype('int')

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)

#### LOGISTIC REGRESSION ####

# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train.values,y_train.values)
y_pred=logreg.predict(X_test)

# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # %69
print("Precision:",metrics.precision_score(y_test, y_pred)) # %71
print("Recall:",metrics.recall_score(y_test, y_pred)) # %77

#### RANDOM FOREST ####
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier()

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # %69
print("Precision:",metrics.precision_score(y_test, y_pred)) # %72
print("Recall:",metrics.recall_score(y_test, y_pred)) # %74

#### SUPPORT VECTOR MACHINE ####
#Import svm model
from sklearn import svm
from sklearn.svm import SVC

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train.values, y_train.values)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # %68
print("Precision:",metrics.precision_score(y_test, y_pred)) # %71
print("Recall:",metrics.recall_score(y_test, y_pred)) # %74


wc = pd.read_csv("https://raw.githubusercontent.com/berkanyuce/World-Cup-2022/main/wc.csv", sep=",")
wc = wc.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)



def match(wc, team1, team2, random_scale=5):
    
    match = pd.DataFrame(columns=['att1','def1','mid1','ovr1','rank1','att2','def2','mid2','ovr2','rank2'], index=[0])
    
    att1 = int(wc[wc.name == team1]['att'].iloc[0])
    def1 = int(wc[wc.name == team1]['def'].iloc[0])
    mid1 = int(wc[wc.name == team1]['mid'].iloc[0])
    ovr1 = int(wc[wc.name == team1]['ovr'].iloc[0])
    rank1 = int(wc[wc.name == team1]['rank'].iloc[0])

    att2 = int(wc[wc.name == team2]['att'].iloc[0])
    def2 = int(wc[wc.name == team2]['def'].iloc[0])
    mid2 = int(wc[wc.name == team2]['mid'].iloc[0])
    ovr2 = int(wc[wc.name == team2]['ovr'].iloc[0])
    rank2 = int(wc[wc.name == team2]['rank'].iloc[0])
    
    match['att1'] = np.random.normal(att1, scale=random_scale)
    match['def1'] = np.random.normal(def1, scale=random_scale)
    match['mid1'] = np.random.normal(mid1, scale=random_scale)
    match['ovr1'] = np.random.normal(ovr1, scale=random_scale)
    match['rank1'] = np.random.normal(rank1, scale=random_scale)

    match['att2'] = np.random.normal(att2, scale=random_scale)
    match['def2'] = np.random.normal(def2, scale=random_scale)
    match['mid2'] = np.random.normal(mid2, scale=random_scale)
    match['ovr2'] = np.random.normal(ovr2, scale=random_scale)
    match['rank2'] = np.random.normal(rank2, scale=random_scale)
    
    match['att'] = match['att1'] - match['att2']
    match['def'] = match['def1'] - match['def2']
    match['mid'] = match['mid1'] - match['mid2']
    match['ovr'] = match['ovr1'] - match['ovr2']
    match['rank'] = match['rank1'] - match['rank2']

    match = match[['att', 'def', 'mid', 'ovr', 'rank']]
    
    match_array = match.values
    
    prediction = logreg.predict(match_array)
    
    winner = None
    
    if prediction == 1:
        winner = team1
    elif prediction == -1:
        winner = team2
    
    return winner
    
def simulate_matches(team1, team2, n_matches=6500):
    
    match_results = []
    for i in range(n_matches):
        match_results.append(match(wc, team1, team2, random_scale=5))
        
    team1_proba = match_results.count(team1) / len(match_results) * 100
    team2_proba = match_results.count(team2) / len(match_results) * 100
    
    print(team1, str(round(team1_proba, 2)) + '%')
    print(team2, str(round(team2_proba,2)) + '%')
    print('-------------------------')
    print()
    
    if team1_proba > team2_proba:
        overall_winner = team1
    else:
        overall_winner = team2
    
    return {'team1': team1,
            'team2': team2,
            'team1_proba': team1_proba, 
            'team2_proba': team2_proba, 
            'overall_winner': overall_winner,
            'match_results': match_results}

####SIMULATION

print('Round of 16:')

ko1 = simulate_matches('netherlands', 'usa')['overall_winner']
ko2 = simulate_matches('argentina', 'australia')['overall_winner']
ko3 = simulate_matches('england', 'senegal')['overall_winner']
ko4 = simulate_matches('france', 'poland')['overall_winner']
ko5 = simulate_matches('japan', 'croatia')['overall_winner']
ko6 = simulate_matches('brazil', 'korea republic')['overall_winner']
ko7 = simulate_matches('morocco', 'spain')['overall_winner']
ko8 = simulate_matches('portugal', 'switzerland')['overall_winner']

print()
print('Quarter Finals:')
print()

quarters1 = simulate_matches(ko1, ko2)['overall_winner']
quarters2 = simulate_matches(ko3, ko4)['overall_winner']
quarters3 = simulate_matches(ko5, ko6)['overall_winner']
quarters4 = simulate_matches(ko7, ko8)['overall_winner']

print()
print('Semi Finals:')
print()

semifinals1 = simulate_matches(quarters1, quarters3)['overall_winner']
semifinals2 = simulate_matches(quarters2, quarters4)['overall_winner']

print()
print('Finals:')
print()

finals = simulate_matches(semifinals1, semifinals2)