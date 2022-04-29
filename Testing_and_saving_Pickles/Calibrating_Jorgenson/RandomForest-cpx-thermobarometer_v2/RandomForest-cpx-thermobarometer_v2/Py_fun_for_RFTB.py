from sklearn.ensemble import ExtraTreesRegressor
import numpy as np


def train_ExtraTreesRegressor(X_train, y_train, n_estimators=201, max_features='Jorgenson', random_state=None):
    
    if max_features == 'Jorgenson': 
        max_features = round(X_train.shape[1]*2/3)
    
    reg = ExtraTreesRegressor(n_estimators=n_estimators, max_features=max_features, random_state=random_state).fit(X_train, y_train)
    return reg

def get_voting_ExtraTreesRegressor(X, reg):
    voting = []
    for tree in reg.estimators_:
        voting.append(tree.predict(X).tolist()) 
    voting = np.asarray(voting)  
    return voting

def get_voting_stats_ExtraTreesRegressor(X, reg, central_tendency='aritmetic_mean', dispersion='dev_std'):
    
    voting = get_voting_ExtraTreesRegressor(X, reg)
    
    # Central tendency
    if central_tendency == 'aritmetic_mean':
        voting_central_tendency = voting.mean(axis=0)
    elif central_tendency == 'median':
        voting_central_tendency = np.median(voting, axis=0)
    
    # Dispersion
    if dispersion == 'dev_std':
        voting_dispersion = voting.std(axis=0)
    elif dispersion == 'IQR':
        voting_dispersion = np.percentile(voting, 75) - np.percentile(voting, 25)
    
    return  voting_central_tendency, voting_dispersion