import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import yaml
import pickle

def train_model():
    # Load parameters
    with open('params.yaml') as f:
        params = yaml.safe_load(f)

    # Load data
    df = pd.read_csv(params['processed_data_source'])
    X = df.drop('species', axis=1)
    y = df['species']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['test_size'], random_state=params['random_state'])

    # Train model
    model = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
    model.fit(X_train, y_train)

        # Save the model
    model_directory = 'model'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    
    with open(params['model_path'], 'wb') as f:
        pickle.dump(model, f)
    
    print(f'Model saved to {params['model_path']}')

    # Evaluate model
    score = model.score(X_test, y_test)
    print(f'Test Accuracy: {score}')

if __name__=="__main__":

    train_model()