import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model(data_path, models_dir):
    df = pd.read_csv(data_path)
    
    X = df.drop('Sleep Disorder', axis=1)
    y = df['Sleep Disorder']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Daily Steps']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'BMI Category'])
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 150],
        'classifier__max_depth': [5, 8, 12],
        'classifier__min_samples_split': [10, 20],
        'classifier__min_samples_leaf': [2, 5]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_

    os.makedirs(models_dir, exist_ok=True)
    artifact_path = os.path.join(models_dir, 'sleep_rf_pipeline.pkl')
    
    with open(artifact_path, 'wb') as f:
        pickle.dump(best_pipeline, f)
        
    return best_pipeline, X_test, y_test

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'cleaned_sleep_health.csv')
    models_dir = os.path.join(base_dir, 'models')
    train_and_save_model(data_path, models_dir)