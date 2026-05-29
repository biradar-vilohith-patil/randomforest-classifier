import os
from sklearn.metrics import accuracy_score, classification_report
from src.model import train_and_save_model

def evaluate_performance(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'cleaned_heart.csv')
    models_dir = os.path.join(base_dir, 'models')
    
    pipeline, X_test, y_test = train_and_save_model(data_path, models_dir)
    evaluate_performance(pipeline, X_test, y_test)