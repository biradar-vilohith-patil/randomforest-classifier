import pickle
import pandas as pd
import os

def load_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pipeline_path = os.path.join(base_dir, 'models', 'heart_rf_pipeline.pkl')
    with open(pipeline_path, 'rb') as f:
        return pickle.load(f)

def run_inference(input_dict):
    pipeline = load_pipeline()
    df_input = pd.DataFrame([input_dict])
    
    raw_prediction = pipeline.predict(df_input)[0]
    
    if str(raw_prediction).strip() == '1':
        prediction = 1
    elif str(raw_prediction).strip() == '0':
        prediction = 0
    else:
        prediction = int(float(raw_prediction))
        
    probability = max(pipeline.predict_proba(df_input)[0]) * 100
    
    return prediction, probability