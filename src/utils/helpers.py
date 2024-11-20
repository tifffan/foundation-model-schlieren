# helpers.py: Helper functions and data loaders
def load_data(path):
    import pandas as pd
    return pd.read_csv(path)

def preprocess(data):
    ...  # Implement preprocessing steps

