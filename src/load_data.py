import pandas as pd

def load_data(path):
    """Loads churn dataset from CSV."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print(f"File not found at: {path}")
        return None
