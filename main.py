from src.load_data import load_data
from src.eda import perform_eda
from src.model import train_model

def main():
    print("📦 Loading data...")
    df = load_data("data/churn_data.csv")

    if df is not None:
        print("🧪 Performing EDA...")
        perform_eda(df)

        print("🤖 Training model...")
        train_model(df)

if __name__ == "__main__":
    main()
