import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_model(df):
    X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
    X = X.dropna()
    y = y[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'model/churn_model.pkl')
    print("✅ Model saved to model/churn_model.pkl")


def predict_churn(input_df):
    model = joblib.load('model/churn_model.pkl')

    input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
    input_df = input_df.fillna(0)

    prediction = model.predict(input_df)

    # ✅ Return "Yes" for churn, "No" otherwise
    return "Yes" if prediction[0] == 1 else "No"
