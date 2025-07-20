import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(df):
    print("🔍 Dataset Shape:", df.shape)
    print("🔍 Columns:\n", df.columns)
    print("🔍 Null values:\n", df.isnull().sum())

    print("\n🔍 Churn Distribution:")
    print(df['Churn'].value_counts())

    sns.countplot(x='Churn', data=df)
    plt.title('Churn Count')
    plt.savefig('churn_distribution.png')
    plt.close()
