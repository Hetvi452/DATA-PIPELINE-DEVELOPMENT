# import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# extract
def extract_data(file_path):
    print("Extracting data...")
    df = pd.read_csv("Titanic-Dataset.csv")
    return df

# transform
def transform_data(df):
    print("Transforming data...")

    # Drop unnecessary columns
    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

    # Handle missing values
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Encode categorical columns
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])

    # Scale numerical columns
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    return df

# load
def load_data(df, output_path):
    print("Loading data (saving result.csv)...")
    df.to_csv(output_path, index=False)

# pipeline execution
def run_pipeline():
    input_file = "Titanic-Dataset.csv"
    output_file = "result.csv"   

    df = extract_data(input_file)
    processed_df = transform_data(df)
    load_data(processed_df, output_file)

    print("ETL Pipeline completed successfully!")

# main
if __name__ == "__main__":
    run_pipeline()
