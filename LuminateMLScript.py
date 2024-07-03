import csv
import os
import requests
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


"""function to get data from the lML server"""


# Load environment variables
raisec_base_url = os.getenv('RAISEC_ENDPOINT')
career_base_url = os.getenv('CAREER_ENDPOINT')
temperament_base_url = os.getenv('TEMPERAMENT_ENDPOINT')
personality_base_url = os.getenv('PERSONALITY_ENDPOINT')


def get_data(endpoint):
    response = requests.get(endpoint)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: Unable to retrieve data from {endpoint}")
        return None


"""function that retrieves data and affix into csv file or update the csv file """


def update_csv(data, filename='data.csv'):
    try:
        with open(filename, 'r', newline='') as file:
            reader = csv.DictReader(file)
            existing_data = list(reader)
    except FileNotFoundError:
        existing_data = []

    existing_data.append(data)
    """write the data into the csv file"""
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerows(existing_data)


"""function to process and structure the data"""


def process_and_structure_data(userId):
    raisec_endpoint = f'{raisec_base_url}/{userId}'
    career_endpoint = f'{career_base_url}/{userId}'
    temperament_endpoint = f'{temperament_base_url}/{userId}'
    personality_endpoint = f'{personality_base_url}/{userId}'

    raisec_data = get_data(raisec_endpoint)
    career_data = get_data(career_endpoint)
    personality_data = get_data(personality_endpoint)
    temperament_data = get_data(temperament_endpoint)

    """structure the data into a dictionary format"""
    if raisec_data and career_data and temperament_data and personality_data:
        structured_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'realistic': next((item['score'] for item in raisec_data if item['area'] == 'Realistic'), 0),
            'investigative': next((item['score'] for item in raisec_data if item['area'] == 'Investigative'), 0),
            'artistic': next((item['score'] for item in raisec_data if item['area'] == 'Artistic'), 0),
            'social': next((item['score'] for item in raisec_data if item['area'] == 'Social'), 0),
            'enterprising': next((item['score'] for item in raisec_data if item['area'] == 'Enterprising'), 0),
            'conventional': next((item['score'] for item in raisec_data if item['area'] == 'Conventional'), 0),
            'temperament': [item['temperamentName'] for item in temperament_data],
            'personality': [item['personalityName'] for item in personality_data],
            'recommended_professions': [rec['title'] for rec in career_data],
        }
        return structured_data
    else:
        return None


"""function to clean the data"""


def integrate_modern_professions(modern_files, data):
    modern_professions = set()
    for file in modern_files:
        df = pd.read_csv(file)
        modern_professions.update(df['title'].tolist())
    data['recommended_professions'] = data['recommended_professions'].apply(
        lambda x: x if x in modern_professions else None)
    return data


"""function to clean the data"""


def clean_data(filename='data.csv', modern_files=[]):
    data = pd.read_csv(filename)

    """Convertning the RIASEC score columns to numeric"""
    ria_scores = ['realistic', 'investigative', 'artistic',
                  'social', 'enterprising', 'conventional']
    data[ria_scores] = data[ria_scores].apply(pd.to_numeric, errors='coerce')

    data = integrate_modern_professions(modern_files, data)

    """Encode categorical variables"""
    label_encoder = LabelEncoder()
    data['temperament'] = label_encoder.fit_transform(data['temperament'])
    data['personality'] = label_encoder.fit_transform(data['personality'])
    data['recommended_professions'] = label_encoder.fit_transform(
        data['recommended_professions'])

    """Normalize RIASEC scores"""
    data[ria_scores] = data[ria_scores] / data[ria_scores].max()

    """Handle recommended careers"""
    X = data.drop(columns=['timestamp', 'recommended_professions'])
    y = data['recommended_professions']

    """Save cleaned data"""
    X.to_csv("preprocessed_student_data.csv", index=False)
    y.to_csv("student_careers.csv", index=False)


"""function to run the ML script"""


def clear():
    paths = ["preprocessed_student_data.csv","student_careers.csv" ]
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


def run_ml_script(user_id):
    data = process_and_structure_data(user_id)
    if data:
        update_csv(data)
        modern_files = ['career_file_1.csv', 'career_file_2.csv']
        """Clean data after updating the CSV"""
        clean_data(modern_files=modern_files)

        """Load the cleaned data"""
        X = pd.read_csv("preprocessed_student_data.csv")
        y = pd.read_csv("student_careers.csv")

        """Split the data into training and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        """Instantiate and fit LabelEncoder"""
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(
            y_train.values.ravel())

        """Training a RandomForestClassifier model"""
        model = RandomForestClassifier()
        model.fit(X_train, y_train_encoded)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        new_data = X_test.iloc[0].values.reshape(1, -1)
        print("before prediction:", new_data)
        recommended_career_index = model.predict(
            new_data)[0]
        print("after prediction:", recommended_career_index)

        recommended_career_text = data['recommended_professions'][recommended_career_index]
        clear()

        return {
            "recommended_career": str(recommended_career_text),
            "accuracy": float(accuracy * 100)            
        }
    
    else:
        return {"error": "Failed to update data"}

if __name__ == 'run_ml_script':
    run_ml_script()
