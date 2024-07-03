# Flask and ML Application

## Overview
This project is a Flask-based web application that provides machine learning predictions for users based on various tests such as RIASEC, temperament, and personality tests. The predictions aim to recommend suitable professions for users.

## Features
- **Homepage**: A simple homepage rendering an HTML template.
- **Prediction API**: A POST endpoint (`/ai-api/predict`) that accepts a `userId` and returns a recommended career based on the user's test results.
- **Data Processing**: Retrieves data from multiple endpoints, processes it, and updates a CSV file.
- **Data Cleaning and ML Model**: Cleans the data, trains a RandomForestClassifier model, and returns predictions.

## Prerequisites
- Python 3.9+
- Flask
- Pandas
- Scikit-learn
- Requests
- Environment variables for endpoints

## Installation
1. Clone the repository:
   ```bash
   git clone .....
   cd  to repo
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  or  `venv\Scripts\activate`
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   export RAISEC_ENDPOINT='http://example.com/raisec'
   export CAREER_ENDPOINT='http://example.com/career'
   export TEMPERAMENT_ENDPOINT='http://example.com/temperament'
   export PERSONALITY_ENDPOINT='http://example.com/personality'
   ```

## Usage
1. Run the Flask application:
   ```bash
   python app.py
   ```

2. Access the homepage by navigating to `http://127.0.0.1:5000/` in your web browser.

3. Use the API to get predictions:
   - Endpoint: `/ai-api/predict`
   - Method: `POST`
   - Body:
     ```json
     {
       "userId": "12345"
     }
     ```

## Code Explanation
### Flask Application
- `app.py`: Main Flask application script.
  - `home()`: Renders the homepage.
  - `predict()`: Handles prediction requests.

### Machine Learning Script
- `LuminateMLScript.py`: Contains functions for data retrieval, processing, cleaning, and running the machine learning model.
  - `get_data(endpoint)`: Retrieves data from the specified endpoint.
  - `update_csv(data, filename)`: Updates or creates a CSV file with new data.
  - `process_and_structure_data(userId)`: Structures data for the given user.
  - `clean_data(filename, modern_files)`: Cleans and preprocesses data.
  - `run_ml_script(user_id)`: Main function to run the ML script and return predictions.

## File Structure
```
your-repo/
│
├── app.py
├── LuminateMLScript.py
├── requirements.txt
└── templates/
    └── index.html
```

## Example Data Flow
1. User sends a POST request to `/ai-api/predict` with their `userId`.
2. The server retrieves the user's data from multiple endpoints.
3. The data is processed and updated in a CSV file.
4. The data is cleaned and used to train a RandomForestClassifier model.
5. The model predicts the recommended career for the user.
6. The server responds with the recommended career and model accuracy.

## Future Enhancements
- Add user authentication and authorization.
- Enhance the frontend with more features and better design.
- Implement more sophisticated machine learning models for better predictions.
- Optimize data retrieval and processing for faster response times.

## License
This project is licensed under the MIT License.

---

By following this README, you should be able to set up and run the Flask application, as well as understand the underlying code and functionality.