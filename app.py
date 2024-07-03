from flask import Flask, request, jsonify
from flask import render_template
from LuminateMLScript import run_ml_script  # Import the function from your luminate.py script

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ai-api/predict', methods=['POST'])
def predict():
    user_id = request.json.get('userId')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    result = run_ml_script(user_id)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)