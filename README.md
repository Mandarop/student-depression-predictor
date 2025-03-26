# Student Depression Predictor

This project predicts the risk of student depression based on various factors such as academic pressure, work pressure, study satisfaction, sleep duration, and financial stress. The model is trained using machine learning and is deployed with a Streamlit frontend and a Flask backend.

## Features
- **Machine Learning Model**: Naive Bayes classifier for prediction.
- **Frontend**: Built using Streamlit for user interaction.
- **Backend**: Flask API for handling requests and making predictions.
- **Dataset**: Uses structured student data with relevant features.

## Installation

### Clone the Repository
```sh
git clone https://github.com/Mandarop/student-depression-predictor.git
cd student-depression-predictor
```

### Set Up Virtual Environment
```sh
python -m venv myenv
source myenv/bin/activate  # For Mac/Linux
myenv\Scripts\activate  # For Windows
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Running the Application

### Start Flask Backend
```sh
python app.py
```

### Start Streamlit Frontend (In a new terminal)
```sh
streamlit run frontend.py
```

## Project Structure
```
student-depression-predictor/
│-- app.py              # Flask API backend
│-- frontend.py         # Streamlit UI
│-- train_model.py      # Model training script
│-- model.pkl          # Trained model file
│-- label_enc.pkl      # Encoded labels
│-- scaler.pkl         # Data scaler
│-- Depression.csv      # Dataset
│-- requirements.txt    # Dependencies
```

## Usage
- Enter the required details in the Streamlit UI.
- Click the **Predict** button.
- Get the depression risk prediction as output.

## Future Improvements
- Improve model accuracy with better feature engineering.
- Add more ML models for comparison.
- Deploy the app on a cloud platform.



