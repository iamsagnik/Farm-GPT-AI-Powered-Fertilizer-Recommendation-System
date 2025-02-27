# AI-Powered Fertilizer Recommendation System (MLP-Based) 🌱🤖
This project implements a multi-layer perceptron (MLP) model using TensorFlow to predict the best fertilizer based on soil type, crop, and nutrient composition. The model achieves 91.5% accuracy, outperforming traditional approaches like Random Forest. The trained model is deployed via a Flask API, allowing seamless integration with a web-based frontend.

## Key Features ✨
- 🧠 Deep neural network architecture with adaptive dropout regularization
- 🌡️ Multi-input processing for soil composition, crop type, and environmental factors
- ⚖️ Dynamic class weighting for handling imbalanced agricultural datasets
- 📊 Real-time NPK (Nitrogen, Phosphorus, Potassium) optimization analytics
- 🔄 Seamless integration of TensorFlow models with production-grade Flask API

## 💊 Web Application

### 1. Predict Fertilizer

- **Endpoint:** `/predict`
- **Method:** `POST`
- **Example Request (HTML Form Submission):**
  - Temperature: `25.4`
  - Soil Type: `Loamy`
  - Nitrogen: `40`
  - Phosphorus: `20`
  - Potassium: `30`
  - Crop: `Wheat`
- **Response Format:** HTML page displaying the recommended fertilizer.


## 📊 Dataset
The dataset includes:
- **Soil Type** (Categorical)
- **Crop Type** (Categorical)
- **Temperature** (Numerical)
- **Nitrogen (N), Phosphorus (P), Potassium (K)** (Numerical)
- **Fertilizer Type** (Categorical - Target Variable)

## Tech Stack 🛠️
**Core AI**  
`TensorFlow` `Keras` `Scikit-learn` `Pandas` `NumPy`

**Backend**  
`Flask` `Joblib` `LabelEncoder` `StandardScaler`

**Frontend**  
`HTML5` `CSS3` `Jinja2`

## 🏗️ Model Architecture
- **Input Layer:** Accepts 6 features (scaled numerical + encoded categorical values).
- **Hidden Layers:**
  - Dense (256 units, ReLU) + Dropout (0.3)
  - Dense (128 units, ReLU)
  - Dense (64 units, ReLU) + Dropout (0.2)
- **Output Layer:** Softmax activation for multi-class classification.

## 🏋️‍♂️ Training Details
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Epochs:** 50
- **Batch Size:** 32
- **Class Balancing:** Computed using `compute_class_weight` from `sklearn`.

## 📈 Performance
- **Final Test Accuracy:** 91.5%
- **Previous Random Forest Model:** Overfitted and underperformed compared to the MLP.

## 🖥️ How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fertilizer-recommendation.git
   cd fertilizer-recommendation   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Flask web application:
   ```bash
   python app.py
4. Open your browser and visit:
   ```bash
   http://127.0.0.1:5000
5. Enter the required inputs into the web form and get the recommended fertilizer.

## 🌟 Web Interface
The frontend provides an intuitive form where users enter input values. The application processes the data and displays the recommended fertilizer dynamically on the webpage.
