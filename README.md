# AI-Powered Fertilizer Recommendation System 🌱🤖
An intelligent decision support system that combines deep learning with agronomic expertise to deliver personalized fertilizer recommendations, achieving **91.5% prediction accuracy** across diverse agricultural conditions.

## Key Features ✨
- 🧠 Deep neural network architecture with adaptive dropout regularization
- 🌡️ Multi-input processing for soil composition, crop type, and environmental factors
- ⚖️ Dynamic class weighting for handling imbalanced agricultural datasets
- 📊 Real-time NPK (Nitrogen, Phosphorus, Potassium) optimization analytics
- 🔄 Seamless integration of TensorFlow models with production-grade Flask API

## Tech Stack 🛠️
**Core AI**  
`TensorFlow` `Keras` `Scikit-learn` `Pandas` `NumPy`

**Backend**  
`Flask` `Joblib` `LabelEncoder` `StandardScaler`

**Frontend**  
`HTML5` `CSS3` `Jinja2`

## Model Architecture 📐
```python
Model: "Fertilizer_Recommender"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 6)]               0         
_________________________________________________________________
dense (Dense)                (None, 256)               1792      
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               32896     
_________________________________________________________________
dense_2 (Dense)              (None, 64)                8256      
_________________________________________________________________
fertilizer_output (Dense)    (None, 22)                1430      
=================================================================
Total params: 44,374
Trainable params: 44,374
Non-trainable params: 0
