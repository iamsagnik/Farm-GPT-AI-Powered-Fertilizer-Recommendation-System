import pickle
import os
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.trained = False

        # Load model if it exists
        if os.path.exists("rf_model.pkl"):
            with open("rf_model.pkl", "rb") as f:
                self.model, self.recommendation_mapping = pickle.load(f)
            self.trained = True

    def train(self, data):
      X = data[['Nitrogen', 'Phosphorus', 'Potassium', 'Soil', 'Crop']]
      y_fertilizer = data['Fertilizer']
      
      # Convert categorical variables to numeric
      X = X.apply(pd.to_numeric, errors='coerce')

      self.model.fit(X, y_fertilizer)
      self.recommendation_mapping = dict(zip(data.index, data['Crop_Specific_Recommendations']))
      
      with open("rf_model.pkl", "wb") as f:
          pickle.dump((self.model, self.recommendation_mapping), f)

      self.trained = True
      print("Random Forest Model Trained Successfully")


    def predict(self, nitrogen, phosphorus, potassium, soil_encoded, crop_encoded):
        if not self.trained:
            raise ValueError("Model is not trained yet. Train the model first.")

        X_input = [[nitrogen, phosphorus, potassium, soil_encoded, crop_encoded]]
        fertilizer_pred = self.model.predict(X_input)[0]

        # Get recommendation
        recommendation = self.recommendation_mapping.get(0, "No recommendation available")
        return fertilizer_pred, recommendation
