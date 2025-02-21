import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import os

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.encoder_soil = LabelEncoder()
        self.encoder_crop = LabelEncoder()
        self.encoder_fertilizer = LabelEncoder()
        self.df = None

        # Load existing encoders if available
        if os.path.exists("data_encoders.pkl"):
            with open("data_encoders.pkl", "rb") as f:
                self.encoder_soil, self.encoder_crop, self.encoder_fertilizer = pickle.load(f)

    def load_and_process_data(self):
        self.df = pd.read_csv(self.file_path)

        # Encode categorical variables
        self.df['Soil'] = self.encoder_soil.fit_transform(self.df['Soil'])
        self.df['Crop'] = self.encoder_crop.fit_transform(self.df['Crop'])
        self.df['Fertilizer'] = self.encoder_fertilizer.fit_transform(self.df['Fertilizer'])

        # Save encoders for future use
        with open("data_encoders.pkl", "wb") as f:
            pickle.dump((self.encoder_soil, self.encoder_crop, self.encoder_fertilizer), f)

        return self.df

    def encode_input(self, soil, crop):
        try:
            soil_encoded = self.encoder_soil.transform([soil])[0]
            crop_encoded = self.encoder_crop.transform([crop])[0]
            return soil_encoded, crop_encoded
        except ValueError:
            raise ValueError(f"Unknown soil type '{soil}' or crop '{crop}'. Make sure they exist in the dataset.")

