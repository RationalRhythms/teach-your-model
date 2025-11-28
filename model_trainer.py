import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
from scipy.special import softmax
from datetime import datetime
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = None
        
    def train_model(self, training_data, model_save_path="models/production/model.joblib"):
        """
        Train model and track with MLflow
        """
        # Start MLflow tracking
        with mlflow.start_run():
            print(" Training model with MLflow tracking...")
            
            X = self.vectorizer.fit_transform(training_data['text'])
            y = training_data['super_label']
            
            self.model = PassiveAggressiveClassifier(
                loss="hinge", 
                C=1.0,
                max_iter=1,           # Important: for partial_fit
                warm_start=True
            )
            
            classes = np.unique(y)
            self.model.partial_fit(X, y, classes = classes)


            predictions = self.model.predict(X)
            accuracy = accuracy_score(y, predictions)
            
            # Log to MLflow
            mlflow.log_param("model_type", "PassiveAggressive Classifier")
            mlflow.log_param("C", 1.0)
            mlflow.log_param("training_samples", len(training_data))
            mlflow.log_metric("accuracy", accuracy)
            
            # Save model
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'trained_at': datetime.now().isoformat(),
                'training_samples': len(training_data)
            }
            joblib.dump(model_data, model_save_path)
            
            # Log model to MLflow
            input_example = self.vectorizer.transform(["sample text for model signature"])
            mlflow.sklearn.log_model(
                self.model, 
                name="PAC_model",  
                input_example=input_example 
            )
                          
            print(f" Model trained! Accuracy: {accuracy:.3f}")
            print(" View results: mlflow ui")
            
            return accuracy
    
    def load_model(self, model_path="models/production/model.joblib"):
        """Load trained model for predictions"""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        return self.model, self.vectorizer
    
    def predict(self, text):
        """Make prediction on new text"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        features = self.vectorizer.transform([text])
        decision_scores = self.model.decision_function(features)[0]

        temperature = 0.4  # Values < 1.0 increase confidence 
        scaled_scores = decision_scores / temperature
        probabilities = softmax(scaled_scores)

        predicted_class = int(np.argmax(decision_scores))
        
        return predicted_class, probabilities
    
    def retrain_model(self, training_data, model_save_path="models/production/model.joblib"):
        with mlflow.start_run():
            print(" Training model with MLflow tracking...")
            
            X = self.vectorizer.transform(training_data['text'])
            y = training_data['super_label']
            
            self.model.partial_fit(X, y)

            original_data = pd.read_csv('data/labeled_pool/initial_labeled.csv')
            combined_data = pd.concat([original_data, training_data], ignore_index=True)

            X = self.vectorizer.transform(combined_data['text'])
            y = combined_data['super_label']

            predictions = self.model.predict(X)
            accuracy = accuracy_score(y, predictions)
            
            # Log to MLflow
            mlflow.log_param("model_type", "PassiveAggressive Classifier")
            mlflow.log_param("C", 1.0)
            mlflow.log_param("training_samples", len(training_data))
            mlflow.log_metric("accuracy", accuracy)
            
            # Save model
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'trained_at': datetime.now().isoformat(),
                'training_samples': len(training_data)
            }
            joblib.dump(model_data, model_save_path)
            
            # Log model to MLflow
            input_example = self.vectorizer.transform(["sample text for model signature"])
            mlflow.sklearn.log_model(
                self.model, 
                name="PAC_model",  
                input_example=input_example 
            )
                          
            print(f" Model trained! Accuracy: {accuracy:.3f}")
            print(" View results: mlflow ui")
            
            return accuracy