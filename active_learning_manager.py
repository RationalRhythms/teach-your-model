import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from model_trainer import ModelTrainer
from uncertainty import UncertaintyCalculator

class ActiveLearningManager:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.uncertainty_calc = UncertaintyCalculator()
        self.unlabeled_file = "data/unlabeled_data.jsonl"
        self.labeled_file = "data/labeled_data.jsonl"
        self.corrections_file = "data/corrections.jsonl"
        self._init_files()
        self._load_model()  
    def _load_model(self):
        """Load the trained model"""
        try:
           self.trainer.load_model()
        except:
          print("Warning: Model not found. Please train the model first.")
    
    def _init_files(self):
        """Initialize data files"""
        os.makedirs("data", exist_ok=True)
    
    def get_current_accuracy(self):
        try:
               from mlflow.tracking import MlflowClient
               client = MlflowClient()
               runs = client.search_runs(experiment_ids=["0"], order_by=["start_time DESC"])
               if runs:
                   latest_accuracy = runs[0].data.metrics.get("accuracy", 0.0)
                   return latest_accuracy
               else:
                   return 0.0
        except:
               return 0.0  # Fallback if MLflow not available
    
    def get_prediction(self, text):
        """Get prediction and handle logging"""
        predicted_class, probabilities = self.trainer.predict(text)
        confidence = np.max(probabilities)
        
        class_names = ['technology', 'science', 'recreation', 'politics', 'forsale']

        val,reasons = self.uncertainty_calc.should_request_human_review(probabilities)
        
        if val:
            print(reasons)
            self._add_unlabeled(text, predicted_class, probabilities)
            return {
                'prediction': class_names[predicted_class],
                'confidence': float(confidence),
                'action': 'logged'
            }
        else:
            return {
                'prediction': class_names[predicted_class],
                'confidence': float(confidence),
                'action': 'ask_user'
            }
    
    def get_unlabeled_samples(self):
        """Get samples needing labels"""
        samples = []
        class_names = ['technology', 'science', 'recreation', 'politics', 'forsale']
        
        try:
            with open(self.unlabeled_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    samples.append({
                        'text': entry['text'],
                        'predicted_class': entry['predicted_class'],
                        'predicted_class_name': class_names[entry['predicted_class']]
                    })
        except FileNotFoundError:
            pass
        
        return samples
    
    def update_label(self, text, human_label):
        """Update label and move to labeled file"""
        # Remove from unlabeled
        unlabeled_entries = []
        labeled_entry = None
        
        try:
            with open(self.unlabeled_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    if entry['text'] == text:
                        labeled_entry = entry
                        labeled_entry['human_label'] = human_label
                        labeled_entry['labeled_at'] = datetime.now().isoformat()
                    else:
                        unlabeled_entries.append(entry)
        except FileNotFoundError:
            return
        
        # Write back unlabeled (without the labeled one)
        with open(self.unlabeled_file, 'w') as f:
            for entry in unlabeled_entries:
                f.write(json.dumps(entry) + '\n')
        
        # Add to labeled
        if labeled_entry:
            with open(self.labeled_file, 'a') as f:
                f.write(json.dumps(labeled_entry) + '\n')
    
    def check_retrain(self):
        """Check if ready to retrain and do it"""
        labeled_count = self._count_labeled_samples()
        
        if labeled_count >= 10:
            accuracy = self._retrain_model()
            self._clear_labeled_data()
            return f"Retrained model. New accuracy: {accuracy:.3f}"
        else:
            return f"Total labeled: {labeled_count}/10 needed for retraining"
    
    def _add_unlabeled(self, text, predicted_class, probabilities):
        """Add sample to unlabeled file"""
        entry = {
            'text': text,
            'predicted_class': -1 if predicted_class is None else int(predicted_class),
            'probabilities': probabilities.tolist(),
            'timestamp': datetime.now().isoformat(),
            'human_label': None
        }
        
        with open(self.unlabeled_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def _count_labeled_samples(self):
        """Count labeled samples"""
        count = 0
        try:
            with open(self.labeled_file, 'r') as f:
                for line in f:
                    count += 1
        except FileNotFoundError:
            pass
        return count
    
    def _retrain_model(self):
        """Retrain model with labeled data"""
        labeled_data = []
        class_names = ['technology', 'science', 'recreation', 'politics', 'forsale']
        
        try:
            with open(self.labeled_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    labeled_data.append({
                        'text': entry['text'],
                        'super_label': entry['human_label'],
                        'super_category': class_names[entry['human_label']]
                    })
        except FileNotFoundError:
            pass
        
        if labeled_data:
            new_data = pd.DataFrame(labeled_data)
            accuracy = self.trainer.retrain_model(new_data)
            return accuracy
        return 0.0
    
    def _clear_labeled_data(self):
        """Clear labeled data after retraining"""
        open(self.labeled_file, 'w').close()

    def check_drift(self):
        """Check drift using embedding similarity"""
        try:
            all_recent_texts = []
            
            try:
                with open(self.unlabeled_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line)
                        all_recent_texts.append(entry['text'])
            except FileNotFoundError:
                pass
            
            try:
                with open(self.labeled_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line)
                        all_recent_texts.append(entry['text'])
            except FileNotFoundError:
                pass
            
            if len(all_recent_texts) < 15:
                return "Need more data for drift check (20+ samples)"
            
            recent_embeddings = self.trainer.vectorizer.transform(all_recent_texts[-15:])

            training_data = pd.read_csv('data/labeled_pool/initial_labeled.csv')
            training_embeddings = self.trainer.vectorizer.transform(training_data['text'])
            
            # Simple drift: average cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(recent_embeddings, training_embeddings)
            avg_similarity = np.mean(similarity)
            
            drift_score = 1 - avg_similarity
            return f"Embedding drift: {drift_score:.3f} (lower = more similar)"
            
        except Exception as e:
            return f"Drift check failed: {e}"