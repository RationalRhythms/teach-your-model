# Teach Your Model

Your personal ML Model - train a smart text classifier through interactive feedback!

## What is this?

A command-line application where you teach a machine learning model to classify text by correcting its mistakes. The more you use it, the smarter it gets through active learning!

## What Can It Classify?

The model learns to categorize supplied paragraph into 5 main topics: technology, science, recreation, politics, forsale

##  Features

- **Interactive Teaching**: Classify text and provide feedback to train the model
- **Self-Improving**: Automatically retrains when it collects enough corrections
- **Smart Uncertainty Detection**: Flags predictions it's unsure about for review
- **Drift Monitoring**: Detects when input patterns change over time
- **No Cloud Required**: Everything runs locally on your machine
- **Persistent Learning**: Your teaching sessions make the model smarter over time

## Technology Stack

### **Machine Learning Core**
- **Passive Aggressive Classifier**: Online learning algorithm that updates weights incrementally, perfect for continuous learning from user feedback
- **TF-IDF Vectorization**: Converts text into numerical features while emphasizing important words and downweighting common ones
- **Scikit-learn**: Industry-standard machine learning library providing the foundation for all ML operations
- **Initial Training Data**: Bootstrapped using the 20 Newsgroups dataset, mapped to the five target categories

### **MLOps & Experiment Tracking**  
- **MLflow**: Complete MLOps platform for tracking experiments, parameters, metrics, and model versions
- **Model Registry**: Automatic versioning and storage of each trained model iteration

### **Active Learning System**
- **Uncertainty Sampling**: Identifies low-confidence predictions using prediction probabilities
- **Human-in-the-Loop**: Seamless integration of human feedback into the training pipeline
- **Automated Retraining**: Triggers model updates when sufficient new labeled data is available

### **Data Management**
- **Stratified Sampling**: Preserves original data distribution during train/test splits
- **Embedding-based Drift Detection**: Uses cosine similarity on TF-IDF features to monitor input pattern changes
- **Persistent Storage**: JSONL files for efficient append-only logging of user interactions


##  Quick Installation

```bash
# 1. Clone and setup
git clone https://github.com/smritik-code/teach-your-model
cd teach-your-model

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the application
pip install -e .

# 4. Start teaching!
tmodel
