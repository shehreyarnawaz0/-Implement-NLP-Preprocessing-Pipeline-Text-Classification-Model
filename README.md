# 📝 NLP Preprocessing Pipeline & Text Classification Model  

## 🎯 Objective  
The goal of this project is to gain hands-on experience in **Natural Language Processing (NLP)** by building a pipeline that cleans and preprocesses raw text data, transforms it into machine-readable features, and trains a classification model (e.g., sentiment analysis).  

---

## 📂 Dataset  
- **Source:** [IMDB Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)  
- **Shape:** 25,000 labeled reviews (positive/negative)  
- **Columns:**  
  - `review`: The movie review text  
  - `sentiment`: Label (`pos` or `neg`)  

---

## 🛠 Preprocessing Steps  
1. Convert text to lowercase  
2. Remove stopwords, punctuation, numbers, and special characters  
3. Perform tokenization  
4. Apply stemming or lemmatization  

---

## ⚙️ Feature Engineering  
- **TF-IDF Vectorizer** (primary)  
- Compared with **Bag of Words (BoW)**  
- (Optional) Tried embeddings: Word2Vec / GloVe / spaCy  

---

## 🤖 Model Training  
Trained and compared multiple ML models:  
- Logistic Regression  
- Naïve Bayes  
- Support Vector Machine (SVM)  

Optimized hyperparameters using **GridSearchCV**.  

---

## 📊 Evaluation Metrics  
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-score**  
- Confusion Matrix visualization  

---

## 💡 Best Practices  
- Wrote **modular, reusable code** (`preprocess.py`, `train.py`, `evaluate.py`)  
- Ensured **reproducibility** with random seeds  
- Documented pipeline for easy adaptation to new datasets  

---

## 📁 Project Structure  
├── data/ # Dataset (raw and processed)
├── src/ # Source code files
│ ├── preprocess.py # Preprocessing functions
│ ├── train.py # Model training
│ ├── evaluate.py # Model evaluation
├── notebook/ # Jupyter notebooks (experiments)
├── models/ # Saved trained models
├── results/ # Evaluation results (confusion matrix, reports)
├── requirements.txt # Dependencies
└── README.md # Project documentation

yaml
Copy code

---

## 🚀 How to Run  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Run Preprocessing
bash
Copy code
python src/preprocess.py
4️⃣ Train Model
bash
Copy code
python src/train.py
5️⃣ Evaluate Model
bash
Copy code
python src/evaluate.py
📈 Results
Logistic Regression achieved: XX% Accuracy, YY% F1-score

Naïve Bayes achieved: XX% Accuracy, YY% F1-score

SVM achieved: XX% Accuracy, YY% F1-score

(Replace with your actual results + screenshots/plots here)

✨ Bonus Work
Experimented with LSTM, BERT, DistilBERT

Added scikit-learn Pipeline for automation

Explored deployment with Flask/FastAPI
