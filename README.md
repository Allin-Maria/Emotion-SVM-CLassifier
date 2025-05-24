
Emotion Classification with SVM
This project performs emotion classification on a dataset of text entries using Support Vector Machines (SVM). It includes data preprocessing, feature extraction with TF-IDF, training with different SVM kernels, evaluation, and model persistence.

📂 File: emotion_classification.py
 Features
Text Preprocessing: Lowercasing, URL/user mention removal, punctuation and digit cleaning, stopword removal, stemming.

Label Encoding: Converts emotion labels to numerical values.

TF-IDF Vectorization: Extracts text features with a limit of 20,000 features.

Model Training: Evaluates SVM with multiple kernels (linear, poly, rbf, sigmoid).

Evaluation: Generates accuracy, classification report, and confusion matrix.

Model Saving: Best performing model and preprocessing tools are saved using joblib.

 Requirements
bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn nltk joblib
You must also download NLTK resources the first time you run:

python
Copy
Edit
import nltk
nltk.download('punkt')
nltk.download('stopwords')

 Dataset
Assumes a CSV file emotions.csv with at least two columns:

text: The raw input text.

label: The emotion label.

 Usage
Place emotions.csv in the appropriate path or change the path in the script.

Run the script:

bash
Copy
Edit
python emotion_classification.py
After execution:

The script prints model performance metrics.

Saves:

Best model: best_svm_model.pkl

Vectorizer: tfidf_vectorizer.pkl

Label encoder: label_encoder.pkl

Output
Console output with classification reports.

Confusion matrices and performance comparison bar chart.

Serialized model files for reuse.


