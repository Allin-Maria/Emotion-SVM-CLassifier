# Emotion-SVM-CLassifier
This project is a Support Vector Machine (SVM) based text classification system designed to detect emotions from text data. It leverages TF-IDF vectorization and compares various SVM kernels to find the most effective one for emotion detection.

📂 Project Structure
emotion_svm_classifier.py: Main script containing all data loading, preprocessing, training, evaluation, and visualization code.

emotions (1).csv: Input dataset with text-emotion label pairs.

🧠 Features
Text preprocessing: case normalization, stopword removal, stemming, and noise cleaning.

Label encoding for categorical emotion labels.

TF-IDF vectorization of textual data.

SVM classifier training using multiple kernel types: linear, poly, rbf, and sigmoid.

Evaluation metrics: Accuracy, Precision, Recall, F1-Score.

Confusion matrix visualization.

Kernel-wise performance comparison with visual plots.

📦 Dependencies
Install the following packages before running the script:

bash
Copy
Edit
pip install pandas numpy scikit-learn nltk matplotlib seaborn
Also, download necessary NLTK resources (automatically done in the script):

python
Copy
Edit
import nltk
nltk.download('punkt')
nltk.download('stopwords')
🚀 How to Run
Make sure the dataset emotions (1).csv is in the same path or update the script with the correct path.

Run the script:

bash
Copy
Edit
python emotion_svm_classifier.py
The script will:

Preprocess the text

Encode the labels

Vectorize using TF-IDF

Train and evaluate SVM with each kernel

Display confusion matrices and a performance comparison bar chart

📊 Output
Console logs of accuracy and classification reports for each kernel.

Confusion matrix heatmaps.

Bar plot comparing accuracy, precision, recall, and f1-score for each kernel.

Identification of the best-performing kernel.
