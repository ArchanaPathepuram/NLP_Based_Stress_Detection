# 🧠 NLP-Based Stress Detection|Bi-Directional-LSTM

This project uses Natural Language Processing (NLP) and deep learning to identify stress-related text. I worked with text data, applied preprocessing techniques, and compared different neural network models to understand how well they classify text as stress-related or non-stress-related.

## ✨ Features

- 📝 Text preprocessing and cleaning
- 📊 Exploratory Data Analysis (EDA)
- 🔤 GloVe word embeddings
- 🤖 Comparison of RNN, LSTM, GRU, and BiLSTM models
- 📈 Model evaluation using accuracy, precision, recall, F1-score, and confusion matrices
- 🧪 Testing the BiLSTM model on separate, unseen text samples

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- NLTK
- Matplotlib and Seaborn
- TensorFlow and Keras
- Scikit-learn
- GloVe word embeddings
- Jupyter Notebook / Google Colab

## 📂 Dataset

The project uses `Stress.csv`, which contains text data with the following columns:

| Column | Description |
|---|---|
| `subreddit` | Source subreddit of the post |
| `text` | Text used for stress detection |
| `label` | Class label indicating stress or non-stress |

The dataset initially contained **2,838 records**. Seven duplicate records were removed during preprocessing.

## 🔄 Text Preprocessing

Before training the models, I applied the following steps:

- Converted text to lowercase.
- Removed URLs, HTML tags, and unwanted characters.
- Tokenized the text using NLTK.
- Removed stopwords while preserving important negation words such as *not*, *no*, and *never*.
- Converted text into sequences and padded them to a maximum length of 153.
- Used **100-dimensional GloVe word embeddings** to represent words numerically.

## 🤖 Models Implemented

I experimented with four deep learning models:

- **RNN:** Used to learn sequential patterns in text.
- **LSTM:** Used to capture longer-term dependencies in text.
- **GRU:** Used as a recurrent model with a simpler gating mechanism than LSTM.
- **BiLSTM:** Processes text in both forward and backward directions to capture contextual information.

## 📊 Model Comparison

The following table shows the recorded accuracy and F1-score results from the model comparison experiments.

| Model | Accuracy | F1-Score |
|---|---:|---:|
| RNN | 56.2% | 52.4% |
| LSTM | 90.4% | 74.8% |
| GRU | 52.8% | 54.7% |
| BiLSTM | 88.1% | 72.7% |

*Note: These are recorded comparison results. The notebook contains separate evaluation runs, so these values may differ from the detailed BiLSTM test evaluation below.*

## 📈 BiLSTM Test Results

The notebook's detailed BiLSTM evaluation used **567 test samples**.

| Metric | No Stress (Class 0) | Stress (Class 1) |
|---|---:|---:|
| Precision | 0.66 | 0.74 |
| Recall | 0.74 | 0.66 |
| F1-Score | 0.70 | 0.70 |
| Support | 269 | 298 |

| Overall Metric | Result |
|---|---:|
| Test Accuracy | 70.02% |
| Macro Average F1-Score | 0.70 |
| Weighted Average F1-Score | 0.70 |

The confusion matrix for this evaluation was:

|  | Predicted No Stress | Predicted Stress |
|---|---:|---:|
| Actual No Stress | 200 | 69 |
| Actual Stress | 101 | 197 |

These results help show where the model correctly identifies stress and non-stress text and where it makes classification errors.

## 🧪 Testing on Unseen Data

I also evaluated the BiLSTM model using a separate set of **21 text samples** that were not part of the main test evaluation. The samples included both stress-related and non-stress-related examples.

The notebook generates predictions for these texts and evaluates them using external validation accuracy, a confusion matrix, and a classification report.

This additional evaluation helps check how the model handles new text. The predictions include both correct classifications and some errors, so further improvement is possible.

**Note:** The unseen-data evaluation is separate from the 567-sample test evaluation. Its accuracy should be reported independently using the external validation output from the notebook.

## 📏 Evaluation Metrics

I used the following metrics to evaluate model performance:

- **Accuracy:** Percentage of predictions that are correct.
- **Precision:** How many predictions for a class are correct.
- **Recall:** How many actual examples of a class the model identifies correctly.
- **F1-Score:** Combines precision and recall into a single metric.
- **Confusion Matrix:** Shows correct and incorrect predictions for each class.
- **Classification Report:** Summarizes precision, recall, F1-score, and support for each class.

## ⚙️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/ArchanaPathepuram/NLP_Based_Stress_Detection.git
cd NLP_Based_Stress_Detection
```

### 2. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn tensorflow
```

### 3. Prepare the dataset and embeddings

Make sure `Stress.csv` and the required GloVe embedding files are available at the paths expected by the notebook. Follow the notebook's download and setup cells if the files are not already present.

### 4. Run the notebook

Open the notebook in Jupyter Notebook or Google Colab and execute the cells in order.

## 🚀 Future Improvements

- Improve performance on unseen text samples.
- Experiment with hyperparameter tuning and different embedding methods.
- Evaluate additional models such as transformer-based text classifiers.
- Test the model on a larger and more diverse dataset.

## ⚠️ Disclaimer

This project is for educational purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional mental health advice.

## 👩‍💻 Author

**Archana Pathepuram**

- GitHub: [ArchanaPathepuram](https://github.com/ArchanaPathepuram)
- Project Repository: [NLP-Based Stress Detection](https://github.com/ArchanaPathepuram/NLP_Based_Stress_Detection)
