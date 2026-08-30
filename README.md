# 📰 Fake News Detector

An end-to-end **Fake News Detection system using Natural Language Processing (NLP) and Machine Learning**.

The project analyzes the textual content of a news article and predicts whether it is **likely Real or Fake**. It includes data exploration, preprocessing, NLP techniques, TF-IDF feature extraction, machine-learning model comparison, evaluation, model selection, and deployment through a Streamlit application.

The application also includes **user authentication, password hashing, MySQL-based prediction history, customizable settings, confidence scoring, and low-confidence warnings**.

---

## 📌 Project Overview

Fake and misleading news can be difficult to identify because articles can be written in a convincing and professional style.

This project attempts to identify patterns in the text of news articles and classify them into two categories:

* **1 → Real News**
* **0 → Fake News**

The project uses traditional machine-learning techniques for text classification rather than relying on an external fact-checking service.

> **Important:** This application is an ML-based text classifier and is not a definitive fact-checking system. A prediction or confidence score does not guarantee that an article is factually true.

---

## 🎯 Objectives

The main objectives of this project are:

1. Explore and understand a real/fake news dataset.
2. Clean and prepare the textual data.
3. Apply NLP preprocessing techniques.
4. Convert text into numerical features using TF-IDF.
5. Train and compare multiple machine-learning models.
6. Evaluate models using multiple classification metrics.
7. Select a suitable model for deployment.
8. Export the trained model and TF-IDF vectorizer.
9. Build a user-facing Streamlit application.
10. Store authenticated users' prediction history using MySQL.

---

## 📂 Dataset

The project uses two separate CSV datasets:

```text
True.csv
Fake.csv
```

The original datasets contain the following columns:

```text
title
text
subject
date
```

The project adds a new `label` column:

```text
Real dataset → label = 1
Fake dataset → label = 0
```

### Dataset Size

Before cleaning:

| Dataset   |   Rows |
| --------- | -----: |
| Real News | 21,417 |
| Fake News | 23,481 |
| Combined  | 44,898 |

Duplicate rows were removed from both datasets before combining them. After the NLP preprocessing stage, duplicate articles based on processed token text were removed again, reducing the working dataset from **44,689 to 38,639 articles**.
The final merged dataset contains:

* **23,478 Fake News articles**
* **21,211 Real News articles**

---

## 🔍 Exploratory Data Analysis

The project performs basic exploratory data analysis to understand the dataset.

The analysis includes:

* Dataset shape
* Column information
* Missing-value checking
* Duplicate checking
* Class distribution
* Title length analysis
* Article text length analysis
* Visualizations of the label distribution

The initial dataset contained no missing values in the inspected columns. Duplicate rows were also explicitly checked before further processing.

The class distribution is relatively close, although the dataset contains more fake-news samples than real-news samples.

---

## 🧹 Data Cleaning

Duplicate records were removed before merging the datasets:

```python
real_dataset = real_dataset.drop_duplicates()
fake_dataset = fake_dataset.drop_duplicates()
```

The two datasets were then merged:

```python
df = pd.concat([real_dataset, fake_dataset], ignore_index=True)
```

The combined dataset was shuffled using a fixed random state for reproducibility:

```python
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
```

---

## 🧠 NLP Preprocessing

Natural Language Processing techniques were applied to the article text.

### 1. Tokenization

The article text was converted into individual words using NLTK:

```python
df["tokens"] = df["text"].apply(word_tokenize)
```

### 2. Stopword Removal

Common English stopwords were removed:

```python
word = set(stopwords.words("english"))

df["tokens"] = df["tokens"].apply(
    lambda words: [w for w in words if w.lower() not in word]
)
```

### 3. Lemmatization

Words were lemmatized using `WordNetLemmatizer`:

```python
lemmatizer = WordNetLemmatizer()

df["tokens"] = df["tokens"].apply(
    lambda words: [lemmatizer.lemmatize(word) for word in words]
)
```

### 4. Reconstructing Text

The processed tokens were joined back into text:

```python
df["tokens"] = df["tokens"].apply(lambda x: " ".join(x))
```

After processing, duplicate records based on the processed token text were removed.

### Modeling Decision

Both processed `tokens` and the original `text` were considered as possible model inputs.

The notebook notes that the processed token representation produced less accurate results, so the original `text` column was ultimately used as the independent variable for model training.

---

## ✂️ Train-Test Split

The data was divided into training and testing sets using:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.2,
    stratify=y
)
```

### Split Configuration

| Parameter      | Value |
| -------------- | ----- |
| Test size      | 20%   |
| Training size  | 80%   |
| Random state   | 42    |
| Stratification | Yes   |

Stratification was used so that the class distribution is maintained across the training and testing sets.

---

## 🔢 TF-IDF Feature Extraction

Machine-learning models cannot directly process raw text, so the article text was converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The project uses:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)
```

The vectorizer is fitted only on the training data and then used to transform the test data.

---

# 🤖 Machine Learning Models

Multiple classification algorithms were trained and evaluated using the same TF-IDF representation.

The models included:

* Logistic Regression
* Linear SVC
* Random Forest Classifier
* Multinomial Naive Bayes
* XGBoost Classifier

---

## 1. Logistic Regression

The first main model was Logistic Regression:

```python
model1 = LogisticRegression()
model1.fit(X_train, y_train)

y1_pred = model1.predict(X_test)
```

### Performance

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 98.68% |
| Precision | 98.27% |
| Recall    | 99.34% |
| F1-Score  | 98.80% |

---

## 2. Linear SVC

A Linear Support Vector Classifier was also trained:

```python
model2 = LinearSVC()
model2.fit(X_train, y_train)

y2_pred = model2.predict(X_test)
```

### Performance

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 99.38% |
| Precision | 99.20% |
| Recall    | 99.67% |
| F1-Score  | 99.44% |

---

## 3. Random Forest Classifier

Random Forest was evaluated as another classification approach.

### Performance

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 94.79% |
| Precision | 96.06% |
| Recall    | 94.36% |
| F1-Score  | 95.20% |

---

## 4. Multinomial Naive Bayes

Multinomial Naive Bayes was evaluated for comparison with the linear models.

### Performance

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 93.97% |
| Precision | 92.38% |
| Recall    | 97.00% |
| F1-Score  | 94.64% |

---

## 5. XGBoost

XGBoost achieved the highest benchmark performance among the tested models.

### Performance

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 99.70% |
| Precision | 99.58% |
| Recall    | 99.88% |
| F1-Score  | 99.73% |

---

# 📊 Model Comparison

| Model               |   Accuracy |  Precision |     Recall |   F1-Score |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     98.68% |     98.27% |     99.34% |     98.80% |
| Linear SVC          |     99.38% |     99.20% |     99.67% |     99.44% |
| Random Forest       |     94.79% |     96.06% |     94.36% |     95.20% |
| Multinomial NB      |     93.97% |     92.38% |     97.00% |     94.64% |
| XGBoost             | **99.70%** | **99.58%** | **99.88%** | **99.73%** |

These results show that the more complex models achieved slightly higher benchmark performance than the basic Logistic Regression model.

---

# ⚙️ Hyperparameter Tuning

Hyperparameter tuning was performed using `RandomizedSearchCV` on Logistic Regression.

The search space included:

```python
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs"],
    "max_iter": [1000, 2000]
}
```

The search used:

```python
n_iter = 20
cv = 5
scoring = "f1"
n_jobs = -1
random_state = 42
```

## The best cross-validated F1 score reported by the search was **0.9915**. The best parameter setting used `C=100`, `solver='liblinear'`, and `max_iter=1000`.

# ✅ Final Model Selection

Although some tuned and alternative models achieved slightly higher benchmark scores, **Logistic Regression was selected as the final deployment model**.

The main reason was application compatibility.

Logistic Regression provides:

```python
predict_proba()
```

which allows the application to generate a confidence score and display a warning when the model is uncertain.

Therefore, model selection was based not only on the highest evaluation metric, but also on the requirements of the deployed application.

---

# 💾 Model Export

The selected Logistic Regression model and TF-IDF vectorizer were saved using Joblib:

```python
joblib.dump(model1, "final_model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
```

They can later be loaded directly by the Streamlit application:

```python
model = joblib.load("final_model.pkl")
tfidf = joblib.load("tfidf.pkl")
```

---

# 🖥️ Streamlit Application

The trained model is integrated into a Streamlit web application.

### Main Prediction Flow

```text
User enters article
        ↓
Text validation
        ↓
TF-IDF transformation
        ↓
Logistic Regression prediction
        ↓
Prediction + confidence score
        ↓
Low-confidence warning when required
        ↓
Optional MySQL history storage
```

The application requires a minimum article length before making a prediction.

---

# 🔐 User Authentication

The application includes:

### Sign Up

Users can create an account using:

* Username
* Password
* Password confirmation

Passwords are hashed using **bcrypt** before being stored.

### Login

During login, the application:

1. Searches for the username in MySQL.
2. Retrieves the stored password hash.
3. Verifies the entered password using bcrypt.
4. Stores the authenticated user's ID and username in Streamlit session state.

---

# 🗄️ MySQL Integration

MySQL is used as the backend database for user accounts and prediction history.

The main relationship is:

```text
USERS
  │
  │ user_id
  ▼
PREDICTIONS
```

Each prediction is associated with the authenticated user's ID.

Prediction records contain information such as:

* User ID
* Article
* Prediction
* Confidence
* Creation time

Users can therefore view predictions associated specifically with their own account.

---

# 📜 Prediction History

Authenticated users can access their previous predictions.

The history page displays:

* Prediction result
* Confidence score
* Article text
* Date/time of prediction

Predictions are retrieved from MySQL using the logged-in user's ID.

The application also provides a **Clear History** option that deletes predictions associated with the current user.

---

# ⚠️ Confidence Score

The application displays the model's confidence score using the probability estimates returned by Logistic Regression.

A confidence threshold of **60%** is used.

When confidence is below 60%, the application displays:

> Low confidence. Please verify this article using reliable sources.

The confidence score represents the model's prediction confidence and should **not** be interpreted as proof that an article is objectively true.

---

# 🎨 Application Settings

The Streamlit application includes customizable appearance settings.

Users can change:

* Font color
* Background color

The selected settings are stored in:

```text
settings.json
```

so that the chosen appearance can persist between sessions.

---

# 📚 Example Articles

The application contains example articles labelled as examples of:

* Fake News
* Real News

It also provides external links for users who want to explore additional examples.

---

# 🧪 Testing

Beyond the standard train/test evaluation, the model was manually tested using news-style articles that were not directly copied from the training examples.

These tests highlighted an important limitation:

A model can sometimes be **highly confident while still making an incorrect prediction**, especially when a fabricated article uses realistic professional news language.

This is one reason the application includes a low-confidence warning and a disclaimer encouraging independent verification.

---

# 📁 Project Structure

```text
fake-news-detector-ml/
│
├── app.py
│
├── New Start(4).ipynb
│
├── final_model.pkl
│
├── tfidf.pkl
│
├── True.csv
│
├── Fake.csv
│
├── settings.json
│
├── requirements.txt
│
├── .gitignore
│
└── .streamlit/
    └── secrets.toml
```

> `secrets.toml` should not be committed to GitHub.

---

# 🛠️ Technologies Used

| Technology              | Purpose                   |
| ----------------------- | ------------------------- |
| Python                  | Core programming language |
| Pandas                  | Data manipulation         |
| NumPy                   | Numerical operations      |
| Matplotlib              | Data visualization        |
| Seaborn                 | Data visualization        |
| NLTK                    | NLP preprocessing         |
| Scikit-learn            | Machine learning          |
| TF-IDF                  | Text feature extraction   |
| Logistic Regression     | Final deployed classifier |
| Linear SVC              | Model comparison          |
| Random Forest           | Model comparison          |
| Multinomial Naive Bayes | Model comparison          |
| XGBoost                 | Model comparison          |
| RandomizedSearchCV      | Hyperparameter tuning     |
| Joblib                  | Model serialization       |
| Streamlit               | Web application           |
| MySQL                   | Database                  |
| bcrypt                  | Password hashing          |

---

# 🚀 How to Run the Project

## 1. Clone the repository

```bash
https://github.com/sakshamsharma0407/fake-news-classifier.git
cd fake-news-detector-ml
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure MySQL

Create a MySQL database named:

```text
FAKENEWS
```

Create the required user and prediction tables according to the application's database schema.

## 4. Configure Streamlit secrets

Create:

```text
.streamlit/secrets.toml
```

and store your database credentials there.

Example:

```toml
[mysql]
host = "localhost"
user = "root"
password = "YOUR_PASSWORD"
database = "FAKENEWS"
```

Do **not** commit this file to GitHub.

## 5. Run the application

```bash
streamlit run app.py
```

---

# 🔒 Security Notes

Database credentials should never be hard-coded into the source code or committed to a public repository.

Use Streamlit secrets or environment variables for sensitive information such as:

* Database passwords
* API keys
* Other credentials

---

# 📈 Results

The project achieved the following benchmark results on the held-out test set:

```text
Logistic Regression → 98.68% Accuracy
Linear SVC          → 99.38% Accuracy
Random Forest       → 94.79% Accuracy
Multinomial NB      → 93.97% Accuracy
XGBoost             → 99.70% Accuracy
```

Logistic Regression was retained as the deployment model because it provides probability estimates required by the application's confidence-scoring system.

---

# ⚠️ Limitations

This project has several limitations:

* The model learns patterns from the training dataset and does not independently verify facts.
* Articles written in a realistic news style can sometimes be misclassified.
* A high confidence score does not guarantee factual correctness.
* The dataset primarily represents historical news and may not fully represent modern writing styles.
* Real-world performance can differ from the held-out test-set performance.

For important information, users should verify claims through reliable and independent sources.

---

# 🔮 Future Improvements

Possible future improvements include:

* AI-generated explanations for why an article received a particular prediction.
* Training on newer and more diverse news sources.
* Testing on larger independently verified datasets.
* Improved detection of professionally written misinformation.
* More advanced NLP or transformer-based models.
* Deployment of the application online.

---

# 👨‍💻 Author

**Saksham Sharma**

Built as a Machine Learning / NLP project combining model development with a full user-facing application.

---

# ⭐ Conclusion

This project demonstrates an end-to-end machine-learning workflow, starting from raw news datasets and progressing through data cleaning, NLP preprocessing, TF-IDF feature extraction, model comparison, evaluation, hyperparameter tuning, final model selection, model serialization, and deployment through Streamlit.

The final system combines **Machine Learning + NLP + Web Application Development + Authentication + Database Management** in a single application.
