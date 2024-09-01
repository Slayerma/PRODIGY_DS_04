import pandas as pd 
import numpy as np 
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

warnings.filterwarnings('ignore')

def load_data(file_path):
    columns_nm = ['Id','Entry','target','text']
    df = pd.read_csv(file_path, encoding='unicode_escape', names=columns_nm)
    return df

def plot_target_distribution(df):
    plt.figure(figsize=(8,6))
    sns.countplot(x='target', data=df, color='#FF00FF')
    plt.title("Count_plot for target data", fontsize=18, c='r')
    plt.ylabel("Total Count series", fontsize=14, c='b')
    plt.xlabel("Target", fontsize=14, c='y')
    plt.show()

    plt.figure(figsize=(8,6))
    sns.histplot(x='target', data=df, kde=True)
    plt.title("Hist_plot for target data", fontsize=18, c='r')
    plt.ylabel("Total Count series", fontsize=14, c='b')
    plt.xlabel("Target", fontsize=14, c='#FFA500')
    plt.show()

def plot_entry_distribution(df):
    plt.figure(figsize=(8,6))
    entry_counts = df['Entry'].value_counts()
    sns.barplot(x=entry_counts.index, y=entry_counts.values)
    plt.title("Count_plot for Entry data", fontsize=18, c='r')
    plt.ylabel("Total Count series", fontsize=14, c='b')
    plt.xlabel("Entry for each Data", fontsize=14, c='#FFA500')
    plt.show()

    plt.figure(figsize=(14,6))
    sns.histplot(x='Entry', data=df, kde=True)
    plt.title("Hist_plot for Entry data", fontsize=18, c='r')
    plt.ylabel("Total Count series", fontsize=16, c='b')
    plt.xlabel("Entry", fontsize=16, c='#FFA500')
    plt.show()

def plot_pairplot(df):
    plt.figure(figsize=(14,12))
    sns.pairplot(df[10:20000], vars=["target", "Entry"])
    plt.show()

def plot_pie_charts(df):
    plt.figure(figsize=(10,10))

    # Pie chart for target
    plt.subplot(2,1,1)
    target_counts = df["target"].value_counts()
    labels = ["1", "2", "3", "4"]
    plt.pie(target_counts, labels=labels, autopct="%1.1f%%")
    plt.legend()
    plt.title("Pie for Target", fontsize=10, c="k")

    # Pie chart for Entry
    plt.subplot(2,1,2)
    entry_counts = df["Entry"].value_counts().head(3)
    labels = ["1", "2", "3"]
    plt.pie(entry_counts, labels=labels, autopct="%1.1f%%")
    plt.legend()
    plt.title("Pie for Entry", fontsize=10, c="k")

    plt.show()

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", "", text)
        tokens = text.split()
        return " ".join(tokens)
    else:
        return ""

def prepare_data(df):
    vectorizer = CountVectorizer(max_features=5000)
    feature = vectorizer.fit_transform(df['text'].apply(preprocess_text))
    return feature.toarray(), vectorizer

def train_test_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=32, test_size=0.25)

    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    nb_score = nb.score(X_test, y_test)
    nb_predictions = nb.predict(X_test)
    nb_report = classification_report(y_test, nb_predictions)
    nb_conf_matrix = confusion_matrix(y_test, nb_predictions)

    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_score = rf_model.score(X_test, y_test)
    rf_predictions = rf_model.predict(X_test)
    rf_report = classification_report(y_test, rf_predictions)
    rf_conf_matrix = confusion_matrix(y_test, rf_predictions)

    # Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    dt_score = dt_model.score(X_test, y_test)
    dt_predictions = dt_model.predict(X_test)
    dt_report = classification_report(y_test, dt_predictions)
    dt_conf_matrix = confusion_matrix(y_test, dt_predictions)

    return {
        "Naive Bayes": (nb_score, nb_report, nb_conf_matrix),
        "Random Forest": (rf_score, rf_report, rf_conf_matrix),
        "Decision Tree": (dt_score, dt_report, dt_conf_matrix)
    }

def plot_confusion_matrices(confusion_matrices, model_names):
    for i, (name, conf_matrix) in enumerate(confusion_matrices.items()):
        plt.figure(figsize=(8,6))
        sns.heatmap(conf_matrix, annot=True, cmap='Greens')
        plt.title(f"Heatmap for {name} model prediction", c='r', fontsize=18)
        plt.show()

def predict_sentiment(text, model, vectorizer):
    processed_text = preprocess_text(text)
    X_new = vectorizer.transform([processed_text])
    predicted_sentiment = model.predict(X_new)
    return predicted_sentiment

def main():
    # Load data
    df = load_data('twitter_training.csv')

    # Plot distributions
    plot_target_distribution(df)
    plot_entry_distribution(df)
    plot_pairplot(df)
    plot_pie_charts(df)

    # Prepare data
    feature_cv, vectorizer = prepare_data(df)
    X = feature_cv[:30000]
    y = df.target[:30000]

    # Train models
    results = train_test_models(X, y)

    # Plot confusion matrices
    plot_confusion_matrices({name: conf_matrix for name, (_, _, conf_matrix) in results.items()}, results.keys())

    # Predictions
    rf_model = RandomForestClassifier()
    rf_model.fit(X, y)  # Assuming the model is trained with the entire dataset for predictions
    print("Predictions with RandomForest:")
    print(predict_sentiment("I love this film", rf_model, vectorizer))
    print(predict_sentiment("I hate this film", rf_model, vectorizer))

    nb_model = MultinomialNB()
    nb_model.fit(X, y)
    print("Predictions with Naive Bayes:")
    print(predict_sentiment("I love this film", nb_model, vectorizer))
    print(predict_sentiment("I hate this film", nb_model, vectorizer))

if __name__ == "__main__":
    main()
