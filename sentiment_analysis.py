import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import gc
import pickle

def run_sentiment_analysis():
    os.makedirs("static/images", exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'amazon_alexa.tsv')
    tfidf_path = os.path.join(script_dir, 'tfidf_matrix.pkl')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")

    # Load the dataset
    df_alexa = pd.read_csv(data_path, sep='\t')

    # Clean the dataset: Remove rows with missing or invalid values
    df_alexa = df_alexa.dropna(subset=['verified_reviews', 'feedback', 'rating', 'variation'])
    df_alexa = df_alexa[df_alexa['verified_reviews'].str.strip() != '']
    df_alexa = df_alexa[df_alexa['feedback'].isin([0, 1])]
    df_alexa = df_alexa[df_alexa['rating'].isin([1, 2, 3, 4, 5])]

    # Ensure feedback and rating are integer types
    df_alexa['feedback'] = df_alexa['feedback'].astype(int)
    df_alexa['rating'] = df_alexa['rating'].astype(int)

    # Use the entire cleaned dataset
    print(f"Total reviews after cleaning: {len(df_alexa)}")
    print("Feedback distribution before plotting:")
    print(df_alexa['feedback'].value_counts())

    positive = df_alexa[df_alexa['feedback'] == 1]
    negative = df_alexa[df_alexa['feedback'] == 0]

    # Generate feedback distribution plot
    plt.figure(figsize=(8, 6))
    sns.countplot(x='feedback', data=df_alexa)
    plt.xlabel('Feedback')
    plt.ylabel('Count')
    plt.title('Distribution of Feedback')
    plt.xticks(ticks=[0, 1], labels=['Negative (0)', 'Positive (1)'])
    plt.savefig('static/images/feedback_distribution.png', bbox_inches='tight')
    plt.close()
    gc.collect()

    # Generate rating distribution plot
    plt.figure(figsize=(8, 6))
    sns.countplot(x='rating', data=df_alexa)
    plt.xlabel('Rating (Stars)')
    plt.ylabel('Count')
    plt.title('Distribution of Ratings')
    plt.savefig('static/images/rating_distribution.png', bbox_inches='tight')
    plt.close()
    gc.collect()

    # Generate feedback by variation plot
    plt.figure(figsize=(12, 6))
    sns.countplot(x='variation', hue='feedback', data=df_alexa)
    plt.xlabel('Product Variation')
    plt.ylabel('Count')
    plt.title('Feedback Distribution by Product Variation')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Feedback', labels=['Negative (0)', 'Positive (1)'])
    plt.savefig('static/images/feedback_by_variation.png', bbox_inches='tight')
    plt.close()
    gc.collect()

    # Debug: Print rating distribution by feedback
    print("Rating distribution for Negative feedback (0):")
    print(df_alexa[df_alexa['feedback'] == 0]['rating'].value_counts())
    print("Rating distribution for Positive feedback (1):")
    print(df_alexa[df_alexa['feedback'] == 1]['rating'].value_counts())

    # Generate rating vs feedback plot (use count plot instead of violin plot)
    plt.figure(figsize=(8, 6))
    sns.countplot(x='rating', hue='feedback', data=df_alexa)
    plt.xlabel('Rating (Stars)')
    plt.ylabel('Count')
    plt.title('Rating Distribution by Feedback')
    plt.legend(title='Feedback', labels=['Negative (0)', 'Positive (1)'])
    plt.savefig('static/images/rating_vs_feedback.png', bbox_inches='tight')
    plt.close()
    gc.collect()

    # Debug: Print rating distribution by feedback
    print("Rating distribution for Negative feedback (0):")
    print(df_alexa[df_alexa['feedback'] == 0]['rating'].value_counts())
    print("Rating distribution for Positive feedback (1):")
    print(df_alexa[df_alexa['feedback'] == 1]['rating'].value_counts())

   # Generate rating vs feedback plot (use count plot instead of violin plot)
    plt.figure(figsize=(8, 6))
    sns.countplot(x='rating', hue='feedback', data=df_alexa)
    plt.xlabel('Rating (Stars)')
    plt.ylabel('Count')
    plt.title('Rating Distribution by Feedback')
    plt.legend(title='Feedback', labels=['Negative (0)', 'Positive (1)'])
    plt.savefig('static/images/rating_vs_feedback.png', bbox_inches='tight')
    plt.close()
    gc.collect()
    # Extract text and labels
    text_data = df_alexa['verified_reviews'].astype(str).tolist()
    labels = df_alexa['feedback'].tolist()

    # Verify lengths match
    if len(text_data) != len(labels):
        raise ValueError(f"Mismatch in number of samples: text_data has {len(text_data)} samples, labels has {len(labels)} samples")

    # Load or compute TF-IDF matrix
    if os.path.exists(tfidf_path):
        with open(tfidf_path, 'rb') as f:
            tfidf_data = pickle.load(f)
        X_tfidf = tfidf_data['X_tfidf']
        tfidf = tfidf_data['tfidf']
        if X_tfidf.shape[0] != len(text_data):
            tfidf = TfidfVectorizer(max_features=1000)
            X_tfidf = tfidf.fit_transform(text_data).toarray()
            with open(tfidf_path, 'wb') as f:
                pickle.dump({'X_tfidf': X_tfidf, 'tfidf': tfidf}, f)
    else:
        tfidf = TfidfVectorizer(max_features=1000)
        X_tfidf = tfidf.fit_transform(text_data).toarray()
        with open(tfidf_path, 'wb') as f:
            pickle.dump({'X_tfidf': X_tfidf, 'tfidf': tfidf}, f)

    num_features = X_tfidf.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42)

    y_train = np.array(y_train).astype('float32')
    y_test = np.array(y_test).astype('float32')

    # Use a simpler model
    ANN_classifier = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(num_features,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    ANN_classifier.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

    epochs = 1
    epochs_hist = ANN_classifier.fit(X_train, y_train,
                                    epochs=epochs,
                                    batch_size=32,
                                    validation_split=0.2,
                                    verbose=0)

    y_pred_train = ANN_classifier.predict(X_train, verbose=0)
    y_pred_train = (y_pred_train > 0.5).astype(int)
    cm_train = confusion_matrix(y_train, y_pred_train)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Training Set')
    plt.savefig('static/images/cm_train.png', bbox_inches='tight')
    plt.close()
    gc.collect()

    y_pred_test = ANN_classifier.predict(X_test, verbose=0)
    y_pred_test = (y_pred_test > 0.5).astype(int)
    cm_test = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Test Set')
    plt.savefig('static/images/cm_test.png', bbox_inches='tight')
    plt.close()
    gc.collect()

    report_train = classification_report(y_train, y_pred_train, output_dict=True)
    report_test = classification_report(y_test, y_pred_test, output_dict=True)

    # Include all test set predictions
    predictions = {
        'y_pred_test': y_pred_test.tolist(),
        'y_test': y_test.tolist()
    }

    summary = {
        'num_reviews': len(df_alexa),
        'num_positive': len(positive),
        'num_negative': len(negative),
        'num_features': num_features,
        'train_accuracy': epochs_hist.history['accuracy'][-1],
        'val_accuracy': epochs_hist.history['val_accuracy'][-1],
        'test_precision_0': report_test['0.0']['precision'],
        'test_recall_0': report_test['0.0']['recall'],
        'test_f1_0': report_test['0.0']['f1-score'],
        'test_precision_1': report_test['1.0']['precision'],
        'test_recall_1': report_test['1.0']['recall'],
        'test_f1_1': report_test['1.0']['f1-score']
    }

    return {
        'summary': summary,
        'report_train': report_train,
        'report_test': report_test,
        'predictions': predictions
    }

# Precompute results when running this script directly
if __name__ == "__main__":
    print("Precomputing sentiment analysis results...")
    results = run_sentiment_analysis()
    with open('static/results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Results saved to static/results.pkl")