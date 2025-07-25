from flask import Flask, render_template, request, jsonify, send_file
import pickle
import pandas as pd
import io

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze_page')
def index():
    return render_template('index.html', results=None)

@app.route('/analyze', methods=['POST'])
def analyze():
    # Load precomputed results
    with open('static/results.pkl', 'rb') as f:
        results = pickle.load(f)
    return render_template('index.html', results=results)

@app.route('/status', methods=['GET'])
def check_status():
    return jsonify({"status": "completed", "message": "Results precomputed", "results": None})

@app.route('/download', methods=['GET'])
def download_report():
    # Load precomputed results
    with open('static/results.pkl', 'rb') as f:
        results = pickle.load(f)

    # Prepare data for the CSV
    summary = results['summary']
    report_test = results['report_test']
    predictions = results['predictions']

    # Create a dictionary for the summary and classification report
    summary_data = {
        'Total Reviews': summary['num_reviews'],
        'Positive Reviews': summary['num_positive'],
        'Negative Reviews': summary['num_negative'],
        'Number of Features': summary['num_features'],
        'Training Accuracy (%)': summary['train_accuracy'] * 100,
        'Validation Accuracy (%)': summary['val_accuracy'] * 100,
        'Test Precision (Negative)': summary['test_precision_0'],
        'Test Recall (Negative)': summary['test_recall_0'],
        'Test F1-Score (Negative)': summary['test_f1_0'],
        'Test Precision (Positive)': summary['test_precision_1'],
        'Test Recall (Positive)': summary['test_recall_1'],
        'Test F1-Score (Positive)': summary['test_f1_1']
    }

    # Create a DataFrame for the summary
    summary_df = pd.DataFrame([summary_data])

    # Create a DataFrame for the predictions
    predictions_df = pd.DataFrame({
        'Sample': range(1, len(predictions['y_pred_test']) + 1),
        'Predicted Label': ['Positive' if pred == 1 else 'Negative' for pred in predictions['y_pred_test']],
        'Actual Label': ['Positive' if actual == 1 else 'Negative' for actual in predictions['y_test']]
    })

    # Write to a CSV buffer
    output = io.StringIO()
    output.write("Analysis Summary\n")
    summary_df.to_csv(output, index=False)
    output.write("\nTest Set Predictions\n")
    predictions_df.to_csv(output, index=False)

    # Create a BytesIO buffer to send the file
    output.seek(0)
    buffer = io.BytesIO(output.getvalue().encode('utf-8'))

    # Send the file for download
    return send_file(
        buffer,
        as_attachment=True,
        download_name='sentiment_analysis_report.csv',
        mimetype='text/csv'
    )

if __name__ == '__main__':
    app.run(debug=True)