// static/js/scripts.js
document.getElementById('analyze-btn').addEventListener('click', function() {
    // Show loading spinner
    document.getElementById('loading').classList.remove('d-none');
    document.getElementById('results').classList.add('d-none');
    document.getElementById('loading-message').innerText = "Running sentiment analysis... Please wait.";

    // Start the analysis
    fetch('/analyze', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        // Start polling for status
        pollStatus();
    })
    .catch(error => {
        console.error('Error starting analysis:', error);
        document.getElementById('loading-message').innerText = "Error starting analysis. Please try again.";
    });
});

function pollStatus() {
    fetch('/status')
    .then(response => response.json())
    .then(data => {
        document.getElementById('loading-message').innerText = data.message;
        if (data.status === 'completed') {
            // Hide loading spinner and show results
            document.getElementById('loading').classList.add('d-none');
            document.getElementById('results').classList.remove('d-none');

            // Populate results
            const results = data.results;
            document.getElementById('num-reviews').innerText = results.summary.num_reviews;
            document.getElementById('num-positive').innerText = results.summary.num_positive;
            document.getElementById('num-negative').innerText = results.summary.num_negative;
            document.getElementById('train-accuracy').innerText = (results.summary.train_accuracy * 100).toFixed(2);
            document.getElementById('val-accuracy').innerText = (results.summary.val_accuracy * 100).toFixed(2);

            document.getElementById('train-precision-0').innerText = results.report_train['0.0'].precision.toFixed(2);
            document.getElementById('train-recall-0').innerText = results.report_train['0.0'].recall.toFixed(2);
            document.getElementById('train-f1-0').innerText = results.report_train['0.0']['f1-score'].toFixed(2);
            document.getElementById('train-support-0').innerText = results.report_train['0.0'].support;

            document.getElementById('train-precision-1').innerText = results.report_train['1.0'].precision.toFixed(2);
            document.getElementById('train-recall-1').innerText = results.report_train['1.0'].recall.toFixed(2);
            document.getElementById('train-f1-1').innerText = results.report_train['1.0']['f1-score'].toFixed(2);
            document.getElementById('train-support-1').innerText = results.report_train['1.0'].support;

            document.getElementById('test-precision-0').innerText = results.report_test['0.0'].precision.toFixed(2);
            document.getElementById('test-recall-0').innerText = results.report_test['0.0'].recall.toFixed(2);
            document.getElementById('test-f1-0').innerText = results.report_test['0.0']['f1-score'].toFixed(2);
            document.getElementById('test-support-0').innerText = results.report_test['0.0'].support;

            document.getElementById('test-precision-1').innerText = results.report_test['1.0'].precision.toFixed(2);
            document.getElementById('test-recall-1').innerText = results.report_test['1.0'].recall.toFixed(2);
            document.getElementById('test-f1-1').innerText = results.report_test['1.0']['f1-score'].toFixed(2);
            document.getElementById('test-support-1').innerText = results.report_test['1.0'].support;

            const predictionsBody = document.getElementById('predictions-body');
            predictionsBody.innerHTML = '';
            for (let i = 0; i < 10; i++) {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${i + 1}</td>
                    <td>${results.predictions.y_pred_test[i] === 1 ? 'Positive' : 'Negative'}</td>
                    <td>${results.predictions.y_test[i] === 1 ? 'Positive' : 'Negative'}</td>
                `;
                predictionsBody.appendChild(row);
            }
        } else if (data.status === 'failed') {
            document.getElementById('loading-message').innerText = data.message;
        } else {
            // Continue polling
            setTimeout(pollStatus, 2000); // Poll every 2 seconds
        }
    })
    .catch(error => {
        console.error('Error checking status:', error);
        document.getElementById('loading-message').innerText = "Error checking status. Please try again.";
    });
}

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});