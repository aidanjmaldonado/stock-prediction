// script.js
document.getElementById('advanced_button').addEventListener('click', function(event) {
    event.preventDefault();
    const advancedOptions = document.getElementById('advanced_options');
    if (advancedOptions.style.display === 'none' || advancedOptions.style.display === '') {
        advancedOptions.style.display = 'block';
    } else {
        advancedOptions.style.display = 'none';
    }
});

document.getElementById('user_input_form').addEventListener('submit', function(event) {
    event.preventDefault();
    const ticker = document.getElementById('ticker_input').value;
    const margin = document.getElementById('margin_input').value;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ data: [ticker, margin] })
    })
    .then(response => response.json())
    .then(data => {
        alert('Prediction: ' + data.prediction);
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
