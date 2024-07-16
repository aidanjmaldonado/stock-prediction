document.getElementById('advanced_button').addEventListener('click', () => {
    // event.preventDefault();
    const advanced_options = document.getElementById('advanced_options');
    if (advanced_options.style.visibility === 'hidden') {
        advanced_options.style.visibility = 'visible';
    } else {
        advanced_options.style.visibility = 'hidden';
    }
});



document.getElementById('user_input_form').addEventListener('submit', () => {
    // event.preventDefault(); // Delete this
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
