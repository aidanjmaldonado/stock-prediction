document.getElementById('advanced_button').addEventListener('click', () => {
    // event.preventDefault();
    const advanced_options = document.getElementById('advanced_options');
    if (advanced_options.style.visibility === 'hidden') {
        advanced_options.style.visibility = 'visible';
    } else {
        advanced_options.style.visibility = 'hidden';
    }
});



// document.getElementById('user_input_form').addEventListener('submit', () => {
//     // event.preventDefault(); // Delete this
//     const ticker = document.getElementById('ticker_input').value;
//     const margin = document.getElementById('margin_input').value;

//     fetch('/predict', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json'
//         },
//         body: JSON.stringify({ data: [ticker, margin] })
//     })
//     .then(response => response.json())
//     .then(data => {
//         alert('Prediction: ' + data.prediction);
//     })
//     .catch(error => {
//         console.error('Error:', error);
//     });
// });

// document.getElementById('grid_container').addEventListener('submit', function(e) {
//     e.preventDefault(); // Prevent the default form submission
    
//     var formData = new FormData(this);
    
//     console.log("Submitting form data...");
    
//     fetch('https://docs.google.com/forms/d/e/1FAIpQLSd7kOgH3G07jp8ibWOA-UfFFvNNsapY8QwiI7v4_euxrZ1v5w/formResponse', {
//         method: 'POST',
//         body: formData
//     })
//     .then(response => {
//         if (!response.ok) {
//             throw new Error('Network response was not ok: ' + response.statusText);
//         }
//         console.log("Form submitted successfully");
//         document.getElementById('ticker_input').value = ''; // Clear form inputs if needed
//         document.getElementById('margin_input').value = '';
//     })
//     .catch(error => console.error('Fetch error:', error)); // Log any fetch errors to the console
// });
