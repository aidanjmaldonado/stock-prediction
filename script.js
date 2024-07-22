// document.getElementById('advanced_button').addEventListener('click', () => {
//     // event.preventDefault();
//     const advanced_options = document.getElementById('advanced_options');
//     if (advanced_options.style.visibility === 'hidden') {
//         advanced_options.style.visibility = 'visible';
//     } else {
//         advanced_options.style.visibility = 'hidden';
//     }
// });


document.getElementById('grid_container').addEventListener('submit', function(e) {
    e.preventDefault(); // Prevent the default form submission
  
    var formData = new FormData(this);
  
    console.log("Submitting form data...");
  
    fetch('https://google.com', {
      method: 'POST',
      body: formData
    })
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok: ' + response.statusText);
      }
      console.log("Form submitted successfully");
      document.getElementById('ticker_input').value = ''; // Clear form inputs if needed
      document.getElementById('margin_input').value = '';
    })
    .catch(error => console.error('Fetch error:', error)); // Log any fetch errors to the console
  });
