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
  
    fetch('http://127.0.0.1:5000/process', {
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
      return response.json(); // Parse the JSON from the response
    })
    .then(data => {
      var stockImage = document.getElementById('photo');
      console.log("DATA RECEIVED:", data);
      // Find the image with the src "Prediction.png"
      console.log("ye");
      // Create a new image element
      const newImage = new Image();
      newImage.src = 'plot.png?' + new Date().getTime();
      // stockImage.style.transition = 'opacity 0.1s ease-in-out, transform 0.1s ease-in-out';
      // stockImage.style.opacity = 0;
      // stockImage.style.transform = 'scale(0.8)';
  
      newImage.onload = () => {
        // Fade out the current image
  
        // Wait for the fade-out to complete, then change the src and fade in
        setTimeout(() => {
          stockImage.src = newImage.src; // Update the existing image src once the new image is loaded
          stockImage.style.transition = 'opacity 0.1s ease-in, transform 0.3s ease-in';
          stockImage.style.opacity = 1; // Fade in
          stockImage.style.transform = 'scale(1)'; // Scale to original size
        }, 0); // 0.08s delay to match the fade-out duration
      }
    })
    .catch(error => console.error('Fetch bad error:', error)); // Log any fetch errors to the console
  });
  
  
  const runner = document.getElementById("run_button");
  runner.onclick = function() {
      console.log("disappear");
      var stockImage = document.getElementById('photo');
      stockImage.style.transition = 'opacity 0.15s ease-in-out, transform 0.15s ease-in-out';
      stockImage.style.opacity = 0;
      stockImage.style.transform = 'scale(0.95)';
  };
  