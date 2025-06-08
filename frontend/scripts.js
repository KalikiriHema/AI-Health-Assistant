async function analyzeSymptoms() {
  const symptoms = document.getElementById('symptoms').value.trim();

  if (!symptoms) {
      alert('Please enter your symptoms.');
      return;
  }

  try {
      const response = await fetch('/predict', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
          },
          body: JSON.stringify({ symptoms: symptoms })
      });

      const data = await response.json();
      displayResults(data.possible_conditions);
  } catch (error) {
      console.error('Error:', error);
      alert('An error occurred. Please try again.');
  }
}

function displayResults(conditions) {
  const resultsDiv = document.getElementById('results');
  const conditionsDiv = document.getElementById('conditions');

  conditionsDiv.innerHTML = '';

  if (conditions.length === 0) {
      const noResultsElement = document.createElement('div');
      noResultsElement.className = 'no-results';
      noResultsElement.innerHTML = '<p><strong>No matching conditions found.</strong></p>';
      conditionsDiv.appendChild(noResultsElement);
  } else {
      const hasSerious = conditions.some(c => c.serious);

      if (hasSerious) {
          const warningElement = document.createElement('div');
          warningElement.className = 'serious-warning';
          warningElement.innerHTML = '⚠️ <strong>URGENT:</strong> Some conditions listed may require immediate medical attention.';
          conditionsDiv.appendChild(warningElement);
      }

      conditions.forEach(condition => {
          const conditionElement = document.createElement('div');
          conditionElement.className = condition.serious ? 'condition serious' : 'condition normal';
          conditionElement.innerHTML = `
              <strong>${condition.condition}</strong><br>
              Confidence Level: ${condition.confidence}<br>
              <p>${condition.explanation}</p>
          `;
          conditionsDiv.appendChild(conditionElement);
      });
  }

  resultsDiv.style.display = 'block';
}