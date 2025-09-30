// Global variable to store our trained TensorFlow.js model
let model;

// Training configuration
const LATENT_DIM = 10;    // Dimension of latent factors (user and movie embeddings)
const EPOCHS = 8;         // Number of training epochs
const BATCH_SIZE = 128;   // Batch size for training

/**
 * Initialization function that runs when the page loads
 * Coordinates data loading, UI setup, and model training
 */
window.onload = async function() {
    try {
        // Step 1: Load and parse the MovieLens dataset
        await loadData();
        updateStatus('Data loaded! Populating dropdowns...');
        
        // Step 2: Populate the user and movie dropdown menus
        populateUserDropdown();
        populateMovieDropdown();
        
        // Step 3: Enable predict button and update status
        document.getElementById('predict-btn').disabled = false;
        updateStatus('Starting model training... This may take a few moments.');
        
        // Step 4: Train the matrix factorization model
        await trainModel();
        
    } catch (error) {
        console.error('Initialization error:', error);
        updateStatus('Error: ' + error.message);
    }
};

/**
 * Populate the user dropdown with available user IDs
 */
function populateUserDropdown() {
    const userSelect = document.getElementById('user-select');
    userSelect.innerHTML = '<option value="">Select a user...</option>';
    
    // Create options for each user (1 to numUsers)
    for (let i = 1; i <= numUsers; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `User ${i}`;
        userSelect.appendChild(option);
    }
}

/**
 * Populate the movie dropdown with movie titles
 */
function populateMovieDropdown() {
    const movieSelect = document.getElementById('movie-select');
    movieSelect.innerHTML = '<option value="">Select a movie...</option>';
    
    // Create options for each movie
    movies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = `${movie.id}. ${movie.title}`;
        movieSelect.appendChild(option);
    });
}

/**
 * Create the Matrix Factorization model architecture
 * This model learns latent factors for users and movies to predict ratings
 */
function createModel() {
    // Input layers for user and movie IDs
    const userInput = tf.input({shape: [1], name: 'userInput'});
    const movieInput = tf.input({shape: [1], name: 'movieInput'});
    
    // Embedding layers: learn dense vector representations
    // Each user/movie gets a LATENT_DIM-dimensional vector
    const userEmbedding = tf.layers.embedding({
        inputDim: numUsers + 1,  // +1 because user IDs start at 1
        outputDim: LATENT_DIM,
        name: 'userEmbedding'
    }).apply(userInput);
    
    const movieEmbedding = tf.layers.embedding({
        inputDim: numMovies + 1, // +1 because movie IDs start at 1
        outputDim: LATENT_DIM,
        name: 'movieEmbedding'
    }).apply(movieInput);
    
    // Reshape embeddings to remove the extra dimension
    const userVector = tf.layers.flatten().apply(userEmbedding);
    const movieVector = tf.layers.flatten().apply(movieEmbedding);
    
    // Dot product of user and movie vectors
    // This represents the predicted rating as the similarity between user and movie in latent space
    const dotProduct = tf.layers.dot({axes: -1}).apply([userVector, movieVector]);
    
    // Create the model with two inputs and one output
    const model = tf.model({
        inputs: [userInput, movieInput],
        outputs: dotProduct
    });
    
    console.log('Model architecture created');
    return model;
}

/**
 * Train the matrix factorization model using the rating data
 */
async function trainModel() {
    try {
        updateStatus('Creating model architecture...');
        
        // Step 1: Create the model
        model = createModel();
        
        // Step 2: Compile the model with optimizer and loss function
        model.compile({
            optimizer: tf.train.adam(0.001),  // Adam optimizer with learning rate 0.001
            loss: 'meanSquaredError'          // MSE loss for regression task
        });
        
        updateStatus('Model compiled. Preparing training data...');
        
        // Step 3: Prepare training data as tensors
        const userTensor = tf.tensor1d(ratings.map(r => r.userId), 'int32');
        const movieTensor = tf.tensor1d(ratings.map(r => r.movieId), 'int32');
        const ratingTensor = tf.tensor1d(ratings.map(r => r.rating));
        
        updateStatus('Starting training...');
        
        // Step 4: Train the model
        const history = await model.fit([userTensor, movieTensor], ratingTensor, {
            epochs: EPOCHS,
            batchSize: BATCH_SIZE,
            validationSplit: 0.1,  // Use 10% of data for validation
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    // Update progress and status after each epoch
                    const progress = ((epoch + 1) / EPOCHS) * 100;
                    document.getElementById('training-progress').value = progress;
                    updateStatus(`Training epoch ${epoch + 1}/${EPOCHS} - Loss: ${logs.loss.toFixed(4)}`);
                }
            }
        });
        
        // Step 5: Clean up tensors to free memory
        userTensor.dispose();
        movieTensor.dispose();
        ratingTensor.dispose();
        
        updateStatus('Model training completed! Ready for predictions.');
        document.getElementById('training-progress').value = 100;
        
        console.log('Training completed. Final loss:', history.history.loss[EPOCHS - 1]);
        
    } catch (error) {
        console.error('Training error:', error);
        updateStatus('Training failed: ' + error.message);
    }
}

/**
 * Predict rating for selected user and movie using the trained model
 */
async function predictRating() {
    const userId = document.getElementById('user-select').value;
    const movieId = document.getElementById('movie-select').value;
    
    if (!userId || !movieId) {
        showResult('Please select both a user and a movie.', 'error');
        return;
    }
    
    if (!model) {
        showResult('Model is not ready yet. Please wait for training to complete.', 'error');
        return;
    }
    
    try {
        showResult('Making prediction...', 'info');
        
        // Create input tensors for the selected user and movie
        const userTensor = tf.tensor1d([parseInt(userId)], 'int32');
        const movieTensor = tf.tensor1d([parseInt(movieId)], 'int32');
        
        // Make prediction
        const prediction = model.predict([userTensor, movieTensor]);
        const rating = await prediction.data();
        const predictedRating = rating[0];
        
        // Clean up tensors
        userTensor.dispose();
        movieTensor.dispose();
        prediction.dispose();
        
        // Display the result
        const movieTitle = movies.find(m => m.id === parseInt(movieId)).title;
        displayPrediction(userId, movieTitle, predictedRating);
        
    } catch (error) {
        console.error('Prediction error:', error);
        showResult('Prediction failed: ' + error.message, 'error');
    }
}

/**
 * Display the prediction result in a user-friendly format
 */
function displayPrediction(userId, movieTitle, predictedRating) {
    // Clamp rating between 1 and 5 (original rating scale)
    const clampedRating = Math.max(1, Math.min(5, predictedRating));
    
    // Create star rating visualization
    const fullStars = Math.floor(clampedRating);
    const partialStar = clampedRating - fullStars;
    let starsHTML = '';
    
    for (let i = 1; i <= 5; i++) {
        if (i <= fullStars) {
            starsHTML += '★';
        } else if (i === fullStars + 1 && partialStar > 0) {
            // Create partial star using CSS gradient
            const percentage = partialStar * 100;
            starsHTML += `☆`;
        } else {
            starsHTML += '☆';
        }
    }
    
    const resultHTML = `
        <div class="prediction-result">
            <h3>Prediction Result</h3>
            <p>User <strong>${userId}</strong> would rate:</p>
            <p><strong>"${movieTitle}"</strong></p>
            <div class="stars">${starsHTML}</div>
            <div class="prediction">
                ${clampedRating.toFixed(1)} / 5.0
            </div>
            <p style="margin-top: 10px; font-size: 0.9em; color: #666;">
                Based on matrix factorization with ${LATENT_DIM} latent factors
            </p>
        </div>
    `;
    
    document.getElementById('result').innerHTML = resultHTML;
}

/**
 * Update training status message
 */
function updateStatus(message) {
    document.getElementById('status').textContent = message;
    console.log('Status:', message);
}

/**
 * Show result message with different types (info, error, success)
 */
function showResult(message, type = 'info') {
    const resultDiv = document.getElementById('result');
    const color = type === 'error' ? '#f44336' : type === 'success' ? '#4caf50' : '#2196f3';
    resultDiv.innerHTML = `<div style="color: ${color}; font-weight: bold;">${message}</div>`;
}
