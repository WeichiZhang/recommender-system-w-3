// Global variables to store parsed data and dimensions
let movies = [];       // Array to store movie objects with id and title
let ratings = [];      // Array to store rating objects {userId, movieId, rating}
let numUsers = 0;      // Number of unique users in the dataset
let numMovies = 0;     // Number of unique movies in the dataset

/**
 * Main function to load and parse MovieLens 100K dataset
 * This function coordinates the loading of both movie and rating data
 */
async function loadData() {
    try {
        console.log('Loading MovieLens 100K dataset...');
        
        // URLs for the MovieLens 100K dataset files
        const moviesUrl = 'https://raw.githubusercontent.com/dryjins/RecSys-LLMs/main/week3/ml-100k/u.item';
        const ratingsUrl = 'https://raw.githubusercontent.com/dryjins/RecSys-LLMs/main/week3/ml-100k/u.data';
        
        // Fetch both files in parallel for better performance
        const [moviesResponse, ratingsResponse] = await Promise.all([
            fetch(moviesUrl),
            fetch(ratingsUrl)
        ]);
        
        // Check if both requests were successful
        if (!moviesResponse.ok || !ratingsResponse.ok) {
            throw new Error('Failed to load dataset files');
        }
        
        // Get the text content from both responses
        const moviesText = await moviesResponse.text();
        const ratingsText = await ratingsResponse.text();
        
        // Parse the raw text data into structured JavaScript objects
        movies = parseItemData(moviesText);
        ratings = parseRatingData(ratingsText);
        
        // Calculate dimensions for our matrix factorization model
        numUsers = Math.max(...ratings.map(r => r.userId));
        numMovies = movies.length;
        
        console.log(`Data loaded successfully: ${numUsers} users, ${numMovies} movies, ${ratings.length} ratings`);
        
        return {
            movies,
            ratings,
            numUsers,
            numMovies
        };
        
    } catch (error) {
        console.error('Error loading data:', error);
        throw error;
    }
}

/**
 * Parse the movie/item data file (u.item)
 * Each line contains: movieId|title|releaseDate|... 
 * We extract movieId and title for our recommender
 */
function parseItemData(text) {
    const lines = text.split('\n').filter(line => line.trim());
    const movieData = [];
    
    for (const line of lines) {
        const parts = line.split('|');
        if (parts.length >= 2) {
            const movieId = parseInt(parts[0]);
            // Clean up the title - remove year and any extra spaces
            let title = parts[1].trim();
            title = title.replace(/\s*\(\d{4}\)$/, ''); // Remove year in parentheses
            
            movieData.push({
                id: movieId,
                title: title
            });
        }
    }
    
    console.log(`Parsed ${movieData.length} movies`);
    return movieData;
}

/**
 * Parse the rating data file (u.data)
 * Each line contains: userId movieId rating timestamp
 * We extract userId, movieId, and rating for training our model
 */
function parseRatingData(text) {
    const lines = text.split('\n').filter(line => line.trim());
    const ratingData = [];
    
    for (const line of lines) {
        const parts = line.split('\t');
        if (parts.length >= 3) {
            ratingData.push({
                userId: parseInt(parts[0]),
                movieId: parseInt(parts[1]),
                rating: parseFloat(parts[2])
            });
        }
    }
    
    console.log(`Parsed ${ratingData.length} ratings`);
    return ratingData;
}
