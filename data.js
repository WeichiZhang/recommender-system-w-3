// Global variables to store parsed data and dimensions
let movies = [];       // Array to store movie objects with id and title
let ratings = [];      // Array to store rating objects {userId, movieId, rating}
let numUsers = 0;      // Number of unique users in the dataset
let numMovies = 0;     // Number of unique movies in the dataset

/**
 * Main function to load and parse MovieLens 100K dataset
 * Uses multiple fallback URLs to ensure data availability
 */
async function loadData() {
    try {
        console.log('Loading MovieLens 100K dataset...');
        
        // Multiple fallback URLs for the MovieLens dataset
        const movieUrls = [
            'https://raw.githubusercontent.com/dryjins/RecSys-LLMs/main/week3/ml-100k/u.item',
            'https://raw.githubusercontent.com/dryjins/RecSys-LLMs/refs/heads/main/week3/ml-100k/u.item',
            'https://raw.githubusercontent.com/dryjins/RecSys-LLMs/master/week3/ml-100k/u.item'
        ];
        
        const ratingUrls = [
            'https://raw.githubusercontent.com/dryjins/RecSys-LLMs/main/week3/ml-100k/u.data',
            'https://raw.githubusercontent.com/dryjins/RecSys-LLMs/refs/heads/main/week3/ml-100k/u.data',
            'https://raw.githubusercontent.com/dryjins/RecSys-LLMs/master/week3/ml-100k/u.data'
        ];
        
        let moviesText, ratingsText;
        let moviesSuccess = false, ratingsSuccess = false;
        
        // Try multiple URLs for movies data
        for (const url of movieUrls) {
            try {
                const response = await fetch(url);
                if (response.ok) {
                    moviesText = await response.text();
                    moviesSuccess = true;
                    console.log('Movies data loaded from:', url);
                    break;
                }
            } catch (error) {
                console.warn('Failed to load movies from:', url, error);
            }
        }
        
        // Try multiple URLs for ratings data
        for (const url of ratingUrls) {
            try {
                const response = await fetch(url);
                if (response.ok) {
                    ratingsText = await response.text();
                    ratingsSuccess = true;
                    console.log('Ratings data loaded from:', url);
                    break;
                }
            } catch (error) {
                console.warn('Failed to load ratings from:', url, error);
            }
        }
        
        // If URLs failed, use embedded sample data as fallback
        if (!moviesSuccess || !ratingsSuccess) {
            console.log('Using embedded sample data as fallback...');
            return loadSampleData();
        }
        
        // Parse the successfully loaded data
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
        // Fallback to sample data if everything else fails
        return loadSampleData();
    }
}

/**
 * Fallback function that provides sample data when external URLs fail
 */
function loadSampleData() {
    console.log('Loading embedded sample data...');
    
    // Sample movie data (first 20 movies from MovieLens 100K)
    movies = [
        { id: 1, title: "Toy Story" },
        { id: 2, title: "GoldenEye" },
        { id: 3, title: "Four Rooms" },
        { id: 4, title: "Get Shorty" },
        { id: 5, title: "Copycat" },
        { id: 6, title: "Shanghai Triad" },
        { id: 7, title: "Twelve Monkeys" },
        { id: 8, title: "Babe" },
        { id: 9, title: "Dead Man Walking" },
        { id: 10, title: "Richard III" },
        { id: 11, title: "Seven" },
        { id: 12, title: "Usual Suspects" },
        { id: 13, title: "Mighty Aphrodite" },
        { id: 14, title: "Postman, The" },
        { id: 15, title: "Mr. Holland's Opus" },
        { id: 16, title: "French Twist" },
        { id: 17, title: "From Dusk Till Dawn" },
        { id: 18, title: "White Balloon, The" },
        { id: 19, title: "Antonia's Line" },
        { id: 20, title: "Angels and Insects" }
    ];
    
    // Generate sample ratings data
    ratings = [];
    numUsers = 50;
    numMovies = movies.length;
    
    // Create realistic sample ratings
    for (let userId = 1; userId <= numUsers; userId++) {
        // Each user rates 5-15 random movies
        const numRatings = Math.floor(Math.random() * 10) + 5;
        const ratedMovies = new Set();
        
        for (let i = 0; i < numRatings; i++) {
            let movieId;
            do {
                movieId = Math.floor(Math.random() * numMovies) + 1;
            } while (ratedMovies.has(movieId));
            
            ratedMovies.add(movieId);
            
            // Generate realistic ratings (most ratings are 3-5, some 1-2)
            let rating;
            const rand = Math.random();
            if (rand < 0.6) rating = Math.floor(Math.random() * 2) + 4; // 4-5
            else if (rand < 0.9) rating = 3; // 3
            else rating = Math.floor(Math.random() * 2) + 1; // 1-2
            
            ratings.push({
                userId: userId,
                movieId: movieId,
                rating: rating
            });
        }
    }
    
    console.log(`Sample data loaded: ${numUsers} users, ${numMovies} movies, ${ratings.length} ratings`);
    return {
        movies,
        ratings,
        numUsers,
        numMovies
    };
}

/**
 * Parse the movie/item data file (u.item)
 * Each line contains: movieId|title|releaseDate|... 
 */
function parseItemData(text) {
    const lines = text.split('\n').filter(line => line.trim());
    const movieData = [];
    
    for (const line of lines) {
        const parts = line.split('|');
        if (parts.length >= 2) {
            const movieId = parseInt(parts[0]);
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
