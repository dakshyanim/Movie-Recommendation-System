A content-based movie recommendation app built with Python and Streamlit that enables users to receive the top 30 similar movies by entering a movie name. Recommendations are generated using NLP techniques (TF-IDF + Cosine Similarity). Users can also download results in Excel format.
<br>
Project Description
This system uses metadata (genres, keywords, cast, director, and overview) to find similar movies. The similarity is computed using TF-IDF vectorization and cosine similarity, allowing efficient and accurate recommendations based on content.

Core Idea
Input: A movie name from the dataset

Process: Text vectorization + similarity score calculation

Output: A list of top 30 similar movies with Excel export functionality

Tech Stack
Frontend: Streamlit (Python-based UI)

Backend Logic: Python, Pandas, NLTK, Scikit-learn

File Generation: Pandas + XlsxWriter

Data Storage: Pickled .pkl files (preprocessed)

Dataset Overview
Contains 4803 movies

Each movie has 24 features, including title, cast, crew, genres, overview, etc.

Cleaned and processed to retain relevant fields for content-based filtering

Features
Movie Selection Dropdown: Select a movie from a preloaded list

Top 30 Similar Movies: Based on cosine similarity of metadata

On-Screen Display: Ranked list of recommended movies

Excel File Export: Download recommendations with a custom filename

Fuzzy Matching: Uses difflib.get_close_matches() for typo-tolerant movie name input

Steps Performed
Step 1: Data Cleaning & Preprocessing
Dropped irrelevant columns: homepage, tagline, revenue, etc.

Removed missing values

Extracted year from release_date

Standardized numeric features

Step 2: Feature Engineering
Combined: genres, keywords, cast, director, overview

Created a new column tags for unified textual data

Step 3: Text Normalization
Used NLTK's PorterStemmer

Removed stopwords

Cleaned tokens

Step 4: TF-IDF Vectorization
Converted tags into 5000-length vectors using TfidfVectorizer

Step 5: Cosine Similarity Calculation
Applied cosine similarity on TF-IDF vectors

Created a similarity matrix for all movie pairs

Step 6: Matching & Recommendation
Used fuzzy matching to resolve user input

Retrieved similarity scores

Sorted and returned top 30 results (excluding the selected movie)

Step 7: Export & Optimization
Pickled the cleaned DataFrame and similarity matrix

Loaded preprocessed data during app runtime for faster response

Created Excel in-memory using BytesIO()

Enabled download using Streamlit's download_button()

Limitations
Limited to the dataset (no new or real-time movie info)

Fuzzy matching is basic – may fail for complex typos

No explicit error handling for unmatched titles

Future improvements: real-time APIs, better user input validation, error messages

Requirements
bash
Copy
Edit
streamlit  
pandas  
numpy  
nltk  
scikit-learn  
xlsxwriter
How to Run
Install dependencies

bash
Copy
Edit
pip install streamlit pandas numpy nltk scikit-learn xlsxwriter
Run the app

bash
Copy
Edit
streamlit run app.py


