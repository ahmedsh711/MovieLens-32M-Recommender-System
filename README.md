# MovieLens AI Recommender

A hybrid movie recommendation system built using the MovieLens dataset. This project combines Collaborative Filtering (SVD) and Content-Based Filtering (TF-IDF) to provide personalized movie suggestions and find similar movies.

## 🚀 Features

- **Personalized Recommendations**: Suggests movies for a specific user based on their historical ratings using SVD (Singular Value Decomposition).
- **Find Similar Movies**: Recommends movies similar to a given title using Content-Based Filtering (TF-IDF on movie metadata).
- **Interactive UI**: A user-friendly web interface built with Streamlit.

## 📂 Project Structure

```
├── data/               # Raw and processed data files
├── models/             # Saved model artifacts (generated after training)
├── Notebooks/          # Jupyter notebooks for exploration
├── src/                # Source code
│   ├── data_processor.py   # Data loading and preprocessing
│   ├── inference.py        # Recommendation logic
│   ├── model_trainer.py    # Model training functions
│   └── streamlit_app.py    # Streamlit web application
├── train.py            # Main script to train and save models
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 🛠️ Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    # Linux/Mac
    source venv/bin/activate
    # Windows
    venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

### 1. Train the Model
Before running the application, you must train the models and generate the necessary artifacts.

```bash
python train.py
```
This command processes the data, trains the SVD and TF-IDF models, and saves them to the `models/` directory as `recommender_artifacts.pkl`.

### 2. Run the Application
Launch the Streamlit web interface:

```bash
streamlit run src/streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`.

## 🧠 Model Details

- **Collaborative Filtering**: Uses `TruncatedSVD` to reduce the dimensionality of the user-item matrix, capturing latent factors that explain user preferences.
- **Content-Based Filtering**: Uses `TfidfVectorizer` on movie genres and tags to compute cosine similarity between movies.
