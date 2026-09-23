# Week 07 — Movie Recommendation Engine

Movie recommendation system following the Week 07 manual: content-based filtering, user-based collaborative filtering, Surprise SVD, a 70/30 hybrid recommender, and Precision@10 / Recall@10 evaluation.

## Structure
```text
Week_07_Movie_Recommendation_Engine/
├── movie_recommender.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/README.md
├── models/.gitkeep
└── outputs/README.md
```

## Requirements
- Windows
- **Python 3.12**
- MovieLens 100K dataset

## Install without venv
```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 movie_recommender.py
```

## Pipeline
1. Load MovieLens ratings and movies and inspect the rating distribution.
2. Build a user-item matrix and report sparsity.
3. Content-based TF-IDF + cosine similarity with `recommend_similar(movie_title, n=10)`.
4. User-based collaborative filtering using the top 5 similar users.
5. Surprise SVD with cross-validation over `n_factors`.
6. Hybrid score = **70% SVD + 30% content similarity**.
7. Compare methods with Precision@10 and Recall@10.

The manual uses RMSE < 0.93 as a target; the script reports the actual result and does not guarantee a score.

## Outputs
Generated files include `rating_distribution.png`, `user_item_heatmap.png`, `evaluation_summary.csv`, `svd_summary.csv`, and `models/svd_model.pkl`.

## GitHub
Repository name: `Week_07_Movie_Recommendation_Engine`

Description: `Movie recommendation engine using content-based filtering, collaborative filtering, SVD matrix factorization, and a hybrid recommender with Precision@10 and Recall@10 evaluation.`
