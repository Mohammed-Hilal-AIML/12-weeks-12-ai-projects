# MovieLens 100K Data

Download **MovieLens 100K** from the official GroupLens page:
https://grouplens.org/datasets/movielens/100k/

The original archive contains `u.data` and `u.item`; the script supports those files directly. If using a CSV version, place `ratings.csv` and `movies.csv` here.

Expected CSV columns:
- `ratings.csv`: `userId,movieId,rating,timestamp`
- `movies.csv`: `movieId,title,genres`

Dataset files are ignored by Git. Download them locally, then run from the project root:
```powershell
py -3.12 movie_recommender.py
```
