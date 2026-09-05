# Food Recommendation System

A **content-based filtering** recommendation engine for canteen food items, built with **FastAPI** and **scikit-learn**. Provides personalized recommendations based on user order history, new/special items, top-rated items, and popular items.

## Features

- **Personalized Recommendations** — Based on user's past orders using cosine similarity
- **New & Specials** — Today's specials from user's home canteen
- **Top Rated** — Highest rated items across all canteens
- **Popular Items** — Most ordered items across all canteens
- **Content-Based Filtering** — Uses item tags, titles, and categories for similarity
- **FastAPI** — Async REST API with automatic docs

## Quick Start

```bash
git clone https://github.com/ayusharyan1309/food-recommendation-system.git
cd food-recommendation-system
pip install -r requirements.txt
python app.py
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.

## API Endpoint

### Get Recommendations
```
GET /recommendations/{user_id}
```

**Response:**
```json
{
  "personalised_recommendations": [...],
  "new_and_specials": [...],
  "top_rated_items": [...],
  "popular_items": [...]
}
```

## How It Works

```
1. Load food data (title, category, tags, ratings, orders)
         │
2. Create "soup" from tags + title + category
         │
3. CountVectorizer → Cosine Similarity Matrix
         │
4. For a given user:
   ├── Get last 3 orders from order history
   ├── Find similar items via cosine similarity
   ├── Get new/specials from user's home canteen
   ├── Get top-rated items globally
   └── Get most popular items globally
         │
5. Return 4 categories of recommendations
```

## Dataset

| File | Description |
|------|-------------|
| `db/food.csv` | Food items (id, title, canteen, price, orders, category, ratings, tags) |
| `db/orders.csv` | User order history (user_id, food_id, timestamp) |
| `db/users.csv` | Users (id, name, home_canteen) |
| `db/new_and_specials.csv` | Today's specials per canteen |

## Tech Stack

| Component | Technology |
|-----------|------------|
| **API** | FastAPI + Uvicorn |
| **ML** | scikit-learn (CountVectorizer, Cosine Similarity) |
| **Data** | Pandas, NumPy |
| **Language** | Python 3.x |
