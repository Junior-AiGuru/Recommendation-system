from fastapi import FastAPI, HTTPException, Request
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Setup
app = FastAPI()

df = pd.read_excel("Places_Dataset.xlsx")
df["categories_list"] = df["categories"].apply(
    lambda x: [c.strip() for c in x.split("|")] if pd.notna(x) else []
)

mlb = MultiLabelBinarizer(classes=[
    "Restaurants", "Cafe", "Bakery", "Seafood", "Street Food",
    "Entertainment", "Shopping", "Nightlife", "Music", "Arts & Crafts",
    "Nature", "Park", "Beaches & Water", "Waterfront", "Outdoor",
    "History & Antiquities", "Culture", "Mosques & Churches", "Tourism"
])
category_matrix = mlb.fit_transform(df["categories_list"])

def clean(value):
    # Avoid returning NaN or infinite values in the response
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


# Endpoint

@app.post("/webhook")
async def webhook(request: Request):
    body = await request.json()

    user_interests = body.get("user_interests", [])
    top_n = body.get("top_n", 300)
    pool_size = body.get("pool_size", 300)

    if not user_interests:
        raise HTTPException(status_code=400, detail="user_interests فاضلة")

    user_vector = mlb.transform([user_interests])
    scores = cosine_similarity(user_vector, category_matrix).flatten()

    temp_df = df.copy()
    temp_df["similarity_score"] = scores

    top_pool = temp_df[temp_df["similarity_score"] > 0]
    top_pool = top_pool.sort_values("similarity_score", ascending=False).head(pool_size)

    recommended = top_pool.sample(
        n=min(top_n, len(top_pool)),
        random_state=None
    )

    return {
        "recommended_places": [
            {
                "place_id": clean(row["place_id"]),
                "name": clean(row["name"]),
                "city": clean(row["city"]),
                "city_en": clean(row["city_en"]),
                "categories": clean(row["categories"]),
                "general_category": clean(row["general_category"]),
                "rating": clean(row["rating"]),
                "reviews_count": int(row["reviews_count"]) if pd.notna(row["reviews_count"]) else None,
                "address": clean(row["address"]),
                "description": clean(row["description"]),
                "photo_url": clean(row["photo_url"]),
                "opening_hours": clean(row["opening_hours"]),
                "lat": float(row["lat"]) if pd.notna(row["lat"]) else None,
                "lng": float(row["lng"]) if pd.notna(row["lng"]) else None,
                "score": round(float(row["similarity_score"]), 4)
            }
            for _, row in recommended.iterrows()
        ]
    }