from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# ============================================================
# Setup
# ============================================================
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

# ============================================================
# Schema
# ============================================================
class RecommendRequest(BaseModel):
    user_interests: list[str]
    top_n: int = 10
    pool_size: int = 30
    city_filter: str = None

# ============================================================
# Endpoint
# ============================================================
@app.post("/recommend")
def recommend(request: RecommendRequest):
    if not request.user_interests:
        raise HTTPException(status_code=400, detail="user_interests فاضلة")

    user_vector = mlb.transform([request.user_interests])
    scores = cosine_similarity(user_vector, category_matrix).flatten()

    temp_df = df.copy()
    temp_df["similarity_score"] = scores

    if request.city_filter:
        temp_df = temp_df[temp_df["city_en"].str.lower() == request.city_filter.lower()]

    # أحسن pool_size مكان مناسبة
    top_pool = temp_df[temp_df["similarity_score"] > 0]
    top_pool = top_pool.sort_values("similarity_score", ascending=False).head(request.pool_size)

    # اختيار top_n عشوائي من الـ pool
    recommended = top_pool.sample(
        n=min(request.top_n, len(top_pool)),
        random_state=None
    )

    return {
        "recommended_places": [
            {
                "place_id": row["place_id"],
                "score": round(float(row["similarity_score"]), 4)
            }
            for _, row in recommended.iterrows()
        ]
    }