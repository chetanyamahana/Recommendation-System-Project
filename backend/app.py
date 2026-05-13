from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import sys

# Ensure backend directory is in path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommendation import get_top_k
from llm import generate_response, init_llm

app = FastAPI(title="ShopBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and data
embedding_model = None
product_embeddings = None
products_df = None

class RecommendRequest(BaseModel):
    query: str
    max_price: Optional[float] = None
    category: Optional[str] = None
    rating: Optional[float] = None
    top_k: int = 5

@app.on_event("startup")
def startup_event():
    global embedding_model, product_embeddings, products_df
    
    # Load embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load data
    print("Loading dataset and embeddings...")
    df_path = os.path.join(os.path.dirname(__file__), "../dataset/products_cleaned.csv")
    npy_path = os.path.join(os.path.dirname(__file__), "../models/embeddings.npy")
    
    if not os.path.exists(df_path) or not os.path.exists(npy_path):
        print("Warning: Data or embeddings not found. Please run generate_embeddings.py first.")
    else:
        products_df = pd.read_csv(df_path)
        product_embeddings = np.load(npy_path)
    
    # Initialize LLM in the background
    init_llm()

@app.post("/recommend")
def recommend(request: RecommendRequest):
    if products_df is None or product_embeddings is None:
        raise HTTPException(status_code=500, detail="Data not loaded. Did you run generate_embeddings.py?")
        
    # Embed the query
    query_embedding = embedding_model.encode([request.query])[0]
    
    # Prepare filters
    filters = {
        'max_price': request.max_price,
        'category': request.category,
        'rating': request.rating
    }
    
    # Get top K products
    top_k_df = get_top_k(query_embedding, product_embeddings, products_df, filters, request.top_k)
    products_list = top_k_df.to_dict(orient="records")
    
    # Get conversational response
    if not products_list:
        response_text = "I couldn't find any products matching your specific filters."
    else:
        response_text = generate_response(request.query, products_list)
        
    return {
        "products": products_list,
        "response": response_text
    }

@app.get("/products")
def get_products(category: Optional[str] = None, min_rating: Optional[float] = None, max_price: Optional[float] = None):
    if products_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded.")
        
    filtered = products_df.copy()
    if category:
        filtered = filtered[filtered['category'].str.lower() == category.lower()]
    if min_rating:
        filtered = filtered[filtered['rating'] >= min_rating]
    if max_price:
        filtered = filtered[filtered['price'] <= max_price]
        
    return filtered.to_dict(orient="records")

@app.get("/trending")
def get_trending_products(top_k: int = 10, category: Optional[str] = None):
    """
    Returns trending products to solve the Cold Start problem for new users.
    Trending is determined by high ratings and highest number of total reviews.
    """
    if products_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded.")
        
    filtered = products_df.copy()
    if category:
        filtered = filtered[filtered['category'].str.lower() == category.lower()]
        
    # Clean up Total Ratings which might be strings with commas like "1,677"
    filtered['Total Ratings'] = pd.to_numeric(filtered['Total Ratings'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    filtered['rating'] = pd.to_numeric(filtered['rating'], errors='coerce').fillna(0)
    
    # Filter for good quality (e.g. rating >= 4.0) and sort by popularity (Total Ratings)
    trending = filtered[filtered['rating'] >= 4.0].sort_values(by='Total Ratings', ascending=False)
    
    # If we don't have enough highly rated products, just sort by Total Ratings
    if len(trending) < top_k:
        trending = filtered.sort_values(by='Total Ratings', ascending=False)
        
    return trending.head(top_k).to_dict(orient="records")

