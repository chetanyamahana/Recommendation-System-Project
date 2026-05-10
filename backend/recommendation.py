import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def get_top_k(query_embedding, product_embeddings, products_df, filters, k=5):
    # Compute cosine similarity between query and all product embeddings
    similarities = cosine_similarity(query_embedding.reshape(1, -1), product_embeddings)[0]
    
    # Add similarities to the dataframe temporarily
    products_df = products_df.copy()
    products_df['similarity'] = similarities
    
    # Retrieve top 50 by similarity score
    top_50 = products_df.nlargest(50, 'similarity')
    
    # Apply filters
    if filters.get('max_price'):
        top_50 = top_50[top_50['price'] <= filters['max_price']]
    
    if filters.get('category'):
        # Case insensitive match
        top_50 = top_50[top_50['category'].str.lower() == filters['category'].lower()]
        
    if filters.get('rating'):
        top_50 = top_50[top_50['rating'] >= filters['rating']]
        
    # Return top_k
    top_k_df = top_50.head(k)
    return top_k_df
