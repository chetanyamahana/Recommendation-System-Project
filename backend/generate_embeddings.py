import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

def clean_price(price_str):
    if pd.isna(price_str):
        return 0.0
    if isinstance(price_str, str):
        price_str = price_str.replace('$', '').replace(',', '').strip()
        try:
            return float(price_str)
        except ValueError:
            return 0.0
    return float(price_str)

def main():
    print("Loading dataset...")
    df = pd.read_csv("../dataset/amazon_products_cleaned.csv")
    
    df = df.rename(columns={
        'Product Name': 'title',
        'Product Category': 'category',
        'Product Rating': 'rating',
        'Product Price': 'price',
        'Product Image': 'image'
    })
    
    df['description'] = df['title']
    df['price'] = df['price'].apply(clean_price)
    
    # Clean data: drop nulls in description, reset index
    df = df.dropna(subset=['description']).reset_index(drop=True)
    
    print("Loading model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Encoding {len(df)} products...")
    descriptions = df['description'].tolist()
    embeddings = model.encode(descriptions, show_progress_bar=True)
    
    # Save embeddings
    os.makedirs("../models", exist_ok=True)
    np.save("../models/embeddings.npy", embeddings)
    
    # Save cleaned dataframe
    df.to_csv("../dataset/products_cleaned.csv", index=False)
    
    print("Done! Total products encoded:", len(df))

if __name__ == "__main__":
    main()
