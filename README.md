# ShopBot - AI Product Recommender

This is a local, full-stack product recommendation system built with FastAPI, vanilla HTML/JS, Sentence-Transformers, and a local Hugging Face LLM (TinyLlama).

## Setup Instructions

1. **Install Dependencies**
   Navigate to the `project/` directory and install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Embeddings**
   Run the embedding script to process the dataset and generate the necessary `.npy` file.
   ```bash
   cd backend
   python generate_embeddings.py
   ```

3. **Run the Backend Server**
   Start the FastAPI server. The first time you run this, it will download the TinyLlama model (approx 2GB).
   ```bash
   cd backend
   uvicorn app:app --reload
   ```

4. **Open the Frontend**
   Simply double-click `project/frontend/index.html` to open it in your browser. No web server is needed for the frontend.

5. **Run Evaluation**
   Open the Jupyter Notebook in `project/eval/eval.ipynb` to test Precision, Recall, and F1 metrics.
   ```bash
   cd eval
   jupyter notebook eval.ipynb
   ```
