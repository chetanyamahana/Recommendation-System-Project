Here is a comprehensive summary of the project. You can copy and paste this entire document directly into Claude to have it generate detailed architecture and flow diagrams.

---

**PROMPT FOR CLAUDE:**
Please read the following summary of my Conversational E-Commerce Recommendation System project. Based on this information, please generate detailed system architecture flow diagrams, data pipeline diagrams, and user interaction flowcharts (e.g., using Mermaid.js or PlantUML).

### 1. Objective
To build an end-to-end, LLM-powered conversational recommender system for an e-commerce platform. The system aims to bridge the gap between structured product databases and natural human language by allowing users to search for products using descriptive, nuanced queries rather than rigid keywords.

### 2. Plan for Fix / Evolution
*Context: The project evolved from a basic dummy-data system to a robust, real-world application.*
*   **Data Migration:** Transitioning the system to handle a massive, real-world Amazon products dataset (`amazon_products_cleaned.csv`).
*   **Data Pipeline Refactoring:** Updating the embedding generation script to clean real-world anomalies (like currency symbols in prices) and mapping Amazon product names to the semantic "description" field.
*   **Evaluation Correction:** Modifying the automated evaluation script (`eval.ipynb`) to accurately reflect the semantic search's success by mapping test queries to the exact category taxonomies present in the new Amazon dataset.

### 3. Requirements
*   **Backend:** Python, FastAPI (for RESTful API), `sentence-transformers` (for vector embeddings), `scikit-learn` (for cosine similarity), `pandas` and `numpy`.
*   **Generative AI:** Local `TinyLlama` model running via HuggingFace `transformers` to generate conversational explanations.
*   **Frontend:** A responsive, interactive web interface (GUI) built with HTML5, Vanilla JavaScript, and Tailwind CSS.
*   **Evaluation:** Jupyter Notebook for calculating Information Retrieval metrics (Precision@5, Recall@5, F1-Score) against the live API.

### 4. Methodology Used
*   **Semantic Vectorization (Offline):** Product descriptions are passed through the `all-MiniLM-L6-v2` model to generate dense, mathematical vector embeddings. These are saved to disk as a `.npy` file.
*   **Hybrid Retrieval Engine (Online):** 
    1.  A user's natural language query is instantly embedded into a vector.
    2.  The backend computes the Cosine Similarity between the query vector and all product vectors to find the Top-K semantic matches.
    3.  Hard filters (Maximum Price, Category, Minimum Rating) are applied to the Top-K matches.
*   **Conversational Generation:** The filtered product results (including title, price, and category) are fed into the TinyLlama LLM prompt, which returns a friendly, personalized summary for the user.

### 5. Codebase Speciality
*   **Modular Architecture:** The codebase is strictly separated by concern: `generate_embeddings.py` (offline data processing), `app.py` (FastAPI routing), `recommendation.py` (similarity math and filtering logic), and `llm.py` (text generation).
*   **Local, Privacy-First AI:** The system uses 100% locally hosted, open-source models (Sentence Transformers and TinyLlama). It does not rely on paid, external APIs like OpenAI, ensuring complete data privacy and cost-free execution.
*   **Hybrid Search:** It uniquely combines soft semantic matching (understanding "comfortable shoes") with hard metadata constraints (Price < $50).

### 6. Final Outcomes
*   A highly responsive, web-based conversational recommendation bot ("ShopBot") that displays dynamic product cards (images, prices, ratings).
*   Successful integration of a large-scale Amazon e-commerce dataset.
*   Strong Information Retrieval metrics confirming the model's semantic understanding (e.g., Average Precision@5 of 0.50 and Recall@5 of 0.83).
*   A fully functional system that successfully demonstrates how Generative AI can transform traditional e-commerce search into a user-adaptive, personalized shopping assistant.
