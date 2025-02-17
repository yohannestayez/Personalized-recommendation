# Personalized Recommendation System

Implementation of a hybrid personalized recommender that combines collaborative insights from user-item interaction graphs with content-based signals derived from textual data.

## Technical Details

- **Content-based Feature Extraction using BERT:**  
  - Item descriptions and relevant textual metadata are processed through a pre-trained BERT model.  
  - BERT extracts semantic embeddings that characterize the content, ensuring that even items with sparse interaction data can be effectively represented.

- **Collaborative Filtering via a Light GNN:**  
  - A graph is constructed where nodes represent users and items.
  - A lightweight Graph Neural Network built with PyTorch and torch-geometric propagates neighborhood information to learn latent features for both users and items.  
  - The GNN is responsible for capturing collaborative patterns and relational structures within the interaction data.

- **Fusion of Collaborative and Content Signals: A Hybrid Approach**  
  - The latent features from the GNN and the semantic embeddings from BERT are combined to produce enriched representations for recommendation.  
  - This fusion approach leverages the strengths of both collaborative filtering and content-based methods, leading to more nuanced and personalized recommendations.

### features
- **Text Feature Extraction:**  
Modules integrating BERT to obtain content-based embeddings from item descriptions.
- **Graph Construction & Training:**  
Scripts that build the interaction graph and train the GNN model to learn user and item representations from users interaction and Item embeddings.
- **Recommendation Pipeline:**  
Combines the outputs from the collaborative and content-based components and produces final recommendations.



### Getting Started

1. **Installation:**  
   Install all dependencies using:
   ```sh
   pip install -r requirements.txt
   ```