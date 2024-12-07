# search.py
import os
import torch
import open_clip
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

device = "cuda" if torch.cuda.is_available() else "cpu"

class ImageSearcher:
    def __init__(self, embedding_path='image_embeddings.pickle', image_folder='coco_images_resized'):
        self.image_folder = image_folder
        self.df = pd.read_pickle(embedding_path)
        self.filenames = self.df['file_name'].values
        self.embeddings = np.stack(self.df['embedding'].values)  # shape (N, D)
        
        # Load CLIP model and tokenizer
        self.model_name = "ViT-B-32"
        self.pretrained = "openai"
        self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
        self.model = self.model.to(device)
        self.model.eval()
        
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        
        # Precompute PCA if desired. For demonstration, we do it now; you could do lazy loading.
        self.pca_k = 50  # example number of components
        self.pca = PCA(n_components=self.pca_k)
        self.embeddings_pca = self.pca.fit_transform(self.embeddings)

    def encode_text(self, text_query: str) -> np.ndarray:
        with torch.no_grad():
            text_tokens = self.tokenizer([text_query]).to(device)
            text_emb = self.model.encode_text(text_tokens)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        return text_emb.cpu().numpy().squeeze()

    def encode_image(self, image_file) -> np.ndarray:
        # image_file could be a file-like object (from request.files)
        img = Image.open(image_file).convert("RGB")
        img_t = self.preprocess_val(img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb = self.model.encode_image(img_t)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        return img_emb.cpu().numpy().squeeze()

    def combine_embeddings(self, text_emb: np.ndarray, image_emb: np.ndarray, lam: float) -> np.ndarray:
        combined = lam * text_emb + (1 - lam) * image_emb
        # Normalize combined embedding
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        return combined

    def search_similar(self, query_emb: np.ndarray, top_k=5, use_pca=False):
        if use_pca:
            query_emb_pca = self.pca.transform(query_emb.reshape(1, -1))[0]
            sim_scores = self.embeddings_pca @ query_emb_pca
        else:
            # embeddings and query_emb are normalized, so dot product = cosine similarity
            sim_scores = self.embeddings @ query_emb
        
        top_idx = np.argsort(sim_scores)[::-1][:top_k]
        top_files = self.filenames[top_idx]
        top_scores = sim_scores[top_idx]
        return list(zip(top_files, top_scores))
