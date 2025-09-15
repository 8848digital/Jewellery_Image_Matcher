# api.py
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import chromadb
import io
import logging
import re
import os
import tarfile
import requests
from pathlib import Path
import argparse
import torch.serialization
from dotenv import load_dotenv
torch.serialization.add_safe_globals([argparse.Namespace])


# ================================
# Load environment variables
# ================================
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID")
MODEL_FILENAME = os.getenv("MODEL_FILENAME")
CHROMADB_URL = os.getenv("CHROMADB_URL")
CHROMADB_TAR = os.getenv("CHROMADB_TAR", "jewelry_chroma_db.tar.gz")
CHROMADB_DIR = os.getenv("CHROMADB_DIR", "jewelry_chroma_db")

# ================================
# Logging
# ================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Jewelry Similarity API")

# ================================
# Serve images (local folder if needed)
# ================================
app.mount("/Pictures", StaticFiles(directory="Pictures"), name="Pictures")

# ================================
# Device setup
# ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# Model Loading
# ================================
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ Model file not found at {os.path.abspath(MODEL_PATH)}")

        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model_state_dict = checkpoint["model_state_dict"]

        # Load DinoV2 backbone without pretrained weights
        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitl14", pretrained=False
        )

        # Adapt checkpoint keys for DinoV2
        dinov2_state_dict = {}
        for key, value in model_state_dict.items():
            if key.startswith("backbone."):
                dinov2_state_dict[key[9:]] = value

        model.load_state_dict(dinov2_state_dict, strict=False)
        model = model.to(DEVICE).eval()

        logger.info(f"✅ Fine-tuned jewelry model loaded from {MODEL_PATH}")
        return model

    except Exception as e:
        logger.error(f"❌ Failed to load local model: {e}")
        raise e

MODEL = load_model()


# ================================
# Download & Extract ChromaDB
# ================================
def ensure_chroma_db():
    tar_path = CHROMADB_TAR
    extract_dir = CHROMADB_DIR

    if os.path.exists(extract_dir):
        logger.info("📂 Using existing ChromaDB folder")
        return extract_dir

    if not os.path.exists(tar_path):
        logger.info("⬇️ Downloading ChromaDB from Hugging Face...")
        response = requests.get(CHROMADB_URL, stream=True)
        with open(tar_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info("✅ Download complete")

    logger.info("📦 Extracting ChromaDB...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall()
    logger.info("✅ Extraction complete")

    return extract_dir

# ================================
# ChromaDB Loading
# ================================
def load_chroma_db():
    try:
        db_path = ensure_chroma_db()
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection("jewelry_embeddings")
        logger.info(f"📚 Database loaded with {collection.count()} embeddings!")
        return collection
    except Exception as e:
        logger.error(f"Failed to load database: {str(e)}")
        return None

COLLECTION = load_chroma_db()

# ================================
# Image Transforms
# ================================
TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ================================
# Embedding Extraction
# ================================
def extract_embedding(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = TRANSFORMS(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = MODEL.forward_features(image_tensor)
        emb = outputs["x_norm_clstoken"].cpu().numpy().squeeze()
        emb = emb / np.linalg.norm(emb)
    return emb

# ================================
# Clean ID Helper
# ================================
def clean_id(full_id: str) -> str:
    match = re.search(r"[A-Z]{2,}\d{3,}", full_id)
    return match.group(0) if match else full_id

# ================================
# Search Function
# ================================
def search_similar(query_emb, top_k=5):
    if COLLECTION is None:
        return []
    results = COLLECTION.query(
        query_embeddings=[query_emb.tolist()],
        n_results=top_k,
        include=["distances"],
    )

    items = []
    for i in range(len(results["ids"][0])):
        items.append({
            "id": clean_id(results["ids"][0][i]),
            "similarity": 1 - results["distances"][0][i],
        })
    return items

# ================================
# API Endpoint
# ================================
@app.post("/query")
async def query_jewelry(file: UploadFile = File(...), top_k: int = Query(5, ge=1, le=20)):
    try:
        image_bytes = await file.read()
        query_emb = extract_embedding(image_bytes)
        results = search_similar(query_emb, top_k=top_k)
        return JSONResponse({"results": results})
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
