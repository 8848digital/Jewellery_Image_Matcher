# extract.py - Updated to use unified model
from unified_model import UnifiedJewelryModel
import chromadb
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_and_store_embeddings(image_directory, model_path, collection_name="jewelry_embeddings"):
    """Extract embeddings and store in ChromaDB using unified model"""
    
    # Initialize unified model
    model = UnifiedJewelryModel(model_path)
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create or get collection
    try:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine distance
        )
        logger.info(f"Created new collection: {collection_name}")
    except Exception:
        collection = client.get_collection(name=collection_name)
        logger.info(f"Using existing collection: {collection_name}")
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in Path(image_directory).rglob('*') 
        if f.suffix.lower() in image_extensions
    ]
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process images in batches
    batch_size = 32
    processed = 0
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        
        # Extract embeddings for batch
        embeddings = []
        ids = []
        metadatas = []
        documents = []
        
        for img_path in batch_files:
            try:
                # Extract embedding using unified model
                embedding = model.extract_embedding(str(img_path))
                
                # Prepare data for ChromaDB
                embeddings.append(embedding.tolist())
                ids.append(str(img_path.stem))
                metadatas.append({
                    'image_path': str(img_path),
                    'filename': img_path.name,
                    'directory': str(img_path.parent)
                })
                documents.append(f"Jewelry image: {img_path.name}")
                
                processed += 1
                if processed % 100 == 0:
                    logger.info(f"Processed {processed}/{len(image_files)} images")
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        # Add batch to ChromaDB
        if embeddings:
            try:
                collection.add(
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents,
                    ids=ids
                )
                logger.info(f"Added batch of {len(embeddings)} embeddings to ChromaDB")
            except Exception as e:
                logger.error(f"Error adding batch to ChromaDB: {e}")
    
    logger.info(f"Extraction complete! Processed {processed} images")
    logger.info(f"Total items in collection: {collection.count()}")
    
    return collection

def test_consistency(model_path, test_image_path, collection_name="jewelry_embeddings"):
    """Test consistency between extraction and search"""
    
    # Load unified model
    model = UnifiedJewelryModel(model_path)
    
    # Load ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name=collection_name)
    
    # Extract embedding for test image
    test_embedding = model.extract_embedding(test_image_path)
    
    # Search in ChromaDB
    results = collection.query(
        query_embeddings=[test_embedding.tolist()],
        n_results=5,
        include=['documents', 'metadatas', 'distances']
    )
    
    print(f"Test image: {test_image_path}")
    print(f"Query embedding shape: {test_embedding.shape}")
    print(f"Query embedding norm: {test_embedding.norm():.4f}")
    print("\nTop 5 similar items:")
    
    for i in range(len(results['ids'][0])):
        distance = results['distances'][0][i]
        similarity = 1.0 - distance  # Convert distance to similarity
        print(f"{i+1}. ID: {results['ids'][0][i]}")
        print(f"   Distance: {distance:.4f}, Similarity: {similarity:.4f}")
        if results['metadatas'][0][i]:
            print(f"   Path: {results['metadatas'][0][i].get('image_path', 'N/A')}")
        print()

if __name__ == "__main__":
    # Configuration
    IMAGE_DIRECTORY = r"C:\Users\Admin\Desktop\new-vitl14\Pictures"  # Update this
    MODEL_PATH = r"C:\Users\Admin\Desktop\new-vitl14\best_model.pth"    # Update this
    TEST_IMAGE = r"C:\Users\Admin\Desktop\new-vitl14\Pictures\3D\BH\DM 3D BH13014v1.jpg"            # Update this
    
    # Extract embeddings
    print("Starting embedding extraction...")
    extract_and_store_embeddings(IMAGE_DIRECTORY, MODEL_PATH)
    
    # Test consistency
    print("\nTesting consistency...")
    test_consistency(MODEL_PATH, TEST_IMAGE)