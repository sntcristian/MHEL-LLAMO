import torch
import faiss
import numpy as np
from typing import Optional
import os


class EmbeddingIndexer:
    def __init__(self, embeddings_path: str, device: int = 0, use_gpu: bool = True):
        """
        Initialize the embedding indexer.

        Args:
            embeddings_path: Path to the .pt file containing embeddings
            device: GPU device ID (default: 0)
            use_gpu: Whether to use GPU for indexing (default: True)
        """
        self.embeddings_path = embeddings_path
        self.device = device
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.embeddings = None
        self.faiss_index = None

    def load_embeddings(self):
        """Load embeddings from .pt file and ensure they're in the right format."""
        print(f"Loading embeddings from {self.embeddings_path}...")

        # Load the tensor
        embeddings_tensor = torch.load(self.embeddings_path, map_location='cpu')

        # Convert to numpy array with float32 dtype (required by FAISS)
        if isinstance(embeddings_tensor, torch.Tensor):
            self.embeddings = embeddings_tensor.detach().cpu().numpy().astype(np.float32)
        else:
            # Handle case where the file contains a dict or other structure
            if isinstance(embeddings_tensor, dict):
                # Try common keys
                for key in ['embeddings', 'data', 'tensor', 'features']:
                    if key in embeddings_tensor:
                        self.embeddings = embeddings_tensor[key].detach().cpu().numpy().astype(np.float32)
                        break
                if self.embeddings is None:
                    raise ValueError(f"Could not find embeddings in dict with keys: {list(embeddings_tensor.keys())}")
            else:
                raise ValueError(f"Unsupported data type: {type(embeddings_tensor)}")

        # Ensure embeddings are 2D
        if len(self.embeddings.shape) == 1:
            self.embeddings = self.embeddings.reshape(1, -1)
        elif len(self.embeddings.shape) != 2:
            raise ValueError(f"Embeddings must be 2D, got shape: {self.embeddings.shape}")

        # Normalize embeddings for cosine similarity (optional but recommended for XLM-R)
        # Comment out the next line if you don't want normalization
        # faiss.normalize_L2(self.embeddings)

        print(f"Loaded embeddings with shape: {self.embeddings.shape}")
        print(f"Embedding dimension: {self.embeddings.shape[1]}")

    def create_gpu_index(self):
        """Create a GPU-based FAISS index."""
        if not self.use_gpu:
            raise ValueError("GPU indexing requested but GPU not available or disabled")

        print(f"Creating GPU index on device {self.device}...")

        # Create GPU resources
        res = faiss.StandardGpuResources()

        # Configure GPU index
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = self.device
        flat_config.useFloat16 = True  # Use half precision to save memory

        # Create the index (using Inner Product for normalized vectors = cosine similarity)
        self.faiss_index = faiss.GpuIndexFlatIP(res, self.embeddings.shape[1], flat_config)

        # Add embeddings to index
        print("Adding embeddings to GPU index...")
        self.faiss_index.add(self.embeddings)

        print(f"GPU index created successfully with {self.faiss_index.ntotal} vectors")

    def create_cpu_index(self):
        """Create a CPU-based FAISS index."""
        print("Creating CPU index...")

        # Create CPU index (Inner Product for cosine similarity with normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])

        # Add embeddings to index
        print("Adding embeddings to CPU index...")
        self.faiss_index.add(self.embeddings)

        print(f"CPU index created successfully with {self.faiss_index.ntotal} vectors")

    def create_index(self):
        """Create the appropriate index based on configuration."""
        if self.embeddings is None:
            self.load_embeddings()

        if self.use_gpu:
            try:
                self.create_gpu_index()
            except Exception as e:
                print(f"GPU indexing failed: {e}")
                print("Falling back to CPU indexing...")
                self.use_gpu = False
                self.create_cpu_index()
        else:
            self.create_cpu_index()

    def save_index(self, output_path: str):
        """Save the FAISS index to disk."""
        if self.faiss_index is None:
            raise ValueError("No index created yet. Call create_index() first.")

        print(f"Saving index to {output_path}...")

        if self.use_gpu:
            # Convert GPU index to CPU index for saving
            cpu_index = faiss.index_gpu_to_cpu(self.faiss_index)
            faiss.write_index(cpu_index, output_path)
        else:
            faiss.write_index(self.faiss_index, output_path)

        print("Index saved successfully!")

    def search(self, query_embeddings: np.ndarray, k: int = 10):
        """
        Search the index for similar embeddings.

        Args:
            query_embeddings: Query embeddings as numpy array
            k: Number of nearest neighbors to return

        Returns:
            distances, indices: Arrays of distances and indices
        """
        if self.faiss_index is None:
            raise ValueError("No index created yet. Call create_index() first.")

        # Ensure query embeddings are float32 and 2D
        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.detach().cpu().numpy()
        query_embeddings = query_embeddings.astype(np.float32)

        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        # Normalize query embeddings if original embeddings were normalized
        # faiss.normalize_L2(query_embeddings)

        # Perform search
        distances, indices = self.faiss_index.search(query_embeddings, k)

        return distances, indices


# Example usage
def main():
    # Configuration
    embeddings_file = "./models/embeddings.pt"  # Replace with your file path
    index_output_file = "./models/faiss.index"

    # Create indexer
    indexer = EmbeddingIndexer(
        embeddings_path=embeddings_file,
        device=0,  # GPU device ID
        use_gpu=True  # Set to False to use CPU only
    )

    try:
        # Create the index
        indexer.create_index()

        # Save the index
        indexer.save_index(index_output_file)

        # Example search (optional)
        if indexer.embeddings.shape[0] > 0:
            # Search using the first embedding as query
            query = indexer.embeddings[0:1]  # First embedding as query
            distances, indices = indexer.search(query, k=5)
            print(f"Search results - distances: {distances[0]}, indices: {indices[0]}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()