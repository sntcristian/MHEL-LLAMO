import torch
import faiss
import hydra
from hydra.experimental import compose, initialize_config_module
from typing import List, Dict, Any, Tuple
import logging
import sqlite3



logger = logging.getLogger(__name__)


class EntityDisambiguator:
    """Clean, functional entity disambiguation model using BELA."""

    def __init__(
            self,
            checkpoint_path: str = "./models/model_wiki.ckpt",
            faiss_index_path: str = "./models/faiss.index",
            wikidata_index_path: str = "./models/index.txt",
            db_path: str = "./models/entities.sqlite",
            embedding_dim : int = 300,
            config_name: str = "joint_el_mel",
            device: str = "cuda:0"
    ):
        """
        Initialize the entity disambiguation model.

        Args:
            checkpoint_path: Path to the trained model checkpoint
            faiss_index_path: Path to the precomputed FAISS index
            wikidata_index_path: Path to the Wikidata QID index file
            db_path : Path to the SQLITE database containing entity information
            config_name: Hydra config name for the model
            device: Device to run the model on
        """

        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.faiss_index_path = faiss_index_path
        self.wikidata_index_path = wikidata_index_path
        self.db_path = db_path
        self.embedding_dim = embedding_dim

        # Load model and components
        self._load_model(config_name)
        self._load_faiss_index()
        self._load_entity_db()

        logger.info(f"EntityDisambiguator initialized")

    def _load_model(self, config_name: str) -> None:
        """Load the BELA model from checkpoint."""
        logger.info("Loading BELA model...")

        with initialize_config_module("bela/conf"):
            cfg = compose(config_name=config_name)
            cfg.task.load_from_checkpoint = self.checkpoint_path
            cfg.task.embedding_dim = self.embedding_dim
            cfg.datamodule.ent_catalogue_idx_path = self.wikidata_index_path
            cfg.datamodule.train_path = None
            cfg.datamodule.val_path = None
            cfg.datamodule.test_path = None


        # Initialize components
        self.transform = hydra.utils.instantiate(cfg.task.transform)
        datamodule = hydra.utils.instantiate(cfg.datamodule, transform=self.transform)
        self.task = hydra.utils.instantiate(cfg.task, datamodule=datamodule, _recursive_=False)

        # Setup and move to device
        self.task.setup("train")
        self.task.eval()
        self.task.to(self.device)


    def _load_faiss_index(self) -> None:
        """Load the precomputed FAISS index."""
        logger.info(f"Loading FAISS index from {self.faiss_index_path}")
        self.faiss_index = faiss.read_index(self.faiss_index_path)

        # Move to GPU if available
        if self.device.type == 'cuda' and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, self.device.index, self.faiss_index)


    def _load_entity_db(self):
        logger.info(f"Loading entity database from {self.db_path}")
        self.conn = sqlite3.connect(self.db_path)
        logger.info("Database loaded")



    def _encode_text(self, texts: List[str], mention_offsets: List[List[int]],
                     mention_lengths: List[List[int]]) -> torch.Tensor:
        """
        Encode text and extract mention representations.

        Args:
            texts: List of input texts
            mention_offsets: List of mention start positions for each text
            mention_lengths: List of mention lengths for each text

        Returns:
            Tensor of mention representations
        """
        # Prepare batch
        batch = {
            "texts": texts,
            "mention_offsets": mention_offsets,
            "mention_lengths": mention_lengths,
        }

        # Transform inputs
        model_inputs = self.transform(batch)
        token_ids = model_inputs["input_ids"].to(self.device)
        mention_offsets_tensor = model_inputs["mention_offsets"]
        mention_lengths_tensor = model_inputs["mention_lengths"]

        with torch.no_grad():
            # Encode text
            _, text_encodings = self.task.encoder(token_ids)
            text_encodings = self.task.project_encoder_op(text_encodings)

            # Extract mention representations
            mention_representations = self.task.span_encoder(
                text_encodings, mention_offsets_tensor, mention_lengths_tensor
            )

            # Filter out empty mentions
            valid_mentions = mention_representations[mention_lengths_tensor != 0]

        return valid_mentions

    def _search_candidates(self, mention_representations: torch.Tensor, k: int = 1) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Search for entity candidates using FAISS index.

        Args:
            mention_representations: Tensor of mention representations
            k: Number of candidates to retrieve

        Returns:
            Tuple of (scores, indices)
        """
        scores, indices = self.faiss_index.search(mention_representations.detach().cpu().numpy(), k=k)
        return torch.from_numpy(scores).to(self.device), torch.from_numpy(indices).to(self.device)



    def get_candidates_batch(self,
                texts: List[str],
                mention_offsets: List[List[int]],
                mention_lengths: List[List[int]],
                k: int = 10
        ) -> List[List[Dict[str, Any]]]:
            """
            Get top-k candidates in a batch of texts.

            Args:
                texts: List of input texts
                mention_offsets: List of mention start positions for each text
                mention_lengths: List of mention lengths for each text

            Returns:
                List of predictions for each text, where each prediction contains:
                - start_pos: Start position of the mention
                - end_pos: End position of the mention
                - entity: Predicted entity ID
                - score: Confidence score
            """
            # Encode mentions
            mention_representations = self._encode_text(texts, mention_offsets, mention_lengths)

            # Search for candidates
            scores, indices = self._search_candidates(mention_representations, k=k)
            scores, indices = scores.tolist(), indices.tolist()

            # Format predictions
            predictions = []
            example_idx = 0

            for text, offsets, lengths in zip(texts, mention_offsets, mention_lengths):
                text_predictions = []

                for offset, length in zip(offsets, lengths):
                    candidates = []
                    if length > 0:  # Valid mention
                        ex_indices = indices[example_idx]
                        ex_scores = scores[example_idx]
                        for index, score in zip(ex_indices, ex_scores):
                            cursor = self.conn.cursor()
                            cursor.execute("""
                                                            SELECT id, wikidata_qid, enwiki, dewiki, itwiki, frwiki, svwiki, fiwiki, nlwiki, type_, min_date
                                                            FROM entities
                                                            WHERE id = ?
                                                        """, (index,))
                            candidate_info = cursor.fetchall()[0]
                            candidates.append({
                                "wb_id": candidate_info[1],
                                "enwiki": candidate_info[2],
                                "dewiki": candidate_info[3],
                                "itwiki": candidate_info[4],
                                "frwiki": candidate_info[5],
                                "svwiki": candidate_info[6],
                                "fiwiki": candidate_info[7],
                                "nlwiki": candidate_info[8],
                                "type": candidate_info[9],
                                "min_date": candidate_info[10],
                                "score": score
                            })
                        text_predictions.append(
                            {"start_pos": offset,
                             "end_pos": offset + length,
                             "surface":text[offset:offset+length],
                             "candidates":candidates}
                        )
                        example_idx += 1
                predictions.append(text_predictions)

            return predictions





def load_disambiguator(
        checkpoint_path: str = "./models/model_wiki.ckpt",
        faiss_index_path: str = "./models/faiss.index",
        wikidata_index_path: str = "./models/index.txt",
        db_path: str = "./models/entities.sqlite",
        device: str = "cuda:0",
        embedding_dim = 300
) -> EntityDisambiguator:
    """
    Factory function to create an EntityDisambiguator instance.

    Args:
        checkpoint_path: Path to model checkpoint
        faiss_index_path: Path to FAISS index
        wikidata_index_path: Path to Wikidata QID index
        device: Device to run on

    Returns:
        Configured EntityDisambiguator instance
    """
    return EntityDisambiguator(
        checkpoint_path=checkpoint_path,
        faiss_index_path=faiss_index_path,
        wikidata_index_path=wikidata_index_path,
        db_path=db_path,
        device=device,
        embedding_dim = embedding_dim
    )

