import torch
import faiss
import hydra
from hydra.experimental import compose, initialize_config_module
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class EntityDisambiguator:
    """Clean, functional entity disambiguation model using BELA."""

    def __init__(
            self,
            checkpoint_path: str = "./models/model_wiki.ckpt",
            faiss_index_path: str = "./models/faiss.index",
            wikidata_index_path: str = "./models/index.txt",
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
            config_name: Hydra config name for the model
            device: Device to run the model on
        """
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.faiss_index_path = faiss_index_path
        self.wikidata_index_path = wikidata_index_path
        self.embedding_dim = embedding_dim

        # Load model and components
        self._load_model(config_name)
        self._load_faiss_index()
        self._load_entity_catalog()

        logger.info(f"EntityDisambiguator initialized with {len(self.entity_catalog)} entities")

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

    def _load_entity_catalog(self) -> None:
        """Load entity catalog from Wikidata index file."""
        logger.info(f"Loading entity catalog from {self.wikidata_index_path}")

        self.entity_catalog = []
        with open(self.wikidata_index_path, 'r', encoding='utf-8') as f:
            for line in f:
                qid = line.strip()
                if qid:  # Skip empty lines
                    self.entity_catalog.append(qid)

        logger.info(f"Loaded {len(self.entity_catalog)} entities from index")

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

    def disambiguate_batch(
            self,
            texts: List[str],
            mention_offsets: List[List[int]],
            mention_lengths: List[List[int]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Disambiguate entities in a batch of texts.

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
        scores, indices = self._search_candidates(mention_representations)

        # Format predictions
        predictions = []
        mention_idx = 0

        for text_idx, (offsets, lengths) in enumerate(zip(mention_offsets, mention_lengths)):
            text_predictions = []

            for offset, length in zip(offsets, lengths):
                if length > 0:  # Valid mention
                    entity_id = self.entity_catalog[indices[mention_idx].item()]
                    score = scores[mention_idx].item()

                    text_predictions.append({
                        "start_pos": offset,
                        "end_pos": offset + length,
                        "entity": entity_id,
                        "score": score
                    })

                    mention_idx += 1

            predictions.append(text_predictions)

        return predictions

    def disambiguate_single(
            self,
            text: str,
            mention_offsets: List[int],
            mention_lengths: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Disambiguate entities in a single text.

        Args:
            text: Input text
            mention_offsets: List of mention start positions
            mention_lengths: List of mention lengths

        Returns:
            List of predictions with entity IDs and scores
        """
        batch_predictions = self.disambiguate_batch([text], [mention_offsets], [mention_lengths])
        return batch_predictions[0]




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

            for text_idx, (offsets, lengths) in enumerate(zip(mention_offsets, mention_lengths)):
                text_predictions = []

                for offset, length in zip(offsets, lengths):
                    if length > 0:  # Valid mention
                        ex_indices = indices[example_idx]
                        ex_scores = scores[example_idx]
                        entity_ids = [self.entity_catalog[entity_idx] for entity_idx in ex_indices]

                        text_predictions.append({
                            "start_pos": offset,
                            "end_pos": offset + length,
                            "entities": entity_ids,
                            "scores": ex_scores
                        })
                        example_idx += 1
                predictions.append(text_predictions)

            return predictions





def create_disambiguator(
        checkpoint_path: str = "./models/model_wiki.ckpt",
        faiss_index_path: str = "./models/faiss.index",
        wikidata_index_path: str = "./models/index.txt",
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
        device=device,
        embedding_dim = 300
    )


# Example usage
def main():
    """Example usage of the EntityDisambiguator."""

    # Initialize disambiguator
    disambiguator = create_disambiguator()

    # Example input
    text = "Leonardo da Vinci lived in Firenze"
    mention_offsets = [0, 27]  # "Leonardo da Vinci", "Firenze"
    mention_lengths = [17, 7]

    # Disambiguate entities
    predictions = disambiguator.disambiguate_single(text, mention_offsets, mention_lengths)

    # Print results
    for pred in predictions:
        mention_text = text[pred["start_pos"]:pred["end_pos"]]
        print(f"Mention: '{mention_text}' -> Entity: {pred['entity']} (score: {pred['score']:.4f})")

    # Batch example
    texts = [
        "Leonardo da Vinci lived in Firenze",
        "Einstein was born in Germany",
        "Ces hommes qui existent ainsi * (les Chartreux de Rome) sont pourtant les mêmes à qui la guerre et toute son activité suffiraient à peine s'ils s'y étaient accoutumés. C'est un sujet inépuisable de réflexion que ﻿ 3467 les différentes combinaisons de la destinée humaine sur la terre. Il se passe dans l'intérieur de l'ame mille accidents, il se forme mille habitudes qui font de chaque individu un monde et son histoire. Connaître un autre parfaitement serait l'étude d'une vie entière; qu'est-ce donc qu'on entend par connaître les hommes? les gouverner, cela se peut, mais les comprendre, Dieu seul le fait. * Corinne, livre 10. Chap. 1. t. 2. p. 114.",
        "Fermezza di carattere e facoltà di generalizzare formano quelli che si chiamano uomini superiori: essi sanno pensare e sanno operare * : ﻿ 3447 dice M. Say ne' Cenni sugli uomini e la Società."
    ]
    batch_offsets = [[0, 27], [0, 21], [50, 613], [160]]
    batch_lengths = [[17, 7], [8, 7], [4, 7], [31]]

    batch_predictions = disambiguator.disambiguate_batch(texts, batch_offsets, batch_lengths)

    for i, (text, predictions) in enumerate(zip(texts, batch_predictions)):
        print(f"\nText {i + 1}: {text}")
        for pred in predictions:
            mention_text = text[pred["start_pos"]:pred["end_pos"]]
            print(f"  '{mention_text}' -> {pred['entity']} (score: {pred['score']:.4f})")

    batch_topk = disambiguator.get_candidates_batch(texts, batch_offsets, batch_lengths, 10)
    for i, (text, predictions) in enumerate(zip(texts, batch_topk)):
        print(f"\nText {i + 1}: {text}")
        for pred in predictions:
            mention_text = text[pred["start_pos"]:pred["end_pos"]]
            print(mention_text, pred["entities"], pred["scores"])

if __name__ == "__main__":
    main()