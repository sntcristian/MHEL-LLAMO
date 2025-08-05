import argparse
import os
import csv
import json
from tqdm import tqdm
from src.retriever import EntityDisambiguator


def load_disambiguator(
        checkpoint_path: str = "./models/model_wiki.ckpt",
        faiss_index_path: str = "./models/faiss.index",
        wikidata_index_path: str = "./models/index.txt",
        db_path: str = "./models/knowledge_base_final.sqlite",
        device: str = "cuda:0",
        embedding_dim: int = 300
):
    return EntityDisambiguator(
        checkpoint_path=checkpoint_path,
        faiss_index_path=faiss_index_path,
        wikidata_index_path=wikidata_index_path,
        db_path=db_path,
        device=device,
        embedding_dim=embedding_dim
    )


def load_dataset(dataset_path):
    paragraphs_path = os.path.join(dataset_path, "paragraphs_test.csv")
    annotations_path = os.path.join(dataset_path, "annotations_test.csv")

    with open(paragraphs_path, "r", encoding="utf-8") as doc_f:
        paragraphs = list(csv.DictReader(doc_f))

    with open(annotations_path, "r", encoding="utf-8") as anno_f:
        annotations = list(csv.DictReader(anno_f))

    texts, offsets, lengths, doc_ids, gt_ids = [], [], [], [], []

    for doc in paragraphs:
        text = doc["text"]
        doc_id = doc["doc_id"]
        doc_anno = [row for row in annotations if row["doc_id"] == doc_id]

        ex_offsets = [int(anno["start_pos"]) for anno in doc_anno]
        ex_lengths = [int(anno["end_pos"]) - int(anno["start_pos"]) for anno in doc_anno]
        ex_gt_ids = [anno["identifier"] for anno in doc_anno]

        if len(ex_offsets)>0 and len(text) <= 1250:
            texts.append(text)
            doc_ids.append(doc_id)
            offsets.append(ex_offsets)
            lengths.append(ex_lengths)
            gt_ids.append(ex_gt_ids)

    return texts, offsets, lengths, doc_ids, gt_ids


def main():
    parser = argparse.ArgumentParser(description="Entity disambiguation with candidate generation")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--lang", type=str, required=True, help="Language code (e.g., 'it', 'en')")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top candidates to retrieve")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of documents in a batch")

    args = parser.parse_args()

    disambiguator = load_disambiguator()
    texts, offsets, lengths, doc_ids, gt_ids = load_dataset(args.dataset_path)

    batch_size = args.batch_size
    all_results = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        batch_offsets = offsets[i:i + batch_size]
        batch_lengths = lengths[i:i + batch_size]
        batch_doc_ids = doc_ids[i:i + batch_size]
        batch_gt_ids = gt_ids[i:i + batch_size]

        batch_predictions = disambiguator.get_candidates_batch(
            batch_texts, batch_offsets, batch_lengths, k=args.top_k, lang=args.lang
        )

        for doc_id, predictions, ex_gt_ids in zip(batch_doc_ids, batch_predictions, batch_gt_ids):
            for pred, gt_id in zip(predictions, ex_gt_ids):
                all_results.append({
                    "doc_id": doc_id,
                    "start_pos": pred["start_pos"],
                    "end_pos": pred["end_pos"],
                    "surface": pred["surface"],
                    "identifier": gt_id,
                    "candidates": pred["candidates"]
                })

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, f"candidates_top{args.top_k}.json"), "w", encoding="utf-8") as out_f:
        json.dump(all_results, out_f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()