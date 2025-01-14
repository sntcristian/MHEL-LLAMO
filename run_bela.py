import sys
import csv
import os
from tqdm import tqdm

sys.path.insert(0, '../')

from bela.evaluation.model_eval import ModelEval
embeddings_path = "./models/embeddings.pt"
ent_catalogue_idx_path = "./models/index.txt"
checkpoint_path = "./models/model_wiki.ckpt"


md_threshold = 0.2
el_threshold = 0.4

print(f"Loading model from checkpoint {checkpoint_path}")
model_eval = ModelEval(checkpoint_path, config_name="joint_el_mel_new",embeddings_path=embeddings_path,ent_catalogue_idx_path=ent_catalogue_idx_path)

model_eval.task.md_threshold = md_threshold
model_eval.task.el_threshold = el_threshold

def process_data(paragraphs, annotations):
    pbar = tqdm(total=len(paragraphs))
    output = []
    for paragraph in paragraphs:
        doc_id = paragraph["doc_id"]
        text = paragraph["text"]
        annotations = [row for row in annotations if row["doc_id"]==doc_id]
        offsets = [int(anno["start_pos"]) for anno in annotations]
        lengths = [len(anno["surface"]) for anno in annotations]
        types = [anno["type"] for anno in annotations]
        entities = [0 for _ in annotations]

        results = model_eval.process_disambiguation_batch(
            texts=[text],
            mention_offsets=[offsets],
            mention_lengths=[lengths],
            entities=[entities]
        )

        entities = results[0]["entities"]
        scores = results[0]["scores"]
        for offset, length, identifier, score, _type in zip(offsets, lengths, entities, scores, types):
            output.append({
                "doc_id":doc_id,
                "start_pos":offset,
                "end_pos":offset+length,
                "type":_type,
                "identifier":identifier,
                "score":score
            })
        pbar.update(1)
    pbar.close()
    return output

with open("bela_experiments/DZ/v0.1/paragraphs_test.csv", "r", encoding="utf-8") as f1:
    data = csv.DictReader(f1)
    data = list(data)

with open("bela_experiments/DZ/v0.1/annotations_test.csv", "r", encoding="utf-8") as f2:
    annotations = csv.DictReader(f2)
    annotations = list(annotations)

output = process_data(data, annotations)

keyz = output[0].keys()
with open("bela_experiments/DZ_output.csv", "w", encoding="utf-8") as f:
    dict_writer = csv.DictWriter(f, keyz)
    dict_writer.writeheader()
    dict_writer.writerows(output)



