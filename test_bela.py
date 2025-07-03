import csv
from multiel import BELA
from tqdm import tqdm


BELA(
    md_threshold=0.2,
    el_threshold=0.4,
    checkpoint_name="wiki",
    device="cuda:0",
    config_name="joint_el_mel_new",
    repo="wannaphong/BELA"
)

def process_data(paragraphs, annotations):
    pbar = tqdm(total=len(paragraphs))
    output = []
    for paragraph in paragraphs:
        doc_id = paragraph["doc_id"]
        text = paragraph["text"]
        annotations = [row for row in annotations if row["doc_id"]==doc_id]
        offsets = [int(anno["start_pos"]) for anno in annotations]
        lengths = [len(anno["surface"]) for anno in annotations]

        results = BELA.process_disambiguation_batch(
            texts=[text],
            mention_offsets=[offsets],
            mention_lengths=[lengths]
        )

        print(results)
        pbar.update(1)
    pbar.close()
    return output

with open("DZ/v1.0/paragraphs_test.csv", "r", encoding="utf-8") as f1:
    data = csv.DictReader(f1)
    data = list(data)

with open("DZ/v1.0/annotations_test.csv", "r", encoding="utf-8") as f2:
    annotations = csv.DictReader(f2)
    annotations = list(annotations)

output = process_data(data[:2], annotations)

# keyz = output[0].keys()
# with open("bela_experiments/DZ_output.csv", "w", encoding="utf-8") as f:
#     dict_writer = csv.DictWriter(f, keyz)
#     dict_writer.writeheader()
#     dict_writer.writerows(output)
