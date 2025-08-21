import json
import csv
import re

tag2type = {
  "PER": [
    "person",
    "person (fictional character)"
  ],
  "ORG": [
    "family",
    "organization",
    "faction",
    "school",
    "government-organization",
    "university",
    "newspaper",
    "journal",
    "magazine",
    "company",
    "military",
    "college",
    "religious-group",
    "empire",
    "band",
    "society"
  ],
  "LOC": [
    "city",
    "country",
    "country region",
    "continent",
    "location",
    "mountain",
    "road",
    "lake",
    "island",
    "building",
    "worship-place",
    "facility",
    "theater",
    "museum",
    "city-district",
    "local-region",
    "country-region",
    "street",
    "square",
    "province",
    "river",
    "village",
    "park",
    "hall"
  ],
  "MISC": [
    "book",
    "books",
    "work-of-art",
    "publication",
    "music",
    "music key",
    "award",
    "event",
    "festival",
    "court decision",
    "war",
    "conference",
    "law",
    "opera",
    "language",
    "thing",
    "symphony",
    "song",
    "concert",
    "scholarship"
  ]
}


type2tag = {}
for k, v in tag2type.items():
    for _c in v:
        type2tag[_c] = k




def convert_json_to_csv(json_file_path, sentences_csv_path, annotations_csv_path):
    """
    Convert JSON entity linking annotations to two CSV files:
    1. sentences.csv - contains document ID and text
    2. annotations.csv - contains entity annotations with positions
    """
    tot_entities = 0
    # Read the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Prepare data for CSV files
    sentences_data = []
    annotations_data = []

    for i, item in enumerate(data):
        # Generate document ID (you can modify this pattern as needed)
        doc_id = f"document_{i:06d}.html"

        sentence = item['sentence']
        entities = item.get('entities', [])

        # Add sentence to sentences data
        sentences_data.append({
            'doc_id': doc_id,
            'text': sentence,
            "publication_date":"1900",
            "lang":"en",
            "genre":"music_periodicals"
        })

        for entity in entities:
            span = entity['span']
            entity_type = entity['type']
            uri = entity.get('uri', "NIL")
            if uri:
                if uri.startswith("("):
                    uri = uri[1:-1]
            else:
                uri = "NIL"

            tot_entities += 1

            start_pos = 0
            while True:
                pos = sentence.find(span, start_pos)
                if pos == -1:
                    break

                end_pos = pos + len(span)

                # Convert type to uppercase format as shown in example
                type_code = type2tag[entity_type]

                annotations_data.append({
                    'doc_id': doc_id,
                    'start_pos': pos,
                    'end_pos': end_pos,
                    'surface': span,
                    'type': type_code,
                    'identifier': uri
                })

                start_pos = pos + 1

    filtered_annotations = []

    positions_dict = dict()

    for row in annotations_data:
        _id = row["doc_id"]
        if _id not in positions_dict:
            positions_dict[_id] = []
        if int(row["start_pos"]) not in positions_dict[_id]:
            positions_dict[_id].append(int(row["start_pos"]))
            filtered_annotations.append(row)
        else:
            continue

    # Write sentences.csv
    with open(sentences_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['doc_id', 'text'])  # Header
        for row in sentences_data:
            writer.writerow([row['doc_id'], row['text']])

    # Write annotations.csv
    with open(annotations_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['doc_id', 'start_pos', 'end_pos', 'surface', 'type', 'identifier'])  # Header
        for row in filtered_annotations:
            writer.writerow([
                row['doc_id'],
                row['start_pos'],
                row['end_pos'],
                row['surface'],
                row['type'],
                row['identifier']
            ])

    print(f"Conversion completed!")
    print(f"Created {len(sentences_data)} sentences in {sentences_csv_path}")
    print(f"Created {len(annotations_data)} annotations in {annotations_csv_path}")
    print(f"Entities in the original dataset: {tot_entities}")




# Example usage
if __name__ == "__main__":
    # Basic usage
    json_file = "../KE-MHISTO/Datasets/MHERCL_EN.json"  # Your input JSON file
    sentences_output = "./test_data/MHERCL_EN/paragraphs_test.csv"
    annotations_output = "./test_data/MHERCL_EN/annotations_test.csv"

    # Use the basic version
    convert_json_to_csv(json_file, sentences_output, annotations_output)
