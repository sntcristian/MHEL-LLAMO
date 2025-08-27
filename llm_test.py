import json
import csv
import transformers
import torch
import re
import os
import argparse
from datetime import datetime
import random


def process_candidates(candidates, n_candidates):
    output = []
    for item in candidates[:min(n_candidates, len(candidates))]:
        output.append({
            "wikipedia_page": item["label"],
            "wikidata_id": item["wb_id"],
            "type": item["type"],
            "descr":item["descr"],
            "date": item["min_date"],
        })
    return output


def main():
    parser = argparse.ArgumentParser(description="LLM Test on Single Entity Disambiguation Example")
    parser.add_argument("--json_f", type=str, required=True, help="Path to JSON list of candidates")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Huggingface repo of LLM")
    parser.add_argument("--hf_token", type=str, default="", help="Huggingface token to access restricted repo.")
    parser.add_argument("--n_candidates", type=int, default=50, help="Number of candidates to put in prompt.")

    args = parser.parse_args()
    
    print(f"Testing model: {args.model_id}")
    
    # Load data
    with open(args.json_f, "r", encoding="utf-8") as f:
        retriever_results = json.load(f)

    with open(os.path.join(args.dataset_path, "paragraphs_test.csv"), "r", encoding="utf-8") as f:
        paragraphs = list(csv.DictReader(f))

    iso_to_lang = {"en":"English", "it":"Italian", "fr":"French", "sv":"Swedish", "de":"German", "fi":"Finnish",
                   "nl":"Dutch"}

    print("Loading pipeline...")
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        token=args.hf_token if args.hf_token else None
    )

    system_prompt = """
    You are an effective multilingual information extraction system specialized in disambiguating entities within noisy 
    historical texts.
    Your task is to analyse the text provided by the user and disambiguate the reference marked by [ENT] tags by 
    selecting a Wikidata entity from a given list of candidates only when highly confident, classifying it with a NIL value otherwise.
    Always respond by returning a JSON-formatted answer; do not generate Python code.
    """

    # Process only the first item
    item = random.sample(retriever_results, 1)[0]
    doc_id = item["doc_id"]
    start_pos = int(item["start_pos"])
    end_pos = int(item["end_pos"])
    date = [p for p in paragraphs if p["doc_id"]==doc_id][0]["publication_date"]
    lang = iso_to_lang[[p for p in paragraphs if p["doc_id"]==doc_id][0]["lang"]]
    genre = [p for p in paragraphs if p["doc_id"]==doc_id][0]["genre"]
    text = [p for p in paragraphs if p["doc_id"]==doc_id][0]["text"]
    processed_text = text[max(0, start_pos - 500):start_pos] + "[ENT] " + text[start_pos:end_pos] + " [ENT] " +text[end_pos:min(len(text), end_pos + 500)]
    processed_candidates = process_candidates(item["candidates"], args.n_candidates)
    
    user_prompt = """
    Read the input text extracted from """ + lang + " " + genre + " published in " + date + """.
    Disambiguate the entity mentioned between the [ENT] tags by selecting the most appropriate Wikidata entity from the list of candidates.    
    Return the corresponding Wikipedia page title, Wikidata ID of the selected entity and a score measuring the confidence of your choice in a JSON object formatted as follows:

    ```json
    {"wikipedia_page":"", "wikidata_id":"", "score": int}
    ```
    The confidence score should be on a scale from 1 to 5 where 1 is "no confidence", 2 is "slightly confident", 3 is "undecided", 4 is "confident" and 5 is "highly confident".
    Make sure to select both the Wikidata ID and the Wikipedia page title from the provided list of candidates.
    Pay attention that the list of candidates may not include the entity mentioned. If none of the candidates match with high confidence the entity tagged with [ENT], use the string "NIL" as value of the "wikidata_id" key.
    ---------------------
    Input Text:
    """ + processed_text + """
    ---------------------
    JSON List of Candidates:
    ```json
    """ + str(processed_candidates) + """ 
    ``` ."""
    
    print(f"Testing entity: {item['surface']}")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    print("Generating response...")
    
    outputs = pipeline(
        messages,
        max_new_tokens=512,
    )
    response = outputs[0]["generated_text"][-1]["content"]
    
    # Extract wikidata_id
    match = re.search(r'"wikidata_id"\s*:\s*"(Q\d+|NIL)"', response)
    if match:
        wikidata_id = match.group(1)
        print(f"Found Wikidata ID: {wikidata_id}")
    else:
        print("No Wikidata ID found")
        wikidata_id = "NIL"

    # Prepare output
    selected_entity = [x for x in item["candidates"] if x["wb_id"] == wikidata_id]
    if len(selected_entity) > 0:
        result = {
            "model": args.model_id,
            "timestamp": datetime.now().isoformat(),
            "doc_id": doc_id,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "surface": item["surface"],
            "gt_id": item["identifier"],
            "type": item["type"],
            "identifier": wikidata_id,
            "title": selected_entity[0]["label"],
            "answer": re.sub(r'\s+', " ", response),
            "score": selected_entity[0]["score"],
            "language": lang,
            "genre": genre,
            "date": date
        }
    else:
        result = {
            "model": args.model_id,
            "timestamp": datetime.now().isoformat(),
            "doc_id": doc_id,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "surface": item["surface"],
            "gt_id": item["identifier"],
            "type": item["type"],
            "identifier": "NIL",
            "title": item["surface"],
            "answer": re.sub(r'\s+', " ", response),
            "score": 0,
            "language": lang,
            "genre": genre,
            "date": date
        }
    
    # Save result
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model_id.replace("/", "_").replace("-", "_")
    output_file = os.path.join(args.output_dir, f"single_test_{model_name}.json")
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(result, out_f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")
    print(f"Ground Truth: {item['identifier']}")
    print(f"Predicted: {wikidata_id}")
    print(f"Correct: {item['identifier'] == wikidata_id}")

if __name__ == "__main__":
    main()