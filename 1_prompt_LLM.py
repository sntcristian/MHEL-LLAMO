import json
import csv
import transformers
import torch
import re
from tqdm import tqdm
import os
import argparse

# Example commands:
# output_dir = "./results/ajmc_en"
# json_f = "./results/ajmc_en/candidates_top50.json"
# dataset_path = "./test_data/AJMC_EN"
# model_id = "meta-llama/Llama-3.1-8B-Instruct"

def process_candidates(candidates):
    output = []
    for item in candidates:
        output.append({
            "wikipedia_page": item["label"],
            "wikidata_id": item["wb_id"],
            "type": item["type"],
            "descr":item["descr"],
            "date": item["min_date"],
        })
    return output


def main():
    parser = argparse.ArgumentParser(description="LLM Prompting for Entity Disambiguation from Candidate List")
    parser.add_argument("--json_f", type=str, required=True, help="Path to JSON list of candidates")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Huggingface repo of LLM")
    parser.add_argument("--hf_token", type=str, default="", help="Huggingface token to access restricted repo.")

    args = parser.parse_args()
    with open(args.json_f, "r", encoding="utf-8") as f:
        retriever_results = json.load(f)

    with open(os.path.join(args.dataset_path, "paragraphs_test.csv", ), "r", encoding="utf-8") as f:
        paragraphs = list(csv.DictReader(f))

    iso_to_lang = {"en":"English", "it":"Italian", "fr":"French", "sv":"Swedish", "de":"German", "fi":"Finnish",
                   "nl":"Dutch"}

    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        token=args.hf_token
    )

    system_prompt = """
    You are an effective multilingual information extraction system specialized in disambiguating entities within noisy 
    historical texts.
    Your task is to analyse the text provided by the user and disambiguate the reference marked by [ENT] tags by selecting a Wikidata entity from a given list of candidates.
    Always respond by returning a JSON-formatted answer; do not generate Python code.
    """

    output = []

    for item in tqdm(retriever_results):
        doc_id = item["doc_id"]
        start_pos = int(item["start_pos"])
        end_pos = int(item["end_pos"])
        date = [p for p in paragraphs if p["doc_id"]==doc_id][0]["publication_date"]
        lang = iso_to_lang[[p for p in paragraphs if p["doc_id"]==doc_id][0]["lang"]]
        text = [p for p in paragraphs if p["doc_id"]==doc_id][0]["text"]
        processed_text = text[:start_pos] + "[ENT] " + text[start_pos:end_pos] + " [ENT] " + text[end_pos:]
        processed_candidates = process_candidates(item["candidates"])
        user_prompt = """
        Read the input text published in """ + date + """ and written in """ + lang + """ .
        Disambiguate the entity mentioned between the [ENT] tags by selecting the most appropriate Wikidata entity from the list of candidates.    
        Return the corresponding Wikipedia page title and Wikidata ID of the selected entity in a JSON object formatted as 
        follows:
    
        ```json
        "wikipedia_page":"", "wikidata_id":""
        ```
    
        Make sure to select both the Wikidata ID from the provided list of candidates.
        If none of the candidates match the entity tagged with [ENT], return an empty JSON object.
        ---------------------
        Input Text:
        """ + processed_text + """
        ---------------------
        JSON List of Candidates:
        ```json
        """ + str(processed_candidates) + """ 
        ``` ."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=512,
        )
        response = outputs[0]["generated_text"][-1]["content"]
        match = re.search(r'"wikidata_id"\s*:\s*"(Q\d+)"', response)

        if match:
            wikidata_id = match.group(1)
            print(f"Found Wikidata ID: {wikidata_id} for entity {item['surface']}")

        else:
            print(f"No Wikidata ID found for entity {item['surface']}.")
            wikidata_id = "NIL"
            print(response)

        selected_entity = [x for x in item["candidates"] if x["wb_id"] == wikidata_id]
        if len(selected_entity)>0:
            output.append({
                "doc_id":doc_id,
                "start_pos":start_pos,
                "end_pos":end_pos,
                "surface":item["surface"],
                "identifier":wikidata_id,
                "gt_id":item["identifier"],
                "title":selected_entity[0]["label"],
                "answer":re.sub(r'\s+', " ", response),
                "score":selected_entity[0]["score"]
            })
        else:
            output.append({
                "doc_id": doc_id,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "surface": item["surface"],
                "identifier": "NIL",
                "gt_id": item["identifier"],
                "title": item["surface"],
                "answer": re.sub(r'\s+', " ", response),
                "score": 0
            })
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "output.csv"), "w", encoding="utf-8") as out_f:
        dict_writer = csv.DictWriter(out_f, output[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(output)

if __name__ == "__main__":
    main()


