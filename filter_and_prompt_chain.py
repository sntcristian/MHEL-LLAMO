import json
import csv
import transformers
import torch
import re
from tqdm import tqdm
import os
import argparse


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
    parser = argparse.ArgumentParser(description="LLM Prompting for Entity Disambiguation from Candidate List")
    parser.add_argument("--json_f", type=str, required=True, help="Path to JSON list of candidates")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--threshold", type=float, default=0, help="Threshold to be used to filter hard negatives.")
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-Small-24B-Instruct-2501", help="Huggingface repo of LLM")
    parser.add_argument("--hf_token", type=str, default="", help="Huggingface token to access restricted repo.")
    parser.add_argument("--n_candidates", type=int, default=50, help="Number of candidates to put in prompt.")

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


    system_prompt1 = """
    You are a highly precise multilingual information extraction system specialized in disambiguating entities within noisy historical texts.
    Your task is to analyse the text provided by the user and determine if the reference marked by [ENT] tags can be associated or not to one of the candidate Wikidata entities provided in the JSON list.
    Always respond by saying either "yes" or "no". Do not generate Python code.
    """
    
    system_prompt2 = """
    You are an effective multilingual information extraction system specialized in disambiguating entities within noisy 
    historical texts.
    Your task is to analyse the text provided by the user and disambiguate the reference marked by [ENT] tags by 
    selecting a Wikidata entity from a given list of candidates.
    Always respond by returning a JSON-formatted answer; do not generate Python code.
    """

    output = []
    paragraphs_dict = {p["doc_id"]: p for p in paragraphs}

    for item in tqdm(retriever_results):
        doc_id = item["doc_id"]
        start_pos = int(item["start_pos"])
        end_pos = int(item["end_pos"])
        if args.threshold > 0 and item["candidates"][0]["score"] >= args.threshold:
            output.append({
                "doc_id":doc_id,
                "start_pos":start_pos,
                "end_pos":end_pos,
                "surface":item["surface"],
                "gt_id": item["identifier"],
                "type":item["type"],
                "identifier":item["candidates"][0]["wb_id"],
                "title":item["candidates"][0]["label"],
                "answer":"",
                "score":item["candidates"][0]["score"]
            })
        else:
            paragraph = paragraphs_dict[doc_id]
            date = paragraph["publication_date"]
            lang = iso_to_lang[paragraph["lang"]]
            genre = paragraph["genre"]
            text = paragraph["text"]
            processed_text = text[max(0, start_pos - 500):start_pos] + "[ENT] " + text[start_pos:end_pos] + " [ENT] " +text[end_pos:min(len(text), end_pos + 500)]
            processed_candidates = process_candidates(item["candidates"], args.n_candidates)
            allowed_qids = set([cand["wikidata_id"].upper() for cand in processed_candidates])
            user_prompt1 = """
            Read the input text extracted from """ + lang + " " + genre + " published in " + date + """.
            Answer if the entity mentioned between the [ENT] tags in the input text corresponds to one of the candidate Wikidata entity provided in the json.
            Give a simple binary answer.
            ---------------------
            Input Text:
            """ + processed_text + """
            ---------------------
            JSON List of Candidates:
            ```json
            """ + str(processed_candidates) + """ 
            ``` ."""
            messages = [
                {"role": "system", "content": system_prompt1},
                {"role": "user", "content": user_prompt1},
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=128,
            )
            response = outputs[0]["generated_text"][-1]["content"]
            if "yes" in response.lower():
                user_prompt2 = """
                Read the input text extracted from """ + lang + " " + genre + " published in " + date + """.
                Disambiguate the entity mentioned between the [ENT] tags by selecting the most appropriate Wikidata entity from a given list of candidates.    
                Return the corresponding Wikipedia page title and Wikidata ID of the selected entity in a JSON object formatted as follows:
            
                ```json
                {"wikipedia_page":"", "wikidata_id":""}
                ```
            
                Make sure to select both the Wikidata ID and the Wikipedia page title from the provided list of candidates. 
                Pay attention that the list of candidates may not include the entity mentioned. If none of the candidates match with high confidence the entity tagged with [ENT], return an empty json.

                ---------------------
                Input Text:
                """ + processed_text + """
                ---------------------
                JSON List of Candidates:
                ```json
                """ + str(processed_candidates) + """ 
                ``` ."""
                messages = [
                {"role": "system", "content": system_prompt2},
                {"role": "user", "content": user_prompt2},
                ]

                outputs = pipeline(
                    messages,
                    max_new_tokens=256,
                )

                response = outputs[0]["generated_text"][-1]["content"]
                match = re.search(r'"wikidata_id"\s*:\s*"(Q\d+)"', response)
                if match and match.group(1).upper() in allowed_qids:
                    wikidata_id = match.group(1).upper()
                    selected_entity = [x for x in item["candidates"] if x["wb_id"] == wikidata_id][0]
                    output.append({
                        "doc_id":doc_id,
                        "start_pos":start_pos,
                        "end_pos":end_pos,
                        "surface":item["surface"],
                        "gt_id": item["identifier"],
                        "type":item["type"],
                        "identifier":wikidata_id,
                        "title":selected_entity["label"],
                        "answer":re.sub(r'\s+', " ", response),
                        "score":selected_entity["score"]
                    })

                else:
                    output.append({
                    "doc_id": doc_id,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "surface": item["surface"],
                    "gt_id": item["identifier"],
                    "type": item["type"],
                    "identifier": "NIL",
                    "title": item["surface"],
                    "answer": re.sub(r'\s+', " ", response),
                    "score": 0
                    })
            
            else:
                output.append({
                    "doc_id": doc_id,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "surface": item["surface"],
                    "gt_id": item["identifier"],
                    "type": item["type"],
                    "identifier": "NIL",
                    "title": item["surface"],
                    "answer": re.sub(r'\s+', " ", response),
                    "score": 0
                    })
        
    print(len(retriever_results))
    print(len(output))
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "output.csv"), "w", encoding="utf-8") as out_f:
        dict_writer = csv.DictWriter(out_f, output[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(output)

if __name__ == "__main__":
    main()


