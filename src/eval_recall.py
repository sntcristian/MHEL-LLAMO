import json


# path_results = "../results/candidates_top50.json"

def read_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def recall_k(data, k: int):
    correct = 0
    total = 0
    skipped_nil = 0

    for item in data:
        identifier = item.get("identifier")
        candidates = item.get("candidates", [])

        if identifier != "NIL":
            total += 1
            # get top-k candidates' wb_id
            top_k = [cand.get("wb_id") for cand in candidates[:k]]

            if identifier in top_k:
                correct += 1
        # If identifier is "NIL" → skip
        else:
            skipped_nil += 1

    recall = correct / total if total > 0 else 0.0
    # print("correct: ",correct, "total: ",total)
    return recall, skipped_nil


data = read_json("results/NEWSEYE_FI/candidates_dev_top50_fi.json")
k_values = [1, 3, 5, 10, 20, 30, 40, 50]

for k in k_values:
    recall, skipped = recall_k(data, k)
    print(f"Recall@{k}: {recall:.4f} (skipped {skipped} NIL cases)")