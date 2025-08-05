import os
import csv
import argparse


# Example commands:
# path_data = "../test_data/HIPE_EN"
# path_results = "../results/hipe_en"


def eval_ed(data, predictions):
    tp = []
    fp = []
    fn = []
    for entity1 in data:
        wb_id1 = entity1["identifier"].strip()
        if not wb_id1.startswith("Q"):
            wb_id1 = "NIL"
        for entity2 in predictions:
            if entity1["doc_id"]==entity2["doc_id"] and entity1["start_pos"]==entity2["start_pos"]:
                wb_id2 = entity2["identifier"].strip()
                if wb_id2 == wb_id1:
                    tp.append(entity2)
                else:
                    fp.append(entity2)
                    fn.append(entity1)
    accuracy = (len(tp) / (len(tp) + len(fp)))*100
    return tp, fp, fn, accuracy


def main():
    parser = argparse.ArgumentParser(description="Script for computing micro averaged accuracy in Entity Disambiguation")
    parser.add_argument("--path_results", type=str, required=True, help="Path to JSON list of candidates")
    parser.add_argument("--path_data", type=str, required=True, help="Path to dataset directory")
    args = parser.parse_args()

    with open(os.path.join(args.path_results, "output.csv"), "r", encoding="utf-8") as f1:
        predictions = list(csv.DictReader(f1, delimiter=","))

    with open(os.path.join(args.path_data, "annotations_test.csv"), "r", encoding="utf-8") as f2:
        data = list(csv.DictReader(f2, delimiter=","))

    tp, fp, fn, accuracy = eval_ed(data, predictions)

    with open(os.path.join(args.path_results, "result.txt"), "w") as output:
        output.write("True Positives: " + str(len(tp)) + "\n\n")
        output.write("False Positives: " + str(len(fp)) + "\n\n")
        output.write("False Negatives: " + str(len(fn)) + "\n\n")
        output.write("Accuracy: " + str(accuracy) + "\n\n")

    p_keys = tp[0].keys()
    fp_keys = fp[0].keys()
    n_keys = fn[0].keys()

    tp_file = open(os.path.join(args.path_results, "tp_ed.csv"), "w", encoding="utf-8")
    dict_writer = csv.DictWriter(tp_file, p_keys)
    dict_writer.writeheader()
    dict_writer.writerows(tp)
    tp_file.close()

    fp_file = open(os.path.join(args.path_results, "fp_ed.csv"), "w", encoding="utf-8")
    dict_writer = csv.DictWriter(fp_file, fp_keys)
    dict_writer.writeheader()
    dict_writer.writerows(fp)
    fp_file.close()

    fn_file = open(os.path.join(args.path_results, "fn_ed.csv"), "w", encoding="utf-8")
    dict_writer = csv.DictWriter(fn_file, n_keys)
    dict_writer.writeheader()
    dict_writer.writerows(fn)
    fn_file.close()



if __name__ == "__main__":
    main()