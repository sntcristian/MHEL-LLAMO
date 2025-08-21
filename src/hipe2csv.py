import csv
import re
import os

filepath = "../archive/HIPE-2022-data/data/v2.1/newseye/fi/HIPE-2022-v2.1-newseye-test-fi.tsv"

with open(filepath, "r", encoding="utf-8") as f:
    tsv_data = f.readlines()
    f.close()

text = ""
curr_pos = 0
named_entity = False
publication_date = ""
doc_id = ""
title = ""
genre = "press articles"
lang = ""

paragraphs = []
annotations = []


nerc_tag2type = {"B-PER":"PER","B-LOC":"LOC", "B-HumanProd":"PROD", "I-PER":"PER", "I-LOC":"LOC", "I-HumanProd":"PROD",
                 "B-ORG":"ORG", "I-ORG":"ORG"}

for line in tsv_data[1:]:
    if line.startswith("# hipe2022:date = "):
        publication_date = line.split("# hipe2022:date = ")[-1].strip()
    elif line.startswith("# hipe2022:document_id ="):
        if len(text) > 0:
            paragraph = {
                "doc_id": doc_id,
                "title": title,
                "text": text.strip(),
                "publication_date": publication_date, 
                "lang":lang,
                "genre":genre
            }
            paragraphs.append(paragraph)

        if named_entity == True:
            if q_id.startswith("Q") or q_id.startswith("NIL"):
                annotation = {
                    "doc_id": doc_id,
                    "surface": text[start_pos:end_pos],
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "type": ner_type,
                    "identifier": q_id
                }
                annotations.append(annotation)
            named_entity = False
        doc_id = line.split("# hipe2022:document_id =")[-1].strip()
        text = ""
        curr_pos = 0

    
    elif line.startswith("# hipe2022:language ="):
        lang = line.split("# hipe2022:language =")[-1].strip()

    elif line.startswith("#"):
        continue
    else:
        if len(line.split("\t"))==10:
            cols = line.split("\t")
            token = cols[0].strip()
            nerc_tag = cols[1].strip()
            nel_tag = cols[7].strip()
            line_info = cols[9].strip()
            text+=token
            if nerc_tag in nerc_tag2type:
                if named_entity != True:
                    named_entity = True
                    start_pos = curr_pos
                ner_type = nerc_tag2type[nerc_tag]
                end_pos = curr_pos + len(token)
                q_id = nel_tag.strip()
            else:
                if named_entity == True:
                    if q_id.startswith("Q") or q_id.startswith("NIL"):
                        annotation = {
                            "doc_id":doc_id,
                            "surface":text[start_pos:end_pos],
                            "start_pos":start_pos,
                            "end_pos":end_pos,
                            "type":ner_type,
                            "identifier":q_id
                        }
                        annotations.append(annotation)
                    named_entity = False

            if "NoSpaceAfter" in line_info:
                curr_pos = len(text)
            else:
                text += " "
                curr_pos = len(text)



sent_idx = set([item["doc_id"] for item in annotations])
filtered_paragraphs = []
for item in paragraphs:
    if item["doc_id"] in sent_idx:
        filtered_paragraphs.append(item)

p_keys = filtered_paragraphs[0].keys()
a_keys = annotations[0].keys()
with open("./test_data/NEWSEYE_FI/paragraphs_test.csv", "w", encoding="utf-8") as f:
    dict_writer = csv.DictWriter(f, p_keys)
    dict_writer.writeheader()
    dict_writer.writerows(filtered_paragraphs)

with open("./test_data/NEWSEYE_FI/annotations_test.csv", "w", encoding="utf-8") as f:
    dict_writer = csv.DictWriter(f, a_keys)
    dict_writer.writeheader()
    dict_writer.writerows(annotations)