import json
import sqlite3
import numpy as np
import re
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
import multiprocessing as mp

# File paths
index_path = './models/index.txt'
db_path = './models/knowledge_base.sqlite'
type2class_path = "./models/type2classes.json"
rdf1_path = "./models/types_and_dates.rdf"
rdf2_path = "./models/labels.rdf"
rdf3_path = "./models/descriptions.rdf"


# Load mappings once
with open(type2class_path, "r", encoding="utf-8") as f2:
    type2class = json.load(f2)

class2type = {}
for k, v in type2class.items():
    for _c in v:
        class2type[_c] = k


def preprocess_types_and_dates(rdf_path, required_qids):
    """
    Pre-process RDF file to extract only relevant information for required QIDs.
    Returns dictionaries mapping QID to types and dates.
    """
    qid_to_types = defaultdict(list)
    qid_to_dates = defaultdict(list)

    # Compile regex patterns once for better performance
    type_pattern = re.compile(r'wd:(Q\d+)\s+wdt:P31\s+wd:(Q\d+) \.')
    date_pattern = re.compile(r'wd:(Q\d+)\s+wdt:P\d+\s+"(.*?)"\^\^xsd:dateTime \.')

    print("Preprocessing RDF file...")
    with open(rdf_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing RDF lines"):

            if line.startswith('wd:Q'):
                qid_match = re.match(r'wd:(Q\d+) ', line)
                if qid_match:
                    line_qid = qid_match.group(1)
                    if line_qid not in required_qids:
                        continue
            else:
                continue

            # Extract type information
            type_match = type_pattern.match(line)
            if type_match:
                entity_qid, class_qid = type_match.groups()
                if class_qid in class2type:
                    qid_to_types[entity_qid].append(class2type[class_qid])

            # Extract date information
            date_match = date_pattern.match(line)
            if date_match:
                entity_qid, date_str = date_match.groups()
                qid_to_dates[entity_qid].append(date_str)

    return qid_to_types, qid_to_dates




def preprocess_labels(rdf_path, required_qids):
    """
    Pre-process RDF file to extract only relevant information for required QIDs.
    Returns dictionaries mapping QID to types and dates.
    """
    qid_to_labels = defaultdict(dict)

    # Compile regex patterns once for better performance
    labels_pattern = re.compile(r'wd:(Q\d+)\s+schema:name\s+"(.*?)"@([a-z]+)\s*\.')

    print("Preprocessing RDF file...")
    with open(rdf_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing RDF lines"):

            match = re.match(labels_pattern, line)

            if match:
                wikidata_entity, label, language = match.groups()
                if wikidata_entity in required_qids:
                    qid_to_labels[wikidata_entity][language] = label

    return qid_to_labels


def preprocess_descriptions(rdf_path, required_qids):
    """
    Pre-process RDF file to extract only relevant information for required QIDs.
    Returns dictionaries mapping QID to types and dates.
    """
    qid_to_descriptions = defaultdict(dict)

    # Compile regex patterns once for better performance
    descriptions_pattern = re.compile(r'wd:(Q\d+)\s+schema:description\s+"(.*?)"@([a-z]+)\s*\.')

    print("Preprocessing RDF file...")
    with open(rdf_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing RDF lines"):

            match = re.match(descriptions_pattern, line)

            if match:
                wikidata_entity, description, language = match.groups()
                if wikidata_entity in required_qids:
                    qid_to_descriptions[wikidata_entity][language] = description

    return qid_to_descriptions

def determine_entity_type(type_list):
    """Determine the most common entity type from a list of types."""
    if not type_list:
        return None

    entity_type_count = {
        "PER": 0,
        "LOC": 0,
        "WORK": 0,
        "ORG": 0
    }

    for entity_type in type_list:
        if entity_type in entity_type_count:
            entity_type_count[entity_type] += 1

    max_count = max(entity_type_count.values())
    if max_count == 0:
        return None

    for tag, count in entity_type_count.items():
        if count == max_count:
            return tag

    return None


def parse_date_string(date_str):
    """Parse a date string and return a numpy datetime64 object."""
    date_part = date_str.split("T")[0]
    sign = date_part[:1]
    actual_date = date_part[1:]
    try:
        parts = actual_date.split("-")
        if len(parts) == 3:
            year, month, day = parts
            # Handle invalid months/days
            if month == "00":
                month = "01"
            if day == "00":
                day = "01"

            if sign == '-':
                formatted_date = f"-{year}-{month}-{day}"
            else:
                formatted_date = f"{year}-{month}-{day}"

            return np.datetime64(formatted_date)

    except Exception:
            # Fallback to year-01-01
        year = actual_date.split("-")[0]
        if sign == '-':
            return np.datetime64(f"-{year}-01-01")
        else:
            return np.datetime64(f"{year}-01-01")


def find_minimum_date(date_list):
    """Find the minimum date from a list of date strings."""
    if not date_list:
        return None

    valid_dates = []
    for date_str in date_list:
        parsed_date = parse_date_string(date_str)
        if parsed_date is not None:
            valid_dates.append(parsed_date)

    return min(valid_dates) if valid_dates else None


def process_entity_batch(batch_data):
    """Process a batch of entities - can be used for multiprocessing."""
    entities, qid_to_types, qid_to_dates, qid_to_labels, qid_to_descriptions = batch_data

    results = {
        "entities":[],
        "enwiki":[],
        "dewiki":[],
        "frwiki":[],
        "itwiki":[],
        "nlwiki":[],
        "svwiki":[],
        "fiwiki":[]
    }
    for idx, wikidata_qid in entities:

            # Get type and date from preprocessed data
            entity_types = qid_to_types.get(wikidata_qid, [])
            entity_dates = qid_to_dates.get(wikidata_qid, [])

            _type = determine_entity_type(entity_types)
            min_date = find_minimum_date(entity_dates)

            if isinstance(min_date, np.datetime64):
                min_date = str(min_date)

            results["entities"].append(
                (idx, wikidata_qid, _type, min_date)
            )

            enwiki = qid_to_labels.get(wikidata_qid, dict()).get("en", "")
            endescr = qid_to_descriptions.get(wikidata_qid, dict()).get("en", "")
            if not (len(enwiki) == 0 and len(endescr) == 0):
                results["enwiki"].append((idx, enwiki, endescr))

            dewiki = qid_to_labels.get(wikidata_qid, dict()).get("de", "")
            dedescr = qid_to_descriptions.get(wikidata_qid, dict()).get("de", "")
            if not (len(dewiki) == 0 and len(dedescr) == 0):
                results["dewiki"].append((idx, dewiki, dedescr))

            frwiki = qid_to_labels.get(wikidata_qid, dict()).get("fr", "")
            frdescr = qid_to_descriptions.get(wikidata_qid, dict()).get("fr", "")
            if not (len(frwiki) == 0 and len(frdescr) == 0):
                results["frwiki"].append((idx, frwiki, frdescr))

            itwiki = qid_to_labels.get(wikidata_qid, dict()).get("it", "")
            itdescr = qid_to_descriptions.get(wikidata_qid, dict()).get("it", "")
            if not (len(itwiki) == 0 and len(itdescr) == 0):
                results["itwiki"].append((idx, itwiki, itdescr))

            nlwiki = qid_to_labels.get(wikidata_qid, dict()).get("nl", "")
            nldescr = qid_to_descriptions.get(wikidata_qid, dict()).get("nl", "")
            if not (len(nlwiki) == 0 and len(nldescr) == 0):
                results["nlwiki"].append((idx, nlwiki, nldescr))

            svwiki = qid_to_labels.get(wikidata_qid, dict()).get("sv", "")
            svdescr = qid_to_descriptions.get(wikidata_qid, dict()).get("sv", "")
            if not (len(svwiki) == 0 and len(svdescr) == 0):
                results["svwiki"].append((idx, svwiki, svdescr))

            fiwiki = qid_to_labels.get(wikidata_qid, dict()).get("fi", "")
            fidescr = qid_to_descriptions.get(wikidata_qid, dict()).get("fi", "")
            if not (len(fiwiki) == 0 and len(fidescr) == 0):
                results["fiwiki"].append((idx, fiwiki, fidescr))

    return results


def main():
    # Step 1: Load Wikipedia to Wikidata mappings
    print("Loading entities...")
    entities = []
    required_qids = set()
    with open(index_path, 'r', encoding='utf-8') as txt_file:
        for idx, line in enumerate(txt_file):
            if line.strip().startswith("Q"):
                entities.append((idx, line.strip()))
                required_qids.add(line.strip())  # Remove 'Q' prefix

    print(f"Found {len(entities)} entities, {len(required_qids)} unique QIDs")


    # Step 3: Preprocess RDF file
    qid_to_types, qid_to_dates = preprocess_types_and_dates(rdf1_path, required_qids)
    qid_to_labels = preprocess_labels(rdf2_path, required_qids)
    qid_to_descriptions = preprocess_descriptions(rdf3_path, required_qids)

    # Step 4: Set up database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY,
        wikidata_qid TEXT,
        type_ TEXT,
        min_date TEXT
    )''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS enwiki (
            id INTEGER PRIMARY KEY,
            label TEXT,
            descr TEXT,
            FOREIGN KEY (id) REFERENCES entities(id)
        )''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dewiki (
            id INTEGER PRIMARY KEY,
            label TEXT,
            descr TEXT,
            FOREIGN KEY (id) REFERENCES entities(id)
        )''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS frwiki (
            id INTEGER PRIMARY KEY,
            label TEXT,
            descr TEXT,
            FOREIGN KEY (id) REFERENCES entities(id)
        )''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS itwiki (
            id INTEGER PRIMARY KEY,
            label TEXT,
            descr TEXT,
            FOREIGN KEY (id) REFERENCES entities(id)
        )''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS nlwiki (
            id INTEGER PRIMARY KEY,
            label TEXT,
            descr TEXT,
            FOREIGN KEY (id) REFERENCES entities(id)
        )''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS svwiki (
            id INTEGER PRIMARY KEY,
            label TEXT,
            descr TEXT,
            FOREIGN KEY (id) REFERENCES entities(id)
        )''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fiwiki (
            id INTEGER PRIMARY KEY,
            label TEXT,
            descr TEXT,
            FOREIGN KEY (id) REFERENCES entities(id)
        )''')

    # Step 5: Process entities (with optional multiprocessing)
    use_multiprocessing = len(entities) > 1000  # Only use for large datasets

    if use_multiprocessing:
        print("Processing entities with multiprocessing...")
        # Split entities into chunks for multiprocessing
        num_processes = min(mp.cpu_count(), 4)  # Limit to 4 processes
        chunk_size = len(entities) // num_processes
        chunks = []

        for i in range(0, len(entities), chunk_size):
            chunk = entities[i:i + chunk_size]
            chunks.append((chunk, qid_to_types, qid_to_dates, qid_to_labels, qid_to_descriptions))

        all_results = {"entities":[],
                       "enwiki":[],
                       "dewiki":[],
                       "frwiki":[],
                       "itwiki":[],
                       "nlwiki":[],
                       "svwiki":[],
                       "fiwiki":[]
                       }

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(process_entity_batch, chunk) for chunk in chunks]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing chunks"):
                all_results["entities"] += future.result()["entities"]
                all_results["enwiki"] += future.result()["enwiki"]
                all_results["dewiki"] += future.result()["dewiki"]
                all_results["frwiki"] += future.result()["frwiki"]
                all_results["itwiki"] += future.result()["itwiki"]
                all_results["nlwiki"] += future.result()["nlwiki"]
                all_results["svwiki"] += future.result()["svwiki"]
                all_results["fiwiki"] += future.result()["fiwiki"]
    else:
        print("Processing entities sequentially...")
        all_results = process_entity_batch((entities, qid_to_types, qid_to_dates, qid_to_labels, qid_to_descriptions))

    # Step 6: Insert into database in batches
    print("Inserting into database...")
    batch_size = 1000
    print("Inserting into entities table...")
    for i in tqdm(range(0, len(all_results["entities"]), batch_size), desc="Inserting entities batches"):
        batch = all_results["entities"][i:i + batch_size]
        cursor.executemany('''
        INSERT INTO entities (id, wikidata_qid, type_, min_date)
        VALUES (?, ?, ?, ?)
        ''', batch)
        conn.commit()

    for i in tqdm(range(0, len(all_results["enwiki"]), batch_size), desc="Inserting enwiki batches"):
        batch = all_results["enwiki"][i:i + batch_size]
        cursor.executemany('''
        INSERT INTO enwiki (id, label, descr)
        VALUES (?, ?, ?)
        ''', batch)
        conn.commit()

    for i in tqdm(range(0, len(all_results["dewiki"]), batch_size), desc="Inserting dewiki batches"):
        batch = all_results["dewiki"][i:i + batch_size]
        cursor.executemany('''
        INSERT INTO dewiki (id, label, descr)
        VALUES (?, ?, ?)
        ''', batch)
        conn.commit()

    for i in tqdm(range(0, len(all_results["frwiki"]), batch_size), desc="Inserting frwiki batches"):
        batch = all_results["frwiki"][i:i + batch_size]
        cursor.executemany('''
        INSERT INTO frwiki (id, label, descr)
        VALUES (?, ?, ?)
        ''', batch)
        conn.commit()

    for i in tqdm(range(0, len(all_results["itwiki"]), batch_size), desc="Inserting itwiki batches"):
        batch = all_results["itwiki"][i:i + batch_size]
        cursor.executemany('''
        INSERT INTO itwiki (id, label, descr)
        VALUES (?, ?, ?)
        ''', batch)
        conn.commit()

    for i in tqdm(range(0, len(all_results["nlwiki"]), batch_size), desc="Inserting nlwiki batches"):
        batch = all_results["nlwiki"][i:i + batch_size]
        cursor.executemany('''
        INSERT INTO nlwiki (id, label, descr)
        VALUES (?, ?, ?)
        ''', batch)
        conn.commit()

    for i in tqdm(range(0, len(all_results["svwiki"]), batch_size), desc="Inserting svwiki batches"):
        batch = all_results["svwiki"][i:i + batch_size]
        cursor.executemany('''
        INSERT INTO svwiki (id, label, descr)
        VALUES (?, ?, ?)
        ''', batch)
        conn.commit()

    for i in tqdm(range(0, len(all_results["fiwiki"]), batch_size), desc="Inserting fiwiki batches"):
        batch = all_results["fiwiki"][i:i + batch_size]
        cursor.executemany('''
        INSERT INTO fiwiki (id, label, descr)
        VALUES (?, ?, ?)
        ''', batch)
        conn.commit()  # Commit each batch

    # Final commit and close
    conn.commit()
    conn.close()

    print(f"Database '{db_path}' created successfully with {len(all_results)} entities.")


if __name__ == "__main__":
    main()