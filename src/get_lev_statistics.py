import csv
from Levenshtein import distance as levenshtein_distance

def compute_levenshtein_for_nil_entities(csv_file_path):
    """
    Read CSV file and compute Levenshtein distances for entities with gt_id == 'NIL'.
    Returns statistics: min, max, median, mean of distances.
    """
    distances = []
    
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            gt_id = row.get('gt_id', '').strip()
            identifier = row.get('identifier', '').strip()
            if gt_id == 'NIL' and identifier != "NIL":
                surface = row.get('surface', '').strip()
                title = row.get('title', '').strip()
                
                # Compute Levenshtein distance (case-insensitive)
                dist = levenshtein_distance(surface.lower(), title.lower())
                distances.append(dist)
                
                print(f"Surface: '{surface}' | Title: '{title}' | Distance: {dist}")
    
    if not distances:
        print("No entities with gt_id == 'NIL' found in the dataset.")
        return None
    
    # Compute statistics
    distances.sort()
    n = len(distances)
    
    min_dist = min(distances)
    max_dist = max(distances)
    mean_dist = sum(distances) / n
    
    # Compute median
    if n % 2 == 0:
        median_dist = (distances[n//2 - 1] + distances[n//2]) / 2
    else:
        median_dist = distances[n//2]
    
    # Print statistics
    print(f"\n{'='*40}")
    print("LEVENSHTEIN DISTANCE STATISTICS")
    print(f"{'='*40}")
    print(f"Total NIL entities: {n}")
    print(f"Minimum distance:   {min_dist}")
    print(f"Maximum distance:   {max_dist}")
    print(f"Mean distance:      {mean_dist:.2f}")
    print(f"Median distance:    {median_dist:.1f}")
    
    return {
        'count': n,
        'min': min_dist,
        'max': max_dist,
        'mean': mean_dist,
        'median': median_dist,
        'distances': distances
    }

# Usage
if __name__ == "__main__":
    # Replace with your CSV file path
    csv_file = "./results/HIPE_FR/llama_3.1_8B_ens_k20_median_fr/output.csv"
    
    try:
        stats = compute_levenshtein_for_nil_entities(csv_file)
        
        if stats:
            print(f"\nAll distances: {stats['distances']}")
            
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")