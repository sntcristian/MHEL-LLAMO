import json
import numpy as np
from typing import List, Dict, Tuple, Optional

def find_optimal_threshold(data: List[Dict], 
                          threshold_range: Tuple[float, float] = (0.0, 30.0),
                          num_thresholds: int = 100) -> Dict:
    """
    Find the optimal score threshold to maximize F1-score for entity linking.
    
    Args:
        data: List of entity annotations with candidates
        threshold_range: Tuple of (min_threshold, max_threshold) to search
        num_thresholds: Number of threshold values to test
        
    Returns:
        Dictionary with optimal threshold, best F1-score, and detailed results
    """
    
    def calculate_f1_at_threshold(threshold: float) -> Tuple[float, int, int, int]:
        """Calculate F1, precision, recall at given threshold."""
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for annotation in data:
            true_identifier = annotation.get('identifier')
            candidates = annotation.get('candidates', [])
            
            # Filter candidates above threshold
            valid_candidates = [c for c in candidates if c.get('score', 0) >= threshold]
            
            if not valid_candidates:
                # No predictions above threshold
                if true_identifier:  # There was a true answer
                    false_negatives += 1
                continue
            
            # Take the highest scoring candidate above threshold
            best_candidate = max(valid_candidates, key=lambda x: x.get('score', 0))
            predicted_id = best_candidate.get('wb_id')
            
            if predicted_id == true_identifier:
                true_positives += 1
            else:
                false_positives += 1
                if true_identifier:  # There was a true answer we missed
                    false_negatives += 1
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1, true_positives, false_positives, false_negatives
    
    # Generate threshold values to test
    min_thresh, max_thresh = threshold_range
    thresholds = np.linspace(min_thresh, max_thresh, num_thresholds)
    
    best_f1 = 0
    best_threshold = 0
    best_metrics = None
    results = []
    
    print(f"Testing {num_thresholds} thresholds from {min_thresh} to {max_thresh}...")
    
    for threshold in thresholds:
        f1, tp, fp, fn = calculate_f1_at_threshold(threshold)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn
            }
    
    return {
        'optimal_threshold': best_threshold,
        'best_f1_score': best_f1,
        'best_metrics': best_metrics,
        'all_results': results
    }

def analyze_dataset_statistics(data: List[Dict]) -> Dict:
    """
    Analyze the dataset to understand score distributions and baseline performance.
    """
    stats = {
        'total_annotations': len(data),
        'annotations_with_candidates': 0,
        'annotations_with_correct_candidate': 0,
        'score_statistics': {},
        'baseline_performance': {}
    }
    
    all_scores = []
    correct_scores = []
    
    for annotation in data:
        true_identifier = annotation.get('identifier')
        candidates = annotation.get('candidates', [])
        
        if candidates:
            stats['annotations_with_candidates'] += 1
            
            # Collect all scores
            scores = [c.get('score', 0) for c in candidates]
            all_scores.extend(scores)
            
            # Check if correct answer exists in candidates
            correct_candidate = next((c for c in candidates if c.get('wb_id') == true_identifier), None)
            if correct_candidate:
                stats['annotations_with_correct_candidate'] += 1
                correct_scores.append(correct_candidate.get('score', 0))
    
    # Score statistics
    if all_scores:
        stats['score_statistics'] = {
            'min_score': min(all_scores),
            'max_score': max(all_scores),
            'mean_score': np.mean(all_scores),
            'median_score': np.median(all_scores),
            'std_score': np.std(all_scores)
        }
    
    if correct_scores:
        stats['correct_answer_scores'] = {
            'min_score': min(correct_scores),
            'max_score': max(correct_scores),
            'mean_score': np.mean(correct_scores),
            'median_score': np.median(correct_scores)
        }
    
    # Baseline performance (threshold = 0, take highest scoring candidate)
    baseline_result = find_optimal_threshold(data, threshold_range=(0, 0), num_thresholds=1)
    stats['baseline_performance'] = baseline_result['best_metrics']
    
    return stats

# Example usage
if __name__ == "__main__":
    
    data_path = "results/DZ_IT/candidates_dev_top10.json"
    with open(data_path, "r", encoding="utf-8") as f:
        sample_data = json.load(f)


    # Analyze dataset
    print("Dataset Statistics:")
    stats = analyze_dataset_statistics(sample_data)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*50 + "\n")
    
    # Find optimal threshold
    result = find_optimal_threshold(sample_data)
    
    print(f"Optimal threshold: {result['optimal_threshold']:.4f}")
    print(f"Best F1-score: {result['best_f1_score']:.4f}")
    print(f"Best metrics: {result['best_metrics']}")
    
    # Show some threshold results around the optimal
    optimal_idx = None
    for i, res in enumerate(result['all_results']):
        if abs(res['threshold'] - result['optimal_threshold']) < 0.001:
            optimal_idx = i
            break
    
    if optimal_idx:
        print(f"\nThreshold analysis around optimal value:")
        start_idx = max(0, optimal_idx - 5)
        end_idx = min(len(result['all_results']), optimal_idx + 6)
        
        for res in result['all_results'][start_idx:end_idx]:
            marker = " <-- OPTIMAL" if abs(res['threshold'] - result['optimal_threshold']) < 0.001 else ""
            print(f"Threshold: {res['threshold']:6.2f}, F1: {res['f1']:.4f}, "
                  f"P: {res['precision']:.4f}, R: {res['recall']:.4f}{marker}")