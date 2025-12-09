"""
Evaluation Script for Speed Bumps Image Generation Project

This script provides comprehensive evaluation tools for generated images.
Can be run as a Python script or converted to a Jupyter notebook.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import glob
from collections import defaultdict
import numpy as np
from PIL import Image

# Configuration
RESULTS_DIR = Path('results/baseline')
LOGS_DIR = Path('logs')
EVALUATION_DIR = Path('evaluation_results')
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

# Failure categories
FAILURE_CATEGORIES = {
    'missing_speed_bump': 'Speed bump is missing or not visible',
    'incorrect_shape': 'Speed bump has incorrect shape or proportions',
    'poor_integration': 'Speed bump poorly integrated with road surface',
    'wrong_perspective': 'Incorrect perspective or angle',
    'artifacts': 'Visual artifacts, distortions, or errors',
    'unrealistic': 'Looks unrealistic or unnatural',
    'wrong_context': 'Wrong setting or context',
    'multiple_issues': 'Multiple problems present',
    'success': 'Successful generation - speed bump is correct'
}

def load_generation_results():
    """Load latest generation log."""
    log_files = sorted(glob.glob(str(LOGS_DIR / 'generation_log_*.json')))
    if not log_files:
        print("❌ No generation logs found.")
        return None, [], []
    
    latest_log = log_files[-1]
    with open(latest_log, 'r') as f:
        generation_data = json.load(f)
    
    successful_generations = [g for g in generation_data['generations'] if g.get('success', False)]
    failed_generations = [g for g in generation_data['generations'] if not g.get('success', True)]
    
    return generation_data, successful_generations, failed_generations

def assess_image_quality(image_path: Path) -> Dict:
    """Assess basic quality metrics for an image."""
    img = Image.open(image_path)
    img_array = np.array(img)
    
    metrics = {
        'width': img.width,
        'height': img.height,
        'format': img.format,
        'mode': img.mode,
        'file_size_kb': image_path.stat().st_size / 1024,
    }
    
    if img_array.size > 0:
        metrics['mean_brightness'] = float(np.mean(img_array))
        metrics['std_brightness'] = float(np.std(img_array))
    
    return metrics

def assess_all_images(generations: List[Dict]) -> List[Dict]:
    """Assess quality of all generated images."""
    assessments = []
    for gen in generations:
        filename = gen.get('filename')
        if not filename:
            continue
        image_path = RESULTS_DIR / filename
        if not image_path.exists():
            continue
        try:
            metrics = assess_image_quality(image_path)
            metrics['filename'] = filename
            metrics['prompt_id'] = gen.get('prompt_id', 'N/A')
            assessments.append(metrics)
        except Exception as e:
            print(f"Error assessing {filename}: {e}")
    return assessments

def generate_evaluation_report(generation_data, quality_assessments):
    """Generate comprehensive evaluation report."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = EVALUATION_DIR / f"evaluation_report_{timestamp}.json"
    
    report = {
        'timestamp': timestamp,
        'generation_summary': {
            'total': generation_data.get('total_prompts', 0),
            'successful': generation_data.get('successful', 0),
            'failed': generation_data.get('failed', 0),
            'success_rate': generation_data.get('success_rate', 0)
        },
        'quality_assessments': quality_assessments,
        'statistics': generation_data.get('statistics', {})
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Evaluation report saved to: {report_path}")
    return report

if __name__ == "__main__":
    print("Loading generation results...")
    generation_data, successful_generations, failed_generations = load_generation_results()
    
    if not generation_data:
        exit(1)
    
    print(f"✅ Loaded {len(successful_generations)} successful generations")
    
    print("Running quality assessment...")
    quality_assessments = assess_all_images(successful_generations)
    print(f"✅ Assessed {len(quality_assessments)} images")
    
    print("Generating evaluation report...")
    report = generate_evaluation_report(generation_data, quality_assessments)
    
    print("\nEvaluation complete!")

