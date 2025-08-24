import numpy as np

def process_tapping_data(tapping_timestamps):
    """Process tapping data to extract features"""
    if len(tapping_timestamps) < 2:
        return {
            'mean_interval': 0,
            'std_interval': 0,
            'risk_contribution': 0
        }
    
    # Calculate inter-tap intervals
    intervals = np.diff(tapping_timestamps)
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    
    # Simple risk contribution based on variability
    # Higher variability suggests more risk
    risk_contribution = min(50, std_interval * 100)
    
    return {
        'mean_interval': mean_interval,
        'std_interval': std_interval,
        'risk_contribution': risk_contribution
    }

def load_symptom_checklist():
    """Return the symptom checklist questions"""
    symptoms = {
        "tremor": "Do you experience tremors or shaking in your hands, arms, legs, or jaw?",
        "rigidity": "Do you feel muscle stiffness or resistance to movement?",
        "bradykinesia": "Do you have slowness of movement or difficulty initiating movement?",
        "postural": "Do you have trouble with balance or experience falls?",
        "gait": "Do you have changes in your walking pattern, like shuffling steps or freezing?",
        "micrographia": "Has your handwriting become smaller or more crowded?",
        "speech": "Has your speech become softer, monotone, or slurred?",
        "facial": "Have you noticed reduced facial expression (masked face)?",
        "sleep": "Do you experience trouble sleeping or excessive daytime sleepiness?",
        "memory": "Do you have problems with memory or thinking clearly?"
    }
    return symptoms