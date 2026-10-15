import pandas as pd

# Load the rules discovered in Step 4
# Note: You can either import 'rules' from mining_engine.py or save them to CSV first
from mining_engine import rules 

def evaluate_rules(all_rules):
    # Filter 1: Pattern must appear in at least 2% of data (lowered for weak supply chain data)
    # Filter 2: Rule accuracy (confidence) must be at least 20% (realistic threshold)
    # Filter 3: Relationship strength (lift) must be greater than 0.8 (any correlation)
    
    refined_rules = all_rules[
        (all_rules['support'] >= 0.02) & 
        (all_rules['confidence'] >= 0.20) & 
        (all_rules['lift'] > 0.8)
    ]
    
    return refined_rules

# Run the evaluation framework
final_rules = evaluate_rules(rules)

# Save the high-quality rules for the Dashboard
final_rules.to_csv('final_delivery_insights.csv', index=False)

print(f"Evaluation Complete! {len(final_rules)} high-quality patterns saved for Dashboard.")