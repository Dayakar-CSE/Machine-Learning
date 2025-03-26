import pandas as pd

# Define the dataset as a list of dictionaries
data = [
    {"income": "medium", "recreation": "skiing", "job": "design", "status": "single", "age_group": "twenties", "home_owner": "no", "risk": "highRisk"},
    {"income": "high", "recreation": "golf", "job": "trading", "status": "married", "age_group": "forties", "home_owner": "yes", "risk": "lowRisk"},
    {"income": "low", "recreation": "speedway", "job": "transport", "status": "married", "age_group": "thirties", "home_owner": "yes", "risk": "medRisk"},
    {"income": "medium", "recreation": "football", "job": "banking", "status": "single", "age_group": "thirties", "home_owner": "yes", "risk": "lowRisk"},
    {"income": "high", "recreation": "flying", "job": "media", "status": "married", "age_group": "fifties", "home_owner": "yes", "risk": "highRisk"},
    {"income": "low", "recreation": "football", "job": "security", "status": "single", "age_group": "twenties", "home_owner": "no", "risk": "medRisk"},
    {"income": "medium", "recreation": "golf", "job": "media", "status": "single", "age_group": "thirties", "home_owner": "yes", "risk": "medRisk"},
    {"income": "medium", "recreation": "golf", "job": "transport", "status": "married", "age_group": "forties", "home_owner": "yes", "risk": "lowRisk"},
    {"income": "high", "recreation": "skiing", "job": "banking", "status": "single", "age_group": "thirties", "home_owner": "yes", "risk": "highRisk"},
    {"income": "low", "recreation": "golf", "job": "unemployed", "status": "married", "age_group": "forties", "home_owner": "yes", "risk": "highRisk"},
]

# Convert the list into a Pandas DataFrame
df = pd.DataFrame(data)

# Calculate the unconditional probability of "golf"
total_entries = len(df)
golf_count = len(df[df["recreation"] == "golf"])
prob_golf = golf_count / total_entries

# Calculate the conditional probability of "single" given "medRisk"
med_risk_df = df[df["risk"] == "medRisk"]
single_given_medrisk = len(med_risk_df[med_risk_df["status"] == "single"]) / len(med_risk_df)

# Print results
print(f"Unconditional Probability of 'golf': {prob_golf:.2f}")
print(f"Conditional Probability of 'single' given 'medRisk': {single_given_medrisk:.2f}")
