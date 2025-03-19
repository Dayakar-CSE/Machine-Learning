import pandas as pd

# Define the dataset
data = {
    'income': ['medium', 'high', 'low', 'medium', 'high', 'low', 'medium', 'medium', 'high', 'low'],
    'recreation': ['skiing', 'golf', 'speedway', 'football', 'flying', 'football', 'golf', 'golf', 'skiing', 'golf'],
    'job': ['design', 'trading', 'transport', 'banking', 'media', 'security', 'media', 'transport', 'banking', 'unemployed'],
    'status': ['single', 'married', 'married', 'single', 'married', 'single', 'single', 'married', 'single', 'married'],
    'age_group': ['twenties', 'forties', 'thirties', 'thirties', 'fifties', 'twenties', 'thirties', 'forties', 'thirties', 'forties'],
    'home_owner': ['no', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes'],
    'risk': ['high Risk', 'low Risk', 'med Risk', 'low Risk', 'high Risk', 'med Risk', 'med Risk', 'low Risk', 'high Risk', 'high Risk']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate the unconditional probability of 'golf'
total_entries = len(df)
golf_count = len(df[df['recreation'] == 'golf'])
unconditional_probability_golf = golf_count / total_entries

# Calculate the conditional probability of 'single' given 'med Risk'
med_risk_count = len(df[df['risk'] == 'med Risk'])
single_and_med_risk_count = len(df[(df['risk'] == 'med Risk') & (df['status'] == 'single')])

# Using a ternary operator for conditional probability
conditional_probability_single_given_med_risk = (single_and_med_risk_count / med_risk_count) if med_risk_count > 0 else 0

# Output
print(f"Unconditional Probability of 'golf': {unconditional_probability_golf:.2f}")
print(f"Conditional Probability of 'single' given 'med Risk': {conditional_probability_single_given_med_risk:.2f}")
