import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset - assuming a vertical data format (1 and 0 values)
# Replace with the actual path to your dataset
data = pd.read_csv('ex.csv')

# Print data to inspect the structure
print("Data Overview:")
print(data.head())

# Make sure all data is in binary format (1 and 0)
# This can be done by converting non-zero values to 1 and leaving zeros as they are
data = data.applymap(lambda x: 1 if x > 0 else 0)

# Apply Apriori Algorithm
# We are assuming the data is in a format where each column represents an item
# and each row represents a transaction (1 means item is purchased, 0 means it is not).

# Minimum support for itemsets, you can adjust this threshold
min_support = 0.2

# Running apriori to find frequent itemsets
frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)

# Print frequent itemsets
print("Frequent Itemsets:")
print(frequent_itemsets)

# Generate association rules from the frequent itemsets
min_threshold = 0.7  # Minimum lift value
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)

# Print association rules
print("Association Rules:")
print(rules)

# Optionally, save the results to a CSV file
frequent_itemsets.to_csv('frequent_itemsets.csv', index=False)
rules.to_csv('association_rules.csv', index=False)
