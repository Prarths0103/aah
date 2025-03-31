import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
file_path = r"/mnt/c/Users/gowri/Downloads/groceries - groceries.csv"  # Change the path if needed
df = pd.read_csv(file_path)

# Preview dataset
print("Dataset Preview:\n", df.head())

# Convert transactions into lists (ignoring NaN values)
transactions = df.iloc[:, 1:].apply(lambda row: row.dropna().tolist(), axis=1)

# Convert transactions into a DataFrame with one-hot encoding
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
print("\nFrequent Itemsets:\n", frequent_itemsets)

# Generate Association Rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print("\nAssociation Rules:\n", rules)
