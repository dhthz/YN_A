import pandas as pd

# Load processed data
df = pd.read_csv("processed_data.csv")

# Get descriptive statistics
desc = df.describe()

# Save as markdown table
with open('processed_data_description.md', 'w') as f:
    f.write("# Descriptive Statistics\n\n")
    f.write(desc.round(4).to_markdown())

print("Description saved to processed_data_description.md")
