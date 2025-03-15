import pandas as pd
import os 
from tabulate import tabulate

file = "alzheimers_disease_data.csv"

if not os.path.exists(file):
    print(f"File {file} not found")
    exit()
else:
    try:
        df = pd.read_csv(file, encoding="utf-8")
        print("File loaded successfully")
        #print(df.head())
        #print(df.info())
        #summary = df.describe()
        
        # Apply One-Hot Encoding
        df["Ethnicity"] = df["Ethnicity"].astype("category")
        df["EducationLevel"] = df["EducationLevel"].astype("category")
        df = pd.get_dummies(df, columns=["Ethnicity"], prefix="Ethnicity",dtype=int)
        df = pd.get_dummies(df, columns=["EducationLevel"], prefix="EducationLevel",dtype=int)
        df.to_csv("onehotEncoded_data.csv", index=False)
    
    
    except FileNotFoundError:
        print(f"❌ Error: File '{file}' not found.")
    except PermissionError:
        print(f"❌ Error: Not permitted to use '{file_path}'.")
    except UnicodeDecodeError:
        print(f"❌ Error: Encoding error, please try again!")
    except Exception as e:
        print(f"❌ An error occured: {e}")