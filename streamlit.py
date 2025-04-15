import json
import pandas as pd
import regex as re
import ndjson
import streamlit as st
import matplotlib.pyplot as plt

# Load data
drive_path = 'export_All-Project-Cards_2025-02-21_16-50-28.ndjson'
with open(drive_path, 'r') as f:
    data = ndjson.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Replace NA/None with empty strings
df.fillna("", inplace=True)

# Clean and standardize data
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].str.lower().str.strip()

# Remove commas from 'Field of Interest (text)'
if 'Field of Interest (text)' in df.columns:
    df['Field of Interest (text)'] = df['Field of Interest (text)'].str.replace(",", "")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Validate and clean URLs in 'Dynamic Link'
if 'Dynamic Link' in df.columns:
    df['Dynamic Link'] = df['Dynamic Link'].apply(
        lambda x: x if re.match(r'^https?:\/\/\S+$', x) else ""
    )

# Extract city, state, and country from 'Location'
if 'Location' in df.columns:
    def parse_location(location):
        parts = location.split(", ")
        city = parts[0] if len(parts) > 0 else ""
        state = parts[1] if len(parts) > 1 else ""
        country = parts[-1] if len(parts) > 2 else state
        return pd.Series({"City": city, "State": state, "Country": country})

    location_split = df['Location'].apply(parse_location)
    df = pd.concat([df, location_split], axis=1)


df['Field of Interests Lists'] = df['Field of Interest (text)'].str.split('  ')

word_counts = pd.Series([word for sublist in df['Field of Interests Lists'] for word in sublist if word != ""]).value_counts()

top_n = st.slider('Select the number of top fields of interest to display:', min_value=1, max_value=50, value=20)

# Create the bar plot for top N words
fig, ax = plt.subplots(figsize=(12, 6))
word_counts.head(top_n).plot(kind='bar', color='skyblue', ax=ax)
ax.set_xlabel("Field of Interest")
ax.set_ylabel("Number of Project Cards")
ax.set_title("Field of Interests Across Project Cards")
ax.set_xticklabels(word_counts.head(top_n).index, rotation=45, ha='right')


st.pyplot(fig)