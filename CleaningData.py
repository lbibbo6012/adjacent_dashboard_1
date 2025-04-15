# Install dependencies if needed
# pip install ndjson pandas streamlit
import ndjson
import pandas as pd
import re
import os

# ---------------------------
# 1. Clean All Chats Data
# ---------------------------
def clean_chats(file_path):
    print(f"🔄 Step 1: Processing '{os.path.basename(file_path)}'")
    
    with open(file_path, 'r') as f:
        df_chats = pd.DataFrame(ndjson.load(f))
    
    # Print column names to understand the data structure
    print(f"Available columns in chats file: {df_chats.columns.tolist()}")
    
    # Clean chat data - only drop columns that exist
    columns_to_drop = ['Modified Date', 'has unread messages', 'Chat Title']
    existing_columns = [col for col in columns_to_drop if col in df_chats.columns]
    df_chats_clean = df_chats.drop(columns=existing_columns)
    
    # Convert dates with explicit format
    date_format = "%b %d, %Y %I:%M %p"
    if 'Creation Date' in df_chats_clean.columns:
        df_chats_clean['Creation Date'] = pd.to_datetime(
            df_chats_clean['Creation Date'],
            format=date_format,
            errors="coerce"
        )
    
    if 'Last sent msg' in df_chats_clean.columns:
        df_chats_clean['Last sent msg'] = pd.to_datetime(
            df_chats_clean['Last sent msg'],
            format=date_format,
            errors="coerce"
        )
    
    # Split conversation participants
    if 'conversationParts' in df_chats_clean.columns:
        df_chats_clean[['Participant1', 'Participant2']] = (
            df_chats_clean['conversationParts']
            .str.split(' , ', expand=True)
            .iloc[:, :2]
        )
    
    # Standardize community names
    if 'Community' in df_chats_clean.columns:
        df_chats_clean['Community'] = (
            df_chats_clean['Community']
            .str.strip()
            .str.title()
        )
    
    return df_chats_clean

# ---------------------------
# 2. Clean Field of Interests (UPDATED)
# ---------------------------
def clean_interests(file_path):
    print(f"\n🔄 Step 2: Processing '{os.path.basename(file_path)}'")
    
    with open(file_path, 'r') as f:
        df_interests = pd.DataFrame(ndjson.load(f))
    
    # Print column names to understand the data structure
    print(f"Available columns in interests file: {df_interests.columns.tolist()}")
    
    # Clean interest data - only drop columns that exist
    columns_to_drop = ['Slug', 'Creator', 'Community']
    existing_columns = [col for col in columns_to_drop if col in df_interests.columns]
    df_interests_clean = df_interests.drop(columns=existing_columns)
    
    # Convert priority to boolean and fill missing values with False
    if 'Priority' in df_interests_clean.columns:
        df_interests_clean['Priority'] = (
            df_interests_clean['Priority']
            .map({'yes': True, 'no': False})
            .fillna(False)  # Fill missing values with False
        )
    
    # Clean titles
    if 'Title' in df_interests_clean.columns:
        df_interests_clean['Title'] = (
            df_interests_clean['Title']
            .str.title()
            .apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x) if isinstance(x, str) else x)
        )
    
    # Convert dates with explicit format
    date_format = "%b %d, %Y %I:%M %p"
    if 'Creation Date' in df_interests_clean.columns:
        df_interests_clean['Creation Date'] = pd.to_datetime(
            df_interests_clean['Creation Date'],
            format=date_format,
            errors="coerce"
        )
    
    if 'Modified Date' in df_interests_clean.columns:
        df_interests_clean['Modified Date'] = pd.to_datetime(
            df_interests_clean['Modified Date'],
            format=date_format,
            errors="coerce"
        )
    
    return df_interests_clean

# ---------------------------
# Function to save data previews
# ---------------------------
def save_data_previews(cleaned_chats, cleaned_interests, folder_path):
    """Save comprehensive previews of the cleaned data for review"""
    
    # Create a preview folder if it doesn't exist
    preview_folder = os.path.join(folder_path, 'data_previews')
    os.makedirs(preview_folder, exist_ok=True)
    
    # Save a detailed preview of chat data
    with open(os.path.join(preview_folder, 'chats_preview.txt'), 'w') as f:
        f.write("=== CHAT DATA PREVIEW ===\n\n")
        f.write(f"Shape: {cleaned_chats.shape}\n\n")
        f.write(f"Columns: {cleaned_chats.columns.tolist()}\n\n")
        f.write("Data Types:\n")
        f.write(str(cleaned_chats.dtypes) + "\n\n")
        f.write("First 5 rows:\n")
        f.write(str(cleaned_chats.head(5)) + "\n\n")
        f.write("Sample values from each column:\n")
        for col in cleaned_chats.columns:
            f.write(f"\n{col}:\n")
            try:
                unique_vals = cleaned_chats[col].dropna().unique()[:5]
                f.write(str(unique_vals) + "\n")
            except:
                f.write("Could not get unique values\n")
    
    # Save a detailed preview of interests data
    with open(os.path.join(preview_folder, 'interests_preview.txt'), 'w') as f:
        f.write("=== INTERESTS DATA PREVIEW ===\n\n")
        f.write(f"Shape: {cleaned_interests.shape}\n\n")
        f.write(f"Columns: {cleaned_interests.columns.tolist()}\n\n")
        f.write("Data Types:\n")
        f.write(str(cleaned_interests.dtypes) + "\n\n")
        f.write("First 5 rows:\n")
        f.write(str(cleaned_interests.head(5)) + "\n\n")
        f.write("Sample values from each column:\n")
        for col in cleaned_interests.columns:
            f.write(f"\n{col}:\n")
            try:
                unique_vals = cleaned_interests[col].dropna().unique()[:5]
                f.write(str(unique_vals) + "\n")
            except:
                f.write("Could not get unique values\n")
    
    # Also save as HTML for easier viewing
    cleaned_chats.head(10).to_html(os.path.join(preview_folder, 'chats_preview.html'))
    cleaned_interests.head(10).to_html(os.path.join(preview_folder, 'interests_preview.html'))
    
    print(f"\n✅ Data previews saved to: {preview_folder}")
    print("Preview files created:")
    print("1. chats_preview.txt and chats_preview.html")
    print("2. interests_preview.txt and interests_preview.html")

# ---------------------------
# 3. Main Function for Cleaning Pipeline
# ---------------------------
def main():
    # Set the folder path
    folder_path = os.path.expanduser('~/Desktop/485 capstone')
    
    # Set file paths
    chats_file = os.path.join(folder_path, 'All Chats Export Jan 14 2025.ndjson')
    interests_file = os.path.join(folder_path, 'Field of Interests Export Feb 21 2025.ndjson')
    
    try:
        # Clean both datasets
        cleaned_chats = clean_chats(chats_file)
        cleaned_interests = clean_interests(interests_file)
        
        # Show results
        print("\n✅ Cleaning Complete!")
        print("\nChat Data Preview:")
        print(cleaned_chats.head(2))
        print("\nInterests Data Preview:")
        print(cleaned_interests.head(2))
        
        # Save detailed data previews
        save_data_previews(cleaned_chats, cleaned_interests, folder_path)
        
        # Export cleaned data to the same folder
        cleaned_chats.to_csv(os.path.join(folder_path, 'cleaned_chats.csv'), index=False)
        cleaned_interests.to_csv(os.path.join(folder_path, 'cleaned_interests.csv'), index=False)
        
        print(f"\n✅ Files saved to: {folder_path}")
        print("1. cleaned_chats.csv")
        print("2. cleaned_interests.csv")
        
        # Return dataframes for use with Streamlit
        return cleaned_chats, cleaned_interests
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Check: 1) Correct files exist in the folder 2) Folder path is correct 3) No special characters in filenames")
        return None, None

# ---------------------------
# Run the script directly
# ---------------------------
if __name__ == "__main__":
    main()