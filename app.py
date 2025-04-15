import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
import calendar

# Import the cleaning functions
from CleaningData import clean_chats, clean_interests

# Set page config
st.set_page_config(
    page_title="Community Conversation Overview",
    page_icon="📈",
    layout="wide"
)

# Page title and description
st.title("Community Conversation Overview")
st.markdown("### Tracking conversation volume and participation trends over time")

# Function to load data
@st.cache_data
def load_data():
    """Load and cache the cleaned data from local directory"""
    # Filenames
    chats_csv = 'cleaned_chats.csv'
    interests_csv = 'cleaned_interests.csv'

    # Try loading cleaned files
    try:
        chats_df = pd.read_csv(chats_csv)
        interests_df = pd.read_csv(interests_csv)

        # Convert date columns
        chats_df = convert_date_columns(chats_df)
        interests_df = convert_date_columns(interests_df)

        return chats_df, interests_df

    except Exception as e:
        st.error(f"Error loading cleaned data: {e}")

        if not (os.path.exists(chats_csv) and os.path.exists(interests_csv)):
            st.warning("Cleaned files not found — attempting to clean raw data...")

            # Raw file names
            chats_file = 'All Chats Export Jan 14 2025.ndjson'
            interests_file = 'Field of Interests Export Feb 21 2025.ndjson'

            # Run cleaning functions
            chats_df = clean_chats(chats_file)
            interests_df = clean_interests(interests_file)

            # Save cleaned CSVs
            chats_df.to_csv(chats_csv, index=False)
            interests_df.to_csv(interests_csv, index=False)

            return chats_df, interests_df

        return None, None

def convert_date_columns(df):
    """Helper to convert columns containing 'date' or 'msg' to datetime"""
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'msg' in col.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

# Load the data
chats_df, interests_df = load_data()

# Sidebar filters
st.sidebar.header("Filters")

# Time range options
time_ranges = {
    "All Time": None,
    "Last 6 Months": timedelta(days=180),
    "Last Year": timedelta(days=365),
    "Last 2 Years": timedelta(days=730),
    "Custom Range": "custom"
}

selected_range = st.sidebar.selectbox("Time Period", list(time_ranges.keys()))

# Custom date range
if selected_range == "Custom Range":
    today = datetime.now()
    start_date = st.sidebar.date_input("Start Date", 
                                      value=datetime(today.year-1, today.month, 1),
                                      min_value=datetime(2022, 1, 1),
                                      max_value=today)
    end_date = st.sidebar.date_input("End Date", 
                                    value=today,
                                    min_value=start_date,
                                    max_value=today)
else:
    end_date = datetime.now()
    if time_ranges[selected_range]:
        start_date = end_date - time_ranges[selected_range]
    else:
        # All time
        start_date = datetime(2022, 1, 1)  # Arbitrary early date

# Time resolution for grouping
time_resolution = st.sidebar.selectbox(
    "Time Resolution", 
    ["Monthly", "Weekly", "Quarterly", "Yearly"],
    index=0
)

# Community selector (multi-select)
if chats_df is not None and 'Community' in chats_df.columns:
    all_communities = sorted(chats_df['Community'].unique().tolist())
    
    # Add "Top 5 Communities" option
    display_options = ["All Communities", "Top 5 Communities"] + all_communities
    community_display = st.sidebar.selectbox("Display", display_options)
    
    if community_display == "All Communities":
        selected_communities = all_communities
    elif community_display == "Top 5 Communities":
        # Find top 5 communities by conversation count
        top_communities = chats_df['Community'].value_counts().head(5).index.tolist()
        selected_communities = top_communities
    else:
        selected_communities = [community_display]
else:
    selected_communities = []

# Add KPI metrics at the top
if chats_df is not None:
    # Filter data based on selected time period
    filtered_chats = chats_df[
        (chats_df['Creation Date'] >= pd.Timestamp(start_date)) & 
        (chats_df['Creation Date'] <= pd.Timestamp(end_date))
    ]
    
    # Filter by selected communities
    if selected_communities:
        filtered_chats = filtered_chats[filtered_chats['Community'].isin(selected_communities)]
    
    if not filtered_chats.empty:
        # Create metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Total conversations
        total_conversations = len(filtered_chats)
        col1.metric("Total Conversations", f"{total_conversations:,}")
        
        # Communities actively engaging in conversations
        active_communities = filtered_chats['Community'].nunique()
        col2.metric("Communities Conversing", active_communities)
        
        # Unique participants
        unique_participants = pd.concat([
            filtered_chats['Participant1'], 
            filtered_chats['Participant2']
        ]).nunique()
        col3.metric("Unique Participants", f"{unique_participants:,}")
        
        # Avg conversations per community
        avg_per_community = total_conversations / active_communities if active_communities > 0 else 0
        col4.metric("Avg Conversations/Community", f"{avg_per_community:.1f}")
        
        # Community Recency Panel
        st.subheader("Days Since Last Activity for Each Community")

        # Get the last activity date by community
        last_activity = filtered_chats.groupby('Community')['Creation Date'].max().reset_index()
        last_activity['Days_Since_Last_Activity'] = (datetime.now() - last_activity['Creation Date']).dt.days

        # Sort by recency (smallest number of days = most recent activity)
        last_activity_sorted = last_activity.sort_values('Days_Since_Last_Activity')

        # Display color-coded blocks for the two communities
        col1, col2 = st.columns(2)

        with col1:
            community1 = last_activity_sorted.iloc[0]
            st.markdown(f"#### {community1['Community']}")
            st.markdown(
                f"<div style='background-color: {'#2d6a4f' if community1['Days_Since_Last_Activity'] < 30 else '#e63946'}; color: white; padding: 10px; border-radius: 5px;'>"
                f"Last Activity: {community1['Days_Since_Last_Activity']} days</div>", 
                unsafe_allow_html=True
            )

        with col2:
            community2 = last_activity_sorted.iloc[1]
            st.markdown(f"#### {community2['Community']}")
            st.markdown(
                f"<div style='background-color: {'#2d6a4f' if community2['Days_Since_Last_Activity'] < 30 else '#e63946'}; color: white; padding: 10px; border-radius: 5px;'>"
                f"Last Activity: {community2['Days_Since_Last_Activity']} days</div>", 
                unsafe_allow_html=True
            )
        st.markdown("<br>", unsafe_allow_html=True)

        # Prepare data for timeline
        # Apply time resolution
        if time_resolution == "Monthly":
            filtered_chats['Period'] = filtered_chats['Creation Date'].dt.to_period('M')
        elif time_resolution == "Weekly":
            filtered_chats['Period'] = filtered_chats['Creation Date'].dt.to_period('W')
        elif time_resolution == "Quarterly":
            filtered_chats['Period'] = filtered_chats['Creation Date'].dt.to_period('Q')
        else:  # Yearly
            filtered_chats['Period'] = filtered_chats['Creation Date'].dt.to_period('Y')
        
        # Convert period to string for plotting
        filtered_chats['Period_Str'] = filtered_chats['Period'].astype(str)
        
        # Create timeline for communities
        community_counts = filtered_chats.groupby(['Period_Str', 'Community']).size().reset_index(name='Conversation_Count')
        community_pivot = community_counts.pivot(index='Period_Str', columns='Community', values='Conversation_Count').fillna(0)
        
        # Reindex to ensure all time periods are present (handling gaps)
        all_periods = filtered_chats['Period_Str'].unique()
        community_pivot = community_pivot.reindex(all_periods)
        
        # Sort by time period
        community_pivot = community_pivot.sort_index()
        
        # Main timeline chart
        st.subheader("Conversation Volume Over Time")
        
        # Determine suitable height based on number of communities
        chart_height = max(500, 100 + (len(selected_communities) * 50))
        
        fig = px.line(
            community_pivot, 
            markers=True,
            labels={"value": "Conversation Count", "variable": "Community"},
            title=f"Community Engagement Timeline ({time_resolution} View)",
            height=chart_height
        )
        
        fig.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Number of Conversations",
            legend_title="Community",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Growth analysis
        st.subheader("Community Growth Analysis")
        
        # Calculate growth rates
        growth_col1, growth_col2 = st.columns(2)
        
        with growth_col1:
            st.markdown("#### Period-over-Period Growth")
            
            # Calculate total conversations by period across all selected communities
            total_by_period = community_pivot.sum(axis=1).reset_index()
            total_by_period.columns = ['Period', 'Total']
            
            # Calculate growth rate
            total_by_period['Growth_Rate'] = total_by_period['Total'].pct_change() * 100
            
            # Create bar chart of growth rates
            fig_growth = px.bar(
                total_by_period.dropna(), 
                x='Period', 
                y='Growth_Rate',
                title=f"{time_resolution} Growth Rate (%)",
                color='Growth_Rate',
                color_continuous_scale=['red', 'lightgrey', 'green'],
                range_color=[-50, 50]  # Adjust color scale range
            )
            
            fig_growth.update_layout(
                xaxis_title="Time Period",
                yaxis_title="Growth Rate (%)",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig_growth, use_container_width=True)
            
        with growth_col2:
            st.markdown("#### Community Activity Comparison")
            
            # Compare first and last period activity for each community
            if len(community_pivot) >= 2:
                first_period = community_pivot.iloc[0]
                last_period = community_pivot.iloc[-1]
                
                comparison_data = pd.DataFrame({
                    'Community': community_pivot.columns,
                    'First_Period': first_period.values,
                    'Last_Period': last_period.values
                })
                
                # Calculate change
                comparison_data['Change'] = comparison_data['Last_Period'] - comparison_data['First_Period']
                comparison_data['Percent_Change'] = (comparison_data['Change'] / comparison_data['First_Period'] * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Sort by change
                comparison_data = comparison_data.sort_values('Change', ascending=False)
                
                # Create comparison chart
                fig_comparison = px.bar(
                    comparison_data,
                    x='Community',
                    y='Change',
                    title=f"Community Growth from {community_pivot.index[0]} to {community_pivot.index[-1]}",
                    color='Percent_Change',
                    color_continuous_scale=['red', 'lightgrey', 'green'],
                    text='Percent_Change'
                )
                
                fig_comparison.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside'
                )
                
                fig_comparison.update_layout(
                    xaxis_title="Community",
                    yaxis_title="Change in Conversation Count",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
            else:
                st.info("Need at least two time periods to show growth comparison.")

# Footer
st.markdown("---")
st.markdown("*Data updated as of: {}*".format(datetime.now().strftime("%b %d, %Y")))

# About Activity, Monthly Growth Rate, and Community Growth
st.markdown("---")
st.markdown("""
    **Activity & Growth Overview**

    - **Activity** refers to the last recorded conversation in each community. "Days since last activity" measures the time elapsed since that conversation, helping track community engagement over time.

    - **Monthly Growth Rate** shows the percentage change in total conversations from one month to the next, highlighting trends in community interaction.

    - **Community Growth** compares the first and last recorded periods to show how conversation volume has changed, indicating the overall growth or decline in community engagement.
""")