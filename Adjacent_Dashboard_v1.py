import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime

# Set page config for a better looking dashboard
st.set_page_config(
    page_title="Adjacent App Content Overview",
    page_icon="📊",
    layout="wide"
)

# Cache data loading to improve performance
@st.cache_data
def load_data():
    """Load and prepare the CSV data files"""
    # Load CSV files
    projects_df = pd.read_csv('Cleaned_Project_Cards.csv')
    skills_df = pd.read_csv('Cleaned_Skills_Dataset.csv')
    search_df = pd.read_csv('Cleaned_Search_History.csv')
    
    # Process dates for skills and searches
    skills_df['Creation Date'] = pd.to_datetime(skills_df['Creation Date'])
    search_df['Creation Date'] = pd.to_datetime(search_df['Creation Date'])
    
    return projects_df, skills_df, search_df

# Process fields of interest
@st.cache_data
def process_fields(projects_df):
    """Extract and count fields of interest from project descriptions"""
    field_counts = {}
    
    for field_text in projects_df['Field of Interest (text)'].dropna():
        # Split by various delimiters (multiple spaces, commas)
        fields = re.split(r'(?:\s{2,}|,\s*)', field_text)
        for field in fields:
            field = field.strip().lower()
            if field:
                field_counts[field] = field_counts.get(field, 0) + 1
    
    # Convert to DataFrame for plotting
    field_df = pd.DataFrame({
        'field': list(field_counts.keys()),
        'count': list(field_counts.values())
    }).sort_values('count', ascending=False)
    
    return field_df

# Process timeline data
@st.cache_data
def process_timeline(skills_df, search_df):
    """Create a timeline of skills and searches by month"""
    # Group skills by month
    skills_by_month = skills_df.groupby(pd.Grouper(key='Creation Date', freq='M')).size()
    skills_by_month = skills_by_month.reset_index()
    skills_by_month.columns = ['Creation Date', 'skills']
    skills_by_month['month'] = skills_by_month['Creation Date'].dt.strftime('%Y-%m')
    
    # Group searches by month
    search_by_month = search_df.groupby(pd.Grouper(key='Creation Date', freq='M')).size()
    search_by_month = search_by_month.reset_index()
    search_by_month.columns = ['Creation Date', 'searches']
    search_by_month['month'] = search_by_month['Creation Date'].dt.strftime('%Y-%m')
    
    # Merge the datasets
    timeline_df = pd.merge(skills_by_month[['month', 'skills']], 
                          search_by_month[['month', 'searches']], 
                          on='month', how='outer').fillna(0)
    
    # Sort chronologically
    timeline_df = timeline_df.sort_values('month')
    
    # Add a more readable month label
    timeline_df['monthLabel'] = timeline_df['month'].apply(
        lambda x: f"{x.split('-')[1]}/{x.split('-')[0][2:]}"
    )
    
    return timeline_df

# Process community engagement
@st.cache_data
def process_community_engagement(projects_df, skills_df, search_df):
    """Calculate engagement metrics across communities"""
    # Projects by community
    projects_by_community = projects_df['Community List'].value_counts().reset_index()
    projects_by_community.columns = ['community', 'projects']
    
    # Skills by community
    skills_by_community = skills_df['Community'].value_counts().reset_index()
    skills_by_community.columns = ['community', 'skills']
    
    # Searches by community
    searches_by_community = search_df['Community'].value_counts().reset_index()
    searches_by_community.columns = ['community', 'searches']
    
    # Merge all datasets
    engagement_df = pd.merge(projects_by_community, skills_by_community, 
                             on='community', how='outer').fillna(0)
    
    engagement_df = pd.merge(engagement_df, searches_by_community, 
                             on='community', how='outer').fillna(0)
    
    # Calculate total engagement
    engagement_df['total'] = engagement_df['projects'] + engagement_df['skills'] + engagement_df['searches']
    
    # Sort by total engagement
    engagement_df = engagement_df.sort_values('total', ascending=False)
    
    return engagement_df

# Process search terms
@st.cache_data
def process_search_terms(search_df):
    """Extract and count search terms from search queries"""
    search_terms = {}
    
    for text in search_df['Text'].dropna():
        # Simple tokenization
        terms = re.sub(r'[^\w\s]', '', text.lower()).split()
        for term in terms:
            if len(term) > 2:  # Ignore very short terms
                search_terms[term] = search_terms.get(term, 0) + 1
    
    # Convert to DataFrame for plotting
    search_terms_df = pd.DataFrame({
        'term': list(search_terms.keys()),
        'count': list(search_terms.values())
    }).sort_values('count', ascending=False)
    
    return search_terms_df

# Process skills distribution by priority
@st.cache_data
def process_skills_distribution(skills_df):
    """Analyze skills distribution by community and priority level"""
    # Create priority column mapping
    priority_mapping = {'Yes': 'priority', 'No': 'regular'}
    
    # Group by community and priority
    skills_pivot = pd.pivot_table(
        skills_df,
        index='Community',
        columns='Priority',
        aggfunc='size',
        fill_value=0
    ).reset_index()
    
    # Rename columns
    skills_pivot.columns.name = None
    if 'Yes' not in skills_pivot.columns:
        skills_pivot['Yes'] = 0
    if 'No' not in skills_pivot.columns:
        skills_pivot['No'] = 0
    
    skills_pivot = skills_pivot.rename(columns={'Yes': 'priority', 'No': 'regular'})
    skills_pivot = skills_pivot.rename(columns={'Community': 'community'})
    
    # Calculate totals
    skills_pivot['total'] = skills_pivot['priority'] + skills_pivot['regular']
    
    # Sort by total
    skills_pivot = skills_pivot.sort_values('total', ascending=False)
    
    return skills_pivot

# Load the data
projects_df, skills_df, search_df = load_data()

# Process data for visualizations
field_df = process_fields(projects_df)
timeline_df = process_timeline(skills_df, search_df)
engagement_df = process_community_engagement(projects_df, skills_df, search_df)
search_terms_df = process_search_terms(search_df)
skills_dist_df = process_skills_distribution(skills_df)

# Dashboard Header
st.title("Adjacent App Content Overview")
st.markdown(f"Analyzing data from {len(projects_df)} projects, {len(skills_df)} skills, and {len(search_df)} searches")

# KPI Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Projects", value=len(projects_df))
with col2:
    st.metric(label="Skills Registered", value=len(skills_df))
with col3:
    st.metric(label="Search Queries", value=len(search_df))
with col4:
    st.metric(label="Active Communities", value=len(engagement_df))

# Create tabs for different dashboard sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Projects", "Skills", "Searches", "Community", "Needs"])

# OVERVIEW TAB
with tab1:
    st.header("Platform Overview")
    
    # Platform Activity Timeline
    st.subheader("Platform Activity Timeline")
    fig_timeline = px.line(
        timeline_df, 
        x='monthLabel', 
        y=['skills', 'searches'],
        labels={'variable': 'Activity Type', 'value': 'Count', 'monthLabel': 'Month'},
        title='Activity Over Time',
        color_discrete_map={'skills': '#0088FE', 'searches': '#00C49F'}
    )
    fig_timeline.update_layout(height=400)
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Two column layout for additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Community Engagement
        st.subheader("Community Engagement")
        fig_community = px.bar(
            engagement_df,
            y='community',
            x=['projects', 'skills', 'searches'],
            orientation='h',
            barmode='stack',
            labels={'value': 'Count', 'community': 'Community'},
            color_discrete_map={
                'projects': '#0088FE',
                'skills': '#00C49F',
                'searches': '#FFBB28'
            }
        )
        fig_community.update_layout(height=400)
        st.plotly_chart(fig_community, use_container_width=True)
    
    with col2:
        # Top Fields of Interest
        st.subheader("Fields of Interest")
        fig_fields = px.pie(
            field_df.head(6),
            values='count',
            names='field',
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_fields.update_layout(height=400)
        st.plotly_chart(fig_fields, use_container_width=True)
    
    # Geographic Distribution
    st.subheader("Geographic Distribution")
    country_counts = projects_df['Country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']
    
    fig_geo = px.bar(
        country_counts.head(8),
        x='country',
        y='count',
        color='count',
        color_continuous_scale='viridis',
        labels={'country': 'Country', 'count': 'Number of Projects'}
    )
    fig_geo.update_layout(height=400)
    st.plotly_chart(fig_geo, use_container_width=True)

# PROJECTS TAB
with tab2:
    st.header("Projects Analysis")
    
    # Create the first row with two columns (Bar chart and Map)
    col1, col2 = st.columns(2)

    projects_df_new = pd.read_json("export_All-Project-Cards_2025-02-21_16-50-28.ndjson", lines=True)
    projects_df_new['Dynamic Link'] = projects_df_new['Dynamic Link'].replace('', pd.NA).fillna('No Link')

    with col1:
        # Projects by Country Bar Chart
        st.subheader("Projects by Country")
        country_counts = projects_df['Country'].value_counts().reset_index()
        country_counts.columns = ['country', 'count']
        
        fig_countries = px.bar(
            country_counts,
            y='country',
            x='count',
            orientation='h',
            color='count',
            color_continuous_scale='Viridis',
            labels={'country': 'Country', 'count': 'Number of Projects'}
        )
        fig_countries.update_layout(height=500)
        st.plotly_chart(fig_countries, use_container_width=True)

        # Second row: Fields of Interest Treemap
        st.subheader("Fields of Interest Distribution")
        fig_treemap = px.treemap(
            field_df,
            path=['field'],
            values='count',
            color='count',
            color_continuous_scale='RdBu'
        )
        fig_treemap.update_layout(height=500)
        st.plotly_chart(fig_treemap, use_container_width=True)
    
    with col2:
        # Projects by Country Map (Choropleth)
        st.subheader("Projects by Country Map")
        fig_countries_map = px.choropleth(
            country_counts,
            locations='country',
            locationmode='country names',
            color='count',
            color_continuous_scale='Viridis',
            title='Number of Projects by Country',
        )
        fig_countries_map.update_layout(
            height=500,
            geo=dict(
                showframe=True,
                showcoastlines=True,
                projection_type='equirectangular',
                lakecolor='white',
                landcolor='lightgray',
                subunitwidth=1,
                countrywidth=1,
            )
        )
        st.plotly_chart(fig_countries_map, use_container_width=True)

        # Projects by Followers from new data
        st.subheader("Projects by Followers")

        # Clean and prep data
        projects_following = projects_df_new[['Dynamic Link', 'Followers (Received)']].copy()
        projects_following['Followers (Received)'] = pd.to_numeric(projects_following['Followers (Received)'], errors='coerce')
        projects_following = projects_following.dropna(subset=['Dynamic Link', 'Followers (Received)'])
        projects_following = projects_following.groupby('Dynamic Link', as_index=False)['Followers (Received)'].sum()
        projects_following = projects_following.sort_values(by='Followers (Received)', ascending=False)

        # Plot
        fig_projects_following = px.bar(
            projects_following,
            x='Followers (Received)',
            y='Dynamic Link',
            orientation='h',
            labels={'Dynamic Link': 'Project Dynamic Link', 'Followers (Received)': 'Followers'},
            color='Followers (Received)',
            color_continuous_scale='Blues'
        )
        fig_projects_following.update_layout(height=500, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_projects_following, use_container_width=True)





# SKILLS TAB
with tab3:
    st.header("Skills Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Skills by Title and Community Count
        st.subheader("Skills by Title (broken down by Community)")

        # Count how many times each skill title appears in each community
        skill_counts = skills_df.groupby(['Title', 'Community']).size().reset_index(name='count')

        # Create bar chart
        fig_skills_comm = px.bar(
            skill_counts,
            x='count',
            y='Title',
            color='Community',
            orientation='h',
            barmode='stack',
            labels={'count': 'Count', 'Title': 'Skill Title', 'Community': 'Community'}
        )

        fig_skills_comm.update_layout(
            height=500,
            xaxis_tickangle=-45,
            legend_title_text='Community'
        )

        st.plotly_chart(fig_skills_comm, use_container_width=True)
        
    with col2:
        # Skills Growth Timeline
        st.subheader("Skills Growth Timeline")
        fig_skills_time = px.line(
            timeline_df,
            x='monthLabel',
            y='skills',
            markers=True,
            labels={'monthLabel': 'Month', 'skills': 'Number of Skills Added'}
        )
        fig_skills_time.update_traces(line_color='#0088FE')
        fig_skills_time.update_layout(height=500)
        st.plotly_chart(fig_skills_time, use_container_width=True)
    
    # Add some insights about skills
    priority_percentage = round(len(skills_df[skills_df['Priority'] == 'Yes']) / len(skills_df) * 100)
    st.info(f"**{priority_percentage}%** of all skills are marked as priority skills, suggesting these are in high demand within the communities.")


# SEARCHES TAB
with tab4:
    st.header("Search Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top Search Terms
        st.subheader("Top Search Terms")
        fig_terms = px.bar(
            search_terms_df.head(10),
            y='term',
            x='count',
            orientation='h',
            color='count',
            color_continuous_scale='Viridis',
            labels={'term': 'Search Term', 'count': 'Number of Searches'}
        )
        fig_terms.update_layout(height=500)
        st.plotly_chart(fig_terms, use_container_width=True)
    
    with col2:
        # Search Activity Timeline
        st.subheader("Search Activity Timeline")
        fig_search_time = px.line(
            timeline_df,
            x='monthLabel',
            y='searches',
            markers=True,
            labels={'monthLabel': 'Month', 'searches': 'Number of Searches'}
        )
        fig_search_time.update_traces(line_color='#00C49F')
        fig_search_time.update_layout(height=500)
        st.plotly_chart(fig_search_time, use_container_width=True)
    
    # Searches by Tab Type
    st.subheader("Search Distribution by Tab")
    tab_counts = search_df['Tab'].value_counts().reset_index()
    tab_counts.columns = ['tab', 'count']
    
    fig_tabs = px.pie(
        tab_counts,
        values='count',
        names='tab',
        color_discrete_sequence=['#0088FE', '#00C49F'],
        hole=0.4
    )
    
    # Calculate percentage for Explore tab
    explore_percentage = round(len(search_df[search_df['Tab'] == 'Explore']) / len(search_df) * 100)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig_tabs, use_container_width=True)
    with col2:
        st.markdown(f"""
        ### Key Search Patterns
        - **{explore_percentage}%** of searches occur in the **Explore tab**
        - Only **{len(search_df[search_df['Pinned?'] == 'Yes'])}** searches are pinned
        - Peak search activity was in **June 2024** with {timeline_df['searches'].max()} searches
        - Most common search term: **"{search_terms_df.iloc[0]['term']}"**
        """)

# COMMUNITY TAB
with tab5:
    st.header("Community Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Community Engagement Metrics
        st.subheader("Community Engagement Metrics")
        fig_engage = px.bar(
            engagement_df,
            y='community',
            x=['projects', 'skills', 'searches'],
            orientation='h',
            barmode='stack',
            labels={'value': 'Count', 'community': 'Community'},
            color_discrete_map={
                'projects': '#0088FE', 
                'skills': '#00C49F', 
                'searches': '#FFBB28'
            }
        )
        fig_engage.update_layout(height=500)
        st.plotly_chart(fig_engage, use_container_width=True)
    
    with col2:
        # Total Community Engagement
        st.subheader("Total Community Engagement")
        fig_total = px.bar(
            engagement_df.head(8),
            x='community',
            y='total',
            color='total',
            color_continuous_scale='Viridis',
            labels={'community': 'Community', 'total': 'Total Engagement'}
        )
        fig_total.update_layout(height=500)
        st.plotly_chart(fig_total, use_container_width=True)
    
    # Community Engagement Radar
    st.subheader("Community Engagement Radar")
    
    # Create radar chart with Plotly
    top_communities = engagement_df.head(6)
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=top_communities['projects'],
        theta=top_communities['community'],
        fill='toself',
        name='Projects',
        line_color='#0088FE'
    ))
    
    fig_radar.add_trace(go.Scatterpolar(
        r=top_communities['skills'],
        theta=top_communities['community'],
        fill='toself',
        name='Skills',
        line_color='#00C49F'
    ))
    
    fig_radar.add_trace(go.Scatterpolar(
        r=top_communities['searches'],
        theta=top_communities['community'],
        fill='toself',
        name='Searches',
        line_color='#FFBB28'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, top_communities[['projects', 'skills', 'searches']].max().max()]
            )),
        showlegend=True
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

# NEEDS TAB
with tab6:
    st.header("Needs Analysis")

    import pandas as pd
    import plotly.express as px

    needs_df = pd.read_json("export_All-Needs_2025-04-04_18-20-21.ndjson", lines=True)

    needs_df = needs_df.rename(columns={
        "Close Attributed to": "Close Attributed",
        "Community": "Community List",
        "Project": "Project Name",
        "Status": "Need Status",
        "Text": "Need Description",
        "Type": "Type of Need"
    })

    needs_df = needs_df[['Need Status', 'Type of Need', 'Community List']].dropna()

    needs_df['Need Status'] = needs_df['Need Status'].str.strip().str.lower()
    needs_df['Status Group'] = needs_df['Need Status'].apply(
        lambda x: 'Closed' if 'completed' in x else ('Open' if 'help offered' in x else 'Other')
    )

    type_status_counts = needs_df.groupby(['Type of Need', 'Status Group']).size().reset_index(name='count')
    type_status_fig = px.bar(
        type_status_counts,
        x='Type of Need',
        y='count',
        color='Status Group',
        barmode='group',
        labels={'count': 'Number of Needs'},
        title='Open vs Closed Needs by Type'
    )
    type_status_fig.update_layout(height=450, xaxis_tickangle=-45)
    st.plotly_chart(type_status_fig, use_container_width=True)

    comm_status_counts = needs_df.groupby(['Community List', 'Status Group']).size().reset_index(name='count')
    comm_status_fig = px.bar(
        comm_status_counts,
        x='Community List',
        y='count',
        color='Status Group',
        barmode='group',
        labels={'count': 'Number of Needs'},
        title='Open vs Closed Needs by Community'
    )
    comm_status_fig.update_layout(height=450, xaxis_tickangle=-45)
    st.plotly_chart(comm_status_fig, use_container_width=True)

    # Chart: Number of Closes by Person
    st.subheader("Number of Closes by Person")

    # Load full data again with 'Close Attributed to' column preserved
    closes_df = pd.read_json("export_All-Needs_2025-04-04_18-20-21.ndjson", lines=True)

    # Clean the field
    closes_df['Close Attributed to'] = closes_df['Close Attributed to'].replace('', pd.NA).fillna('Unattributed')
    closes_df = closes_df[closes_df['Close Attributed to'].notna()]

    # Count number of closes per person
    closes_by_person = closes_df['Close Attributed to'].value_counts().reset_index()
    closes_by_person.columns = ['Person', 'Number of Closes']

    # Plot
    fig_closes = px.bar(
        closes_by_person,
        x='Number of Closes',
        y='Person',
        orientation='h',
        color='Number of Closes',
        color_continuous_scale='Teal',
        title='Closes Attributed by Person'
    )
    fig_closes.update_layout(height=500, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_closes, use_container_width=True)


# Key Insights Section
st.header("Key Insights")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("Project Insights")
        st.markdown("""
        - Most projects are focused on **community building** (14 projects), with **business** (7) and **technology development** (5) also being popular
        - Majority of projects (50%) are based in the USA, with international presence across 11 countries
        - Diverse range of fields suggests broad entrepreneurial interests in the community
        """)
    
    with st.container(border=True):
        st.subheader("Skills Insights")
        priority_percentage = round(len(skills_df[skills_df['Priority'] == 'Yes']) / len(skills_df) * 100)
        st.markdown(f"""
        - {priority_percentage}% of registered skills are marked as priority
        - Major skill growth occurred in October 2024 (180 skills added), suggesting a platform initiative or event
        - Skills are evenly distributed across communities, suggesting consistent template application
        """)

with col2:
    with st.container(border=True):
        st.subheader("Search Insights")
        explore_percentage = round(len(search_df[search_df['Tab'] == 'Explore']) / len(search_df) * 100)
        st.markdown(f"""
        - **{explore_percentage}%** of searches occur in the Explore tab, showing platform discovery is key
        - Peak search activity was in June 2024 (26 searches), possibly indicating a promotional event
        - Top search terms include "virtual" (4 searches), "development", and "agency" (3 each), revealing user interests
        """)
    
    with st.container(border=True):
        st.subheader("Community Insights")
        st.markdown("""
        - "Adjacent" is the most active community with highest engagement across all metrics
        - Community naming inconsistency exists (case sensitivity: "Adjacent" vs "adjacent")
        - Communities show different engagement patterns - some focused on projects, others on skills
        """)