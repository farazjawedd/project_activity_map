import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import umap
import streamlit as st  # <-- The main import

# --- 1. Page Configuration (Sets up the web page) ---
st.set_page_config(
    page_title="Project Activity Explorer",
    page_icon="🗺️",
    layout="wide"  # Use the full width of the screen
)

# --- 2. Data Processing (Cached) ---
# We put all the slow code into one function.
# @st.cache_data tells Streamlit to run this
# *only once* and save the result.
@st.cache_data
def load_and_process_data():
    print("--- RUNNING DATA PROCESSING (This happens once) ---")
    
    # --- Load Data ---
    try:
        df = pd.read_csv('data.csv')
    except FileNotFoundError:
        st.error("Error: 'data.csv' not found. Please place it in the same folder.")
        return None
        
    df.dropna(subset=['Activity', 'Sectors'], inplace=True)
    df['Activity'] = df['Activity'].astype(str)
    
    # --- TF-IDF ---
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(df['Activity'])
    
    # --- LSA (SVD) ---
    n_topics = 100
    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    lsa_embeddings = svd.fit_transform(tfidf_matrix)
    
    # --- UMAP ---
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.0,
        n_components=2,
        metric='cosine',
        random_state=42
    )
    embedding_2d = reducer.fit_transform(lsa_embeddings)
    
    # Add coordinates back to the original dataframe
    df['x'] = embedding_2d[:, 0]
    df['y'] = embedding_2d[:, 1]
    
    return df

# --- 3. Main Application UI ---
st.title("🗺️ Interactive Project Activity Explorer")
st.write("This app visualizes project activities in a 2D 'topic map' using TF-IDF, LSA, and UMAP.")

# Load the data (this will be instant after the first run)
df = load_and_process_data()

if df is not None:
    # --- 4. Interactive Filters (in a sidebar) ---
    st.sidebar.header("Explore the Data")
    
    # --- Sector Filter (Your request) ---
    all_sectors = sorted(df['Sectors'].unique())
    selected_sectors = st.sidebar.multiselect(
        "Filter by Sector:",
        options=all_sectors,
        default=all_sectors  # Start with all selected
    )
    
    # --- Activity Search (Your request) ---
    search_term = st.sidebar.text_input(
        "Search in Activity text (case-insensitive):"
    )

    # --- 5. Filter the Dataframe ---
    df_filtered = df[df['Sectors'].isin(selected_sectors)]
    
    if search_term:
        df_filtered = df_filtered[
            df_filtered['Activity'].str.contains(search_term, case=False, na=False)
        ]

    # --- 6. Create and Display the Plot ---
    st.header(f"Showing {len(df_filtered)} of {len(df)} Activities")

    if len(df_filtered) == 0:
        st.warning("No activities match your filters.")
    else:
        fig = px.scatter(
            df_filtered,
            x='x',
            y='y',
            color='Sectors',
            hover_name='Activity',
            hover_data={'Sectors': True, 'Activity': False, 'x': False, 'y': False},
            title='Interactive LSA/Topic Map of Activities by Sector',
            height=700
        )
        
        # This is the magic line that displays the Plotly chart
        st.plotly_chart(fig, use_container_width=True)
    
    # --- 7. (Optional) Show the raw data ---
    with st.expander("See filtered data table"):
        st.dataframe(df_filtered)