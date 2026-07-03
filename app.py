import streamlit as st

st.set_page_config(layout="wide", page_title="Dashboard Banjir Bandung")

pg = st.navigation([
    st.Page("pages/UMAP.py",          title="UMAP"           ),
    st.Page("pages/PCA_Elbow.py",     title="PCA Elbow"     ),
    st.Page("pages/PCA_Silhouette.py",title="PCA Silhouette" ),
])
pg.run()
