import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

CSS = """
<style>
.block-container { padding-top: 3rem; }
.title { text-align: center; font-size: 32px; font-weight: bold; color: #333333; padding-top: 0.75rem; line-height: 1.3; }
.subheader { text-align: center; font-size: 24px; font-weight: bold; color: #555555; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Load model & data ──────────────────────────────────────────────────────────
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('umap_reducer.pkl', 'rb') as f:
    reducer = pickle.load(f)

umap_2d_df = pd.read_csv('koordinat_umap.csv')
main_data  = pd.read_csv('hasil_kmeans_umap.csv')

with open('BandungRaya_merged.geojson') as f:
    geojson_bandung = json.load(f)

try:
    main_image = Image.open('Pic2.png')
except FileNotFoundError:
    main_image = None

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="title">Cluster Banjir Pada Bandung Raya (UMAP)</h1>', unsafe_allow_html=True)
if main_image:
    st.image(main_image, use_container_width=True)

# ── Scatter UMAP + Karakteristik ───────────────────────────────────────────────
left_col, right_col = st.columns(2)

with left_col:
    st.markdown('<h2 class="subheader">Visualisasi Cluster (UMAP)</h2>', unsafe_allow_html=True)
    scatter_df = umap_2d_df.copy()
    scatter_df['cluster_label'] = scatter_df['cluster'].astype(str).apply(lambda x: f"Cluster {x}")
    fig = px.scatter(
        scatter_df,
        x='UMAP1', y='UMAP2',
        color='cluster_label',
        title="Sebaran Cluster dengan UMAP",
        labels={'UMAP1': 'Dimensi UMAP 1', 'UMAP2': 'Dimensi UMAP 2'},
        template='plotly',
        category_orders={'cluster_label': sorted(scatter_df['cluster_label'].unique())}
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(height=600, legend_title_text='Cluster')
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.markdown('<h2 class="subheader">Karakteristik Cluster</h2>', unsafe_allow_html=True)

    exclude_cols = {'UMAP1', 'UMAP2', 'kecamata', 'kecamatan', 'year', 'month', 'lon', 'lat', 'kab_kota'}
    feat_cols = [c for c in main_data.columns if c not in exclude_cols and c != 'cluster']
    karakteristik_df = (
        main_data.groupby('cluster')[feat_cols]
        .mean(numeric_only=True)
        .round(3)
        .reset_index()
    )

    n = len(karakteristik_df)
    cell_h = max(60, (600 - 40) // n)

    fig_table = go.Figure(data=[go.Table(
        header=dict(
            values=list(karakteristik_df.columns),
            fill_color='#1e3a5f',
            font=dict(color='white', size=20),
            align='center',
            height=40,
        ),
        cells=dict(
            values=[karakteristik_df[col] for col in karakteristik_df.columns],
            fill_color=[['#0e1117', '#1a1a2e'] * n],
            font=dict(color='white', size=20),
            align='center',
            height=cell_h,
        )
    )])
    fig_table.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_table, use_container_width=True, config={"staticPlot": True})

# ── Peta Sebaran Cluster (semua tahun digabung) ────────────────────────────────
st.markdown('<h2 class="subheader">Peta Sebaran Cluster dalam 3 Tahun</h2>', unsafe_allow_html=True)

map_data = main_data.copy()
map_data['cluster'] = map_data['cluster'].astype(str).apply(lambda x: f"Cluster {x}")

fig_map = px.choropleth_mapbox(
    map_data,
    geojson=geojson_bandung,
    locations="kecamata",
    featureidkey="properties.kecamata",
    color="cluster",
    color_discrete_sequence=px.colors.qualitative.Set1,
    category_orders={"cluster": sorted(map_data['cluster'].unique())},
    mapbox_style="carto-positron",
    zoom=10,
    center={"lat": -6.9175, "lon": 107.6191},
    opacity=0.6,
    labels={'cluster': 'Cluster'}
)
fig_map.update_layout(
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    height=1000,
    legend=dict(
        title="Cluster", x=0.01, y=0.99,
        bgcolor="rgba(255,255,255,0.8)",
        font=dict(color="black"),
        title_font=dict(color="black")
    )
)
st.plotly_chart(fig_map, use_container_width=True)
