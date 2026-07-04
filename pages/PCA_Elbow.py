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
.year-label { text-align: center; font-size: 18px; font-weight: bold; color: #444444; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

NEW_DIR = 'New PCA El n Sil'

# ── Load model & data ──────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open(f'{NEW_DIR}/scaler(0,31).pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(f'{NEW_DIR}/pca_reducer(0,31).pkl', 'rb') as f:
        pca = pickle.load(f)
    return scaler, pca

@st.cache_data
def load_data():
    df = pd.read_csv(f'{NEW_DIR}/main_data_with_pred(0,31).csv')
    with open('BandungRaya_merged.geojson') as f:
        geojson = json.load(f)
    return df, geojson

scaler, pca = load_models()
main_data, geojson_bandung = load_data()

try:
    main_image = Image.open('Pic2.png')
except FileNotFoundError:
    main_image = None

# ── Hitung PC1 & PC2 untuk scatter ────────────────────────────────────────────
FEATURE_COLS = ['NDVI', 'NDWI', 'elevation', 'slope', 'kepadatan',
                'rainfall', 'TEXTURE_USDA', 'DRAINAGE', 'AWC', 'GFI']

df = main_data.copy()
X_pca      = pca.transform(scaler.transform(df[FEATURE_COLS]))
df['PC1']  = X_pca[:, 0]
df['PC2']  = X_pca[:, 1]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="title">Cluster Banjir Pada Bandung Raya (PCA Elbow)</h1>', unsafe_allow_html=True)
if main_image:
    st.image(main_image, use_container_width=True)

# ── Scatter PCA + Karakteristik ────────────────────────────────────────────────
left_col, right_col = st.columns(2)

with left_col:
    st.markdown('<h2 class="subheader">Visualisasi Cluster (PCA)</h2>', unsafe_allow_html=True)

    scatter_df = df.copy()
    scatter_df['cluster_label'] = scatter_df['cluster'].astype(str).apply(lambda x: f"Cluster {x}")

    fig = px.scatter(
        scatter_df,
        x='PC1', y='PC2',
        color='cluster_label',
        title="Sebaran Cluster dengan PCA",
        labels={'PC1': 'PC 1', 'PC2': 'PC 2'},
        template='plotly',
        category_orders={'cluster_label': sorted(scatter_df['cluster_label'].unique())}
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(height=600, legend_title_text='Cluster')
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.markdown('<h2 class="subheader">Karakteristik Cluster</h2>', unsafe_allow_html=True)

    exclude_cols = {'kecamata', 'kecamatan', 'kab_kota', 'year', 'month',
                    'lon', 'lat', 'PC1', 'PC2', 'cluster', 'cluster_sil'}
    feat_cols = [c for c in df.columns if c not in exclude_cols]
    cat_cols  = {'TEXTURE_USDA', 'DRAINAGE'}
    agg_dict  = {col: (lambda x: x.mode()[0]) if col in cat_cols else 'mean' for col in feat_cols}
    karakteristik_df = (
        df.groupby('cluster')
        .agg(agg_dict)
        .round(3)
        .reset_index()
    )

    n      = len(karakteristik_df)
    cell_h = max(60, (600 - 40) // n)

    fig_table = go.Figure(data=[go.Table(
        header=dict(
            values=list(karakteristik_df.columns),
            fill_color='#1e3a5f',
            font=dict(color='white', size=16),
            align='center',
            height=40,
        ),
        cells=dict(
            values=[karakteristik_df[col] for col in karakteristik_df.columns],
            fill_color=[['#0e1117', '#1a1a2e'] * n],
            font=dict(color='white', size=15),
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

# ── Peta per Tahun ─────────────────────────────────────────────────────────────
st.markdown('<h2 class="subheader">Peta Sebaran Cluster per Tahun</h2>', unsafe_allow_html=True)

years    = sorted(df['year'].unique())
map_cols = st.columns(len(years))

for col, year in zip(map_cols, years):
    with col:
        st.markdown(f'<p class="year-label">Tahun {year}</p>', unsafe_allow_html=True)

        year_kec = (
            df[df['year'] == year]
            .groupby('kecamata')['cluster']
            .agg(lambda x: x.mode()[0])
            .reset_index()
        )
        year_kec['cluster_label'] = year_kec['cluster'].astype(str).apply(lambda x: f"Cluster {x}")

        fig_year = px.choropleth_mapbox(
            year_kec,
            geojson=geojson_bandung,
            locations='kecamata',
            featureidkey='properties.kecamata',
            color='cluster_label',
            color_discrete_sequence=px.colors.qualitative.Set1,
            category_orders={'cluster_label': sorted(year_kec['cluster_label'].unique())},
            mapbox_style='carto-positron',
            zoom=9,
            center={'lat': -6.9175, 'lon': 107.6191},
            opacity=0.6,
            labels={'cluster_label': 'Cluster'}
        )
        fig_year.update_layout(
            margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
            height=500,
            legend=dict(
                title="Cluster",
                font=dict(color="black"),
                title_font=dict(color="black"),
                bgcolor="rgba(255,255,255,0.8)"
            )
        )
        st.plotly_chart(fig_year, use_container_width=True)
