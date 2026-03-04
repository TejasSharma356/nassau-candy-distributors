"""Nassau Candy Distributor - Shipping Route Efficiency Dashboard."""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path for imports (must be before src import)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)
# pylint: disable=wrong-import-position
from src.analytics import compute_route_kpis, merge_factory_data  # noqa: E402
from src.ml_model import load_model, build_features  # noqa: E402

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy · Shipping Analytics",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# PREMIUM THEME CSS
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main { background: linear-gradient(135deg, #0a0e17 0%, #111827 50%, #0f172a 100%); }
.stApp { background: transparent; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #1e293b 100%);
    border-right: 1px solid rgba(99,102,241,0.15);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
}

/* ── Glassmorphism card ── */
div[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 20px 24px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(99,102,241,0.2);
}
div[data-testid="stMetric"] label {
    color: #94a3b8 !important;
    font-weight: 500;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-weight: 700;
    font-size: 1.8rem;
}

/* ── Section headers ── */
h1 {
    background: linear-gradient(90deg, #818cf8, #6366f1, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}
h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

/* ── Tabs container ── */
div[data-testid="stTabs"] button {
    color: #94a3b8 !important;
    font-weight: 500;
    border-bottom-color: transparent !important;
    transition: all 0.2s ease;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #818cf8 !important;
    border-bottom-color: #6366f1 !important;
}
div[data-testid="stTabs"] button:hover {
    color: #c7d2fe !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    background: rgba(30, 41, 59, 0.4);
    border-radius: 12px;
    border: 1px solid rgba(99,102,241,0.12);
}

/* ── Divider ── */
hr { border-color: rgba(99,102,241,0.15) !important; }

/* ── Plotly charts background ── */
.js-plotly-plot .plotly .main-svg { border-radius: 12px; }

/* ── Glass container ── */
.glass-container {
    background: rgba(30, 41, 59, 0.45);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}
.glass-container h3 {
    margin-top: 0;
    color: #c7d2fe !important;
}

/* ── Badge ── */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-green  { background: rgba(34,197,94,0.15); color: #4ade80; }
.badge-red    { background: rgba(239,68,68,0.15); color: #f87171; }
.badge-blue   { background: rgba(99,102,241,0.15); color: #818cf8; }
.badge-amber  { background: rgba(245,158,11,0.15); color: #fbbf24; }

/* ── Hero banner ── */
.hero {
    padding: 32px 0 8px 0;
}
.hero-sub {
    color: #64748b;
    font-size: 1.05rem;
    margin-top: -8px;
    font-weight: 400;
}

/* ── Selectbox Cursor ── */
div[data-baseweb="select"] > div {
    cursor: pointer !important;
}
div[data-baseweb="select"] input {
    cursor: pointer !important;
}

/* ── Smooth fade-in animation ── */
@keyframes fadeInUp {
    0%   { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
.main .block-container { animation: fadeInUp 0.6s ease-out; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ──────────────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(family="Inter", color="#cbd5e1", size=12),
    margin=dict(l=16, r=16, t=36, b=16),
    coloraxis_colorbar=dict(
        bgcolor="rgba(0,0,0,0)",
        tickfont=dict(color="#94a3b8"),
        title_font=dict(color="#94a3b8"),
    ),
)

ACCENT = "#818cf8"
GRADIENT_GOOD = [[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#22c55e"]]
GRADIENT_BAD = [[0, "#22c55e"], [0.5, "#f59e0b"], [1, "#ef4444"]]


# ──────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and merge all datasets for the dashboard."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_orders = pd.read_csv(
        os.path.join(base, "data/processed/cleaned_orders.csv")
    )
    raw_orders['Order Date'] = pd.to_datetime(raw_orders['Order Date'])
    raw_orders['Ship Date'] = pd.to_datetime(raw_orders['Ship Date'])
    mapping_df = pd.read_csv(
        os.path.join(base, "data/raw/product_factories.csv")
    )
    factory_coords = pd.read_csv(
        os.path.join(base, "data/raw/factory_coordinates.csv")
    )
    merged = merge_factory_data(raw_orders, mapping_df)
    return merged, factory_coords


orders_full, coords_df = load_data()

# ──────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍬 Nassau Candy")
    st.caption("Shipping Analytics Console")
    st.divider()

    # Date range
    min_date = orders_full['Order Date'].min().date()
    max_date = orders_full['Order Date'].max().date()
    try:
        date_range = st.date_input(
            "📅 Order Date Range", (min_date, max_date)
        )
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date
    except (ValueError, TypeError):
        start_date, end_date = min_date, max_date
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    st.divider()

    # Geographic
    regions = ["All Regions"] + sorted(orders_full['Region'].unique())
    selected_region = st.selectbox("🌎 Region", regions)
    if selected_region != "All Regions":
        avail_states = sorted(
            orders_full[orders_full['Region'] == selected_region]['State/Province'].unique()
        )
    else:
        avail_states = sorted(orders_full['State/Province'].unique())
    selected_states = st.multiselect("📍 State / Province", avail_states)

    st.divider()

    # Ship mode
    ship_modes = sorted(orders_full['Ship Mode'].unique())
    selected_modes = st.multiselect(
        "🚚 Ship Mode", ship_modes, default=ship_modes
    )

    # Threshold
    delay_threshold = st.slider(
        "⏱️ Delay Threshold (days)", 1, 30, 7
    )

    st.divider()
    st.markdown(
        '<p style="color:#64748b;font-size:0.78rem;">'
        'All charts & KPIs update live as you change filters.</p>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────
# APPLY FILTERS
# ──────────────────────────────────────────────────────────────────────
filtered_df = orders_full[
    (orders_full['Order Date'] >= start_date)
    & (orders_full['Order Date'] <= end_date)
    & (orders_full['Ship Mode'].isin(selected_modes))
]
if selected_region != "All Regions":
    filtered_df = filtered_df[filtered_df['Region'] == selected_region]
if selected_states:
    filtered_df = filtered_df[
        filtered_df['State/Province'].isin(selected_states)
    ]

if filtered_df.empty:
    st.warning("No data matches the current filters. Adjust your selections.")
    st.stop()

route_kpis = compute_route_kpis(
    filtered_df, threshold=delay_threshold,
    group_by_col='State/Province'
)

# ──────────────────────────────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero">', unsafe_allow_html=True)
st.title("Shipping Route Efficiency")
st.markdown(
    '<p class="hero-sub">'
    'Factory-to-customer logistics intelligence · Nassau Candy Distributor'
    '</p>',
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# TOP KPI CARDS
# ──────────────────────────────────────────────────────────────────────
if not route_kpis.empty:
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Orders", f"{len(filtered_df):,}")
    with k2:
        avg_lt = filtered_df['lead_time_days'].mean()
        st.metric("Avg Lead Time", f"{avg_lt:.1f} days")
    with k3:
        delayed = (filtered_df['lead_time_days'] > delay_threshold).sum()
        delay_pct = (delayed / len(filtered_df)) * 100
        st.metric("Delay Rate", f"{delay_pct:.1f}%")
    with k4:
        avg_eff = route_kpis['route_efficiency_score'].mean()
        st.metric("Efficiency Score", f"{avg_eff:.2f}")

st.markdown("")  # spacer

# ──────────────────────────────────────────────────────────────────────
# TABBED MODULES
# ──────────────────────────────────────────────────────────────────────
tab_overview, tab_map, tab_modes, tab_drill, tab_predict = st.tabs([
    "📊  Route Efficiency",
    "🗺️  Geographic Map",
    "📦  Ship Mode Analysis",
    "🔬  Route Drill-Down",
    "🤖  Predict Delays",
])

# ━━━━━━━━━━ TAB 1 : ROUTE EFFICIENCY ━━━━━━━━━━
with tab_overview:
    st.markdown(
        '<div class="glass-container">'
        '<h3>Route Efficiency Benchmark</h3>'
        '<p style="color:#94a3b8;margin-top:-8px;">'
        'Top 10 most efficient vs bottom 10 least efficient routes</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if len(route_kpis) >= 10:
        top_10 = route_kpis.head(10).copy()
        bottom_10 = route_kpis.tail(10).copy()
        top_10['route_label'] = top_10['FACTORY'] + ' → ' + top_10['State/Province']
        bottom_10['route_label'] = bottom_10['FACTORY'] + ' → ' + bottom_10['State/Province']

        col_t, col_b = st.columns(2)

        with col_t:
            st.markdown(
                '<span class="badge badge-green">✦ TOP 10</span>',
                unsafe_allow_html=True,
            )
            fig_top = px.bar(
                top_10,
                x='route_efficiency_score',
                y='route_label',
                hover_data=['avg_lead_time', 'delay_frequency'],
                color='route_efficiency_score',
                color_continuous_scale=GRADIENT_GOOD,
                orientation='h',
            )
            fig_top.update_layout(
                **CHART_LAYOUT,
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                coloraxis_showscale=False,
                xaxis_title="Efficiency Score",
                yaxis_title="",
            )
            st.plotly_chart(fig_top, use_container_width=True)

        with col_b:
            st.markdown(
                '<span class="badge badge-red">▼ BOTTOM 10</span>',
                unsafe_allow_html=True,
            )
            fig_bot = px.bar(
                bottom_10,
                x='route_efficiency_score',
                y='route_label',
                hover_data=['avg_lead_time', 'delay_frequency'],
                color='route_efficiency_score',
                color_continuous_scale=GRADIENT_BAD,
                orientation='h',
            )
            fig_bot.update_layout(
                **CHART_LAYOUT,
                yaxis={'categoryorder': 'total descending'},
                showlegend=False,
                coloraxis_showscale=False,
                xaxis_title="Efficiency Score",
                yaxis_title="",
            )
            st.plotly_chart(fig_bot, use_container_width=True)
    else:
        st.dataframe(
            route_kpis,
            use_container_width=True,
        )

# ━━━━━━━━━━ TAB 2 : GEOGRAPHIC MAP ━━━━━━━━━━
with tab_map:
    st.markdown(
        '<div class="glass-container">'
        '<h3>Geographic Efficiency Map</h3>'
        '<p style="color:#94a3b8;margin-top:-8px;">'
        'State-level efficiency heatmap with factory locations</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    state_agg = route_kpis.groupby('State/Province').agg(
        route_efficiency_score=('route_efficiency_score', 'mean'),
        route_volume=('route_volume', 'sum'),
        avg_lead_time=('avg_lead_time', 'mean'),
    ).reset_index()

    US_ABBREV = {
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ",
        "Arkansas": "AR", "California": "CA", "Colorado": "CO",
        "Connecticut": "CT", "Delaware": "DE", "Florida": "FL",
        "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
        "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
        "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
        "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
        "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
        "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
        "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
        "New Mexico": "NM", "New York": "NY",
        "North Carolina": "NC", "North Dakota": "ND",
        "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
        "Pennsylvania": "PA", "Rhode Island": "RI",
        "South Carolina": "SC", "South Dakota": "SD",
        "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
        "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
        "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
        "District of Columbia": "DC",
    }
    state_agg['State_Code'] = state_agg['State/Province'].map(US_ABBREV)

    fig_map = px.choropleth(
        state_agg,
        locations='State_Code',
        locationmode="USA-states",
        color='route_efficiency_score',
        hover_name='State/Province',
        hover_data=['route_volume', 'avg_lead_time'],
        color_continuous_scale=GRADIENT_GOOD,
        scope="usa",
        labels={'route_efficiency_score': 'Efficiency'},
    )
    fig_map.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        geo=dict(
            bgcolor="rgba(15,23,42,0.5)",
            lakecolor="rgba(15,23,42,0.5)",
            landcolor="rgba(30,41,59,0.8)",
            subunitcolor="rgba(99,102,241,0.15)",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            title="Efficiency",
            bgcolor="rgba(0,0,0,0)",
            tickfont=dict(color="#94a3b8"),
            title_font=dict(color="#94a3b8"),
        ),
    )

    # Factory markers
    fig_factories = px.scatter_geo(
        coords_df,
        lat='LATITUDE', lon='LONGITUDE',
        hover_name='FACTORY', scope="usa",
    )
    fig_factories.update_traces(marker=dict(
        size=14, color=ACCENT, symbol='star',
        line=dict(width=2, color='white'),
    ))
    fig_map.add_traces(fig_factories.data)
    
    map_col, scatter_col = st.columns([3, 2])
    with map_col:
        st.plotly_chart(fig_map, use_container_width=True)
    
    with scatter_col:
        st.markdown(
            '<p style="text-align:center; font-weight:600; margin-bottom:-5px;">'
            'Geographic Bottleneck Analysis</p>',
            unsafe_allow_html=True
        )
        # Scatter plot for Volume vs Poor Performance
        fig_scatter = px.scatter(
            state_agg,
            x='route_volume',
            y='avg_lead_time',
            color='route_efficiency_score',
            color_continuous_scale=GRADIENT_BAD, # Lower score = Red
            hover_name='State/Province',
            size='route_volume',
            labels={'route_volume': 'Shipment Volume', 'avg_lead_time': 'Avg Lead Time (Days)'}
        )
        fig_scatter.update_layout(
            **CHART_LAYOUT,
            height=300,
            coloraxis_showscale=False
        )
        fig_scatter.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Top Right = High Volume + High Delay (Critical Bottlenecks)")

    # Summary metrics row
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        best = state_agg.loc[state_agg['route_efficiency_score'].idxmax()]
        st.metric("🏆 Best State", best['State/Province'],
                  f"Score: {best['route_efficiency_score']:.2f}")
    with mc2:
        worst = state_agg.loc[state_agg['route_efficiency_score'].idxmin()]
        st.metric("⚠️ Worst State", worst['State/Province'],
                  f"Score: {worst['route_efficiency_score']:.2f}")
    with mc3:
        st.metric("📍 States Covered", len(state_agg))

# ━━━━━━━━━━ TAB 3 : SHIP MODE COMPARISON ━━━━━━━━━━
with tab_modes:
    st.markdown(
        '<div class="glass-container">'
        '<h3>Ship Mode Performance</h3>'
        '<p style="color:#94a3b8;margin-top:-8px;">'
        'Lead time distributions and delay rates by shipping method</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    col_box, col_tbl = st.columns([2, 1])

    with col_box:
        fig_mode = px.box(
            filtered_df,
            x="Ship Mode", y="lead_time_days",
            color="Ship Mode",
            points="outliers",
            color_discrete_sequence=["#818cf8", "#22c55e", "#f59e0b", "#ef4444"],
        )
        fig_mode.update_layout(
            **CHART_LAYOUT,
            showlegend=False,
            xaxis_title="",
            yaxis_title="Lead Time (days)",
        )
        st.plotly_chart(fig_mode, use_container_width=True)

    with col_tbl:
        mode_tmp = filtered_df.copy()
        mode_tmp['_delayed'] = (
            mode_tmp['lead_time_days'] > delay_threshold
        ).astype(int)
        mode_summary = mode_tmp.groupby('Ship Mode').agg(
            Orders=('Order ID', 'count'),
            Avg_Days=('lead_time_days', 'mean'),
            Delay_Rate=('_delayed', 'mean'),
        ).reset_index()
        mode_summary['Avg_Days'] = mode_summary['Avg_Days'].round(1)
        mode_summary['Delay_Rate'] = (
            (mode_summary['Delay_Rate'] * 100).round(1).astype(str) + '%'
        )
        st.dataframe(
            mode_summary, use_container_width=True, hide_index=True
        )

        # Quick insight
        fastest = mode_summary.loc[
            mode_summary['Avg_Days'].idxmin(), 'Ship Mode'
        ]
        st.markdown(
            f'<span class="badge badge-blue">⚡ Fastest: {fastest}</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("#### Cost-Time Tradeoff Analysis")
    st.info(
        "**Descriptive Evaluation:** \n"
        "- **Same Day / First Class**: These expedited shipping methods provide maximum customer satisfaction with near-zero lead times. However, they carry premium logistical overhead. Ideal for high-margin, fragile, or time-sensitive bulk orders.\n"
        "- **Second Class**: The optimal middle-ground. Significantly reduces freight costs while maintaining acceptable delay thresholds.\n"
        "- **Standard Class**: The most economical choice for bulk inventory restocks where volume > velocity. It yields the highest delay rates, but massive freight cost savings."
    )

# ━━━━━━━━━━ TAB 4 : DRILL-DOWN ━━━━━━━━━━
with tab_drill:
    st.markdown(
        '<div class="glass-container">'
        '<h3>Route Drill-Down</h3>'
        '<p style="color:#94a3b8;margin-top:-8px;">'
        'Inspect a specific factory→state route in detail</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        factories = sorted(filtered_df['FACTORY'].dropna().unique())
        drill_factory = st.selectbox("Select Factory", factories)
    with col_f2:
        drill_states = sorted(
            filtered_df[
                filtered_df['FACTORY'] == drill_factory
            ]['State/Province'].unique()
        )
        if len(drill_states) > 0:
            chosen_state = st.selectbox(  # pylint: disable=invalid-name
                "Select Destination State", drill_states
            )
        else:
            chosen_state = None  # pylint: disable=invalid-name
            st.warning("No destinations for this factory under current filters.")

    if chosen_state:
        route_data = filtered_df[
            (filtered_df['FACTORY'] == drill_factory)
            & (filtered_df['State/Province'] == chosen_state)
        ]

        # KPI row
        dk1, dk2, dk3, dk4 = st.columns(4)
        with dk1:
            st.metric("📦 Volume", len(route_data))
        with dk2:
            avg = route_data['lead_time_days'].mean()
            st.metric("⏱️ Avg Lead Time", f"{avg:.1f} days")
        with dk3:
            r_delays = (route_data['lead_time_days'] > delay_threshold).sum()
            freq = (r_delays / len(route_data)) * 100
            st.metric("🔴 Delay Rate", f"{freq:.1f}%")
        with dk4:
            std = route_data['lead_time_days'].std()
            st.metric("📊 Std Dev", f"{std:.1f}" if pd.notna(std) else "N/A")

        st.markdown("")

        # Histogram
        fig_hist = px.histogram(
            route_data, x="lead_time_days", nbins=20,
            color_discrete_sequence=[ACCENT],
            opacity=0.85,
        )
        fig_hist.add_vline(
            x=delay_threshold, line_dash="dash",
            line_color="#ef4444",
            annotation_text=f"Threshold ({delay_threshold}d)",
            annotation_font_color="#f87171",
        )
        fig_hist.update_layout(
            **CHART_LAYOUT,
            xaxis_title="Lead Time (days)",
            yaxis_title="Order Count",
            bargap=0.08,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Order table
        st.markdown(
            '<span class="badge badge-amber">'
            f'📋 Showing up to 500 of {len(route_data)} orders</span>',
            unsafe_allow_html=True,
        )
        disp_cols = [
            'Order ID', 'Order Date', 'Ship Date', 'Ship Mode',
            'Sales', 'Units', 'Gross Profit', 'Cost', 'lead_time_days',
        ]
        safe_cols = [c for c in disp_cols if c in route_data.columns]
        drill_view = route_data[safe_cols].head(500).copy()
        if 'Order Date' in drill_view:
            drill_view['Order Date'] = (
                drill_view['Order Date'].dt.strftime('%Y-%m-%d')
            )
        if 'Ship Date' in drill_view:
            drill_view['Ship Date'] = (
                drill_view['Ship Date'].dt.strftime('%Y-%m-%d')
            )
        st.dataframe(drill_view, use_container_width=True, hide_index=True)


# ━━━━━━━━━━ TAB 5 : ML PREDICTION ━━━━━━━━━━
with tab_predict:
    st.markdown(
        '<div class="glass-container">'
        '<h3>AI Delay Predictor</h3>'
        '<p style="color:#94a3b8;margin-top:-8px;">'
        'Uses a Random Forest model trained on historical data to predict if an '
        'order configuration will be abnormally delayed.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    try:
        model, metrics, encoders = load_model()

        model, metrics, encoders = load_model()

        st.markdown("#### Configure Order")
        
        # We use regions/states found in orders_full
        form_col1, form_col2, form_col3 = st.columns(3)
        
        with form_col1:
            regs = sorted(orders_full['Region'].dropna().unique())
            p_region = st.selectbox("Region", regs, index=None, placeholder="Choose...", key='p_reg')

            states_for_reg = []
            if p_region:
                states_for_reg = sorted(
                    orders_full[orders_full['Region'] == p_region]
                    ['State/Province'].dropna().unique()
                )
            p_state = st.selectbox("State", states_for_reg, index=None, placeholder="Choose...", key='p_st')

        with form_col2:
            facs = sorted(orders_full['FACTORY'].dropna().unique())
            p_factory = st.selectbox("Factory", facs, index=None, placeholder="Choose...", key='p_fac')

            modes = sorted(orders_full['Ship Mode'].dropna().unique())
            p_mode = st.selectbox("Ship Mode", modes, index=None, placeholder="Choose...", key='p_mod')

        with form_col3:
            # Simple numeric estimates
            p_sales = st.number_input("Est. Sales ($)", min_value=0.0, value=500.0)
            p_units = st.number_input("Units", min_value=1, value=50)

        st.markdown("<br>", unsafe_allow_html=True)
        # Fix button visibility by removing type="primary" 
        submit_pred = st.button("Predict Delay Risk", use_container_width=True)

        st.divider()
        st.markdown("#### Prediction Result")
        if submit_pred:
            if not all([p_region, p_state, p_factory, p_mode]):
                st.warning("⚠️ Please select a value for all options (Region, State, Factory, Ship Mode) to run the prediction.")
            else:
                # Build a single-row dataframe for prediction
                pred_df = pd.DataFrame({
                    'Order Date': [pd.Timestamp.today()],
                    'Ship Mode': [p_mode],
                    'Region': [p_region],
                    'State/Province': [p_state],
                    'FACTORY': [p_factory],
                    'Division': ['Unknown'],
                    'Sales': [p_sales],
                    'Units': [p_units],
                    'Cost': [p_sales * 0.6],
                    'lead_time_days': [0]
                })

                # Transform using saved encoders
                x_pred, _, _, _ = build_features(
                    pred_df, encoders=encoders, fit=False)

                # Predict
                prob = model.predict_proba(x_pred)[0][1]
                is_delayed = model.predict(x_pred)[0]

                if is_delayed == 1:
                    st.error(
                        f"⚠️ **High Risk of Delay** "
                        f"({prob*100:.1f}% probability)"
                    )
                    thresh = metrics.get('threshold', 1600)
                    st.markdown(
                        f"This configuration typically exceeds the historic "
                        f"{thresh:.0f}-day long-haul threshold."
                    )
                else:
                    st.success(
                        f"✅ **Likely Normal Timeframe** "
                        f"({(1-prob)*100:.1f}% probability)"
                    )
                    st.markdown(
                        "This configuration is expected to arrive within "
                        "standard expected timeframes."
                    )

                # Show dynamic charts instead of static metrics
                st.divider()
                st.markdown("#### Delay Risk Analysis")
                g_col1, g_col2 = st.columns(2)
                
                with g_col1:
                    st.markdown(
                        '<p style="text-align:center; font-weight:bold;'
                        'margin-bottom:-20px;">Delay Probability</p>',
                        unsafe_allow_html=True
                    )
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        number={'suffix': "%", 'font': {'size': 36}},
                        gauge={
                            'axis': {
                                'range': [0, 100], 'tickwidth': 1,
                                'tickcolor': "#475569"
                            },
                            'bar': {
                                'color': "#ef4444" if is_delayed else "#3b82f6"
                            },
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 0,
                            'steps': [
                                {
                                    'range': [0, 50],
                                    'color': "rgba(59, 130, 246, 0.15)"
                                },
                                {
                                    'range': [50, 100],
                                    'color': "rgba(239, 68, 68, 0.15)"
                                }
                            ],
                        }
                    ))
                    fig_gauge.update_layout(**CHART_LAYOUT)
                    fig_gauge.update_layout(
                        height=220, 
                        margin=dict(l=20, r=20, t=30, b=20)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with g_col2:
                    st.markdown(
                        '<p style="text-align:center; font-weight:bold;'
                        'margin-bottom:-10px;">Key Drivers (Model Global)</p>',
                        unsafe_allow_html=True
                    )
                    importances = metrics.get('feature_importances', [])
                    feat_names = metrics.get('feature_names', [])
                    if importances and feat_names:
                        clean_names = [n.replace('_enc', '') for n in feat_names]
                        fi_df = pd.DataFrame(
                            {'Feature': clean_names, 'Importance': importances}
                        ).sort_values('Importance').tail(5)

                        fig_fi = px.bar(
                            fi_df, x='Importance', y='Feature', orientation='h',
                            color='Importance',
                            color_continuous_scale=px.colors.sequential.Teal
                        )
                        fig_fi.update_layout(**CHART_LAYOUT)
                        fig_fi.update_layout(
                            height=220, 
                            margin=dict(l=10, r=10, t=20, b=10),
                            coloraxis_showscale=False,
                            xaxis_title="", yaxis_title=""
                        )
                        st.plotly_chart(fig_fi, use_container_width=True)
                    else:
                        st.info("Feature impact data not available.")
        else:
            st.info("Configure the order and click *Predict Delay Risk* to generate the AI analysis report.")

    except FileNotFoundError:
        st.warning(
            "Model not found. Run `python src/ml_model.py` first."
        )

# ──────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="text-align:center;color:#475569;font-size:0.8rem;">'
    '🍬 Nassau Candy Distributor · Shipping Route Efficiency Dashboard'
    ' · Built with Streamlit & Plotly'
    '</p>',
    unsafe_allow_html=True,
)
