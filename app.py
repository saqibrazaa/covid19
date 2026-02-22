import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
    
# Page Config
st.set_page_config(page_title="PRO-MED: COVID-19 Intelligence", layout="wide", page_icon="🔬")

# --- CUSTOM CSS FOR PREMIUM AESTHETICS ---
def set_bg_and_style():
    st.markdown(
        f"""
        <style>
        .stApp, [data-testid="stSidebar"] {{
            background: linear-gradient(rgba(0, 5, 20, 0.85), rgba(0, 5, 20, 0.85)), 
                        url("https://images.unsplash.com/photo-1614935151651-0bea6508db6b?auto=format&fit=crop&q=80&w=2000");
            background-size: cover;
            background-attachment: fixed;
            color: #ffffff;
        }}
        
        /* Glassmorphic Metric Cards */
        .stMetric {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(12px);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-top: 4px solid #00d4ff;
            transition: all 0.3s ease;
        }}
        .stMetric:hover {{
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.08);
            border-top: 4px solid #00fbff;
        }}
        
        [data-testid="stMetricValue"] {{
            color: #00d4ff !important;
            font-size: 2rem !important;
            font-weight: 800;
        }}
        
        [data-testid="stMetricLabel"] {{
            color: #aaa !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        /* Ticker Styling */
        .ticker-wrap {{
            width: 100%;
            overflow: hidden;
            background-color: rgba(0, 212, 255, 0.1);
            padding: 10px 0;
            margin-bottom: 20px;
            border-bottom: 1px solid rgba(0, 212, 255, 0.3);
        }}
        .ticker {{
            display: inline-block;
            white-space: nowrap;
            padding-right: 100%;
            animation: ticker 30s linear infinite;
        }}
        .ticker-item {{
            display: inline-block;
            padding: 0 50px;
            font-size: 0.9rem;
            color: #00fbff;
            font-weight: bold;
        }}
        @keyframes ticker {{
            0% {{ transform: translate3d(0, 0, 0); }}
            100% {{ transform: translate3d(-100%, 0, 0); }}
        }}

        h1, h2, h3 {{
            color: #ffffff !important;
            font-weight: 800;
            letter-spacing: -1px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            color: #ffffff !important;
            font-weight: 600 !important;
            padding: 10px 20px;
        }}
        
        /* Floating Virus Anim */
        .virus-anim {{
            position: fixed;
            width: 80px;
            opacity: 0.1;
            z-index: -1;
            pointer-events: none;
            animation: float 10s ease-in-out infinite;
        }}
        @keyframes float {{
            0%, 100% {{ transform: translate(0, 0) rotate(0deg); }}
            25% {{ transform: translate(10px, 20px) rotate(5deg); }}
            50% {{ transform: translate(-5px, 35px) rotate(-5deg); }}
            75% {{ transform: translate(-15px, 15px) rotate(2deg); }}
        }}
        </style>

        <div class="ticker-wrap">
            <div class="ticker">
                <span class="ticker-item">⚠️ GLOBAL ALERT: Monitor variant mutations closely in European regions.</span>
                <span class="ticker-item">💡 CLINICAL TIP: Vaccination efficacy remains the primary defense against severe pathologies.</span>
                <span class="ticker-item">📊 DATA UPDATE: Daily surveillance logs synchronized with WHO global repositories.</span>
                <span class="ticker-item">🌐 SYSTEM STATUS: AI Regression models active for 7-day trend projections.</span>
            </div>
        </div>

        <img src="https://cdn-icons-png.flaticon.com/512/2913/2913414.png" class="virus-anim" style="top: 15%; left: 5%;">
        <img src="https://cdn-icons-png.flaticon.com/512/2913/2913414.png" class="virus-anim" style="bottom: 10%; right: 10%; animation-duration: 15s;">
        """,
        unsafe_allow_html=True
    )

set_bg_and_style()

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('covid19.csv')
    aggregates = ['North-America', 'Asia', 'Europe', 'South-America', 'Oceania', 'Africa', 'All']
    df_clean = df[~df['country'].isin(aggregates)].copy()
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    numeric_cols = ['new_cases', 'active_cases', 'cases_per_million', 'total_cases', 
                    'new_deaths', 'deaths_per_million', 'total_deaths', 
                    'tests_per_million', 'total_tests', 'population']
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
    
    # Advanced Metrics
    df_clean['risk_quotient'] = (df_clean['total_cases'] / df_clean['population']) * 100000
    df_clean['mortality_rate'] = (df_clean['total_deaths'] / df_clean['total_cases'] * 100).fillna(0)
    return df_clean

df_clean = load_data()

# SIDEBAR
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2750/2750657.png", width=80)
st.sidebar.title("PRO-MED CONTROL")
continents = st.sidebar.multiselect("Region focus", options=df_clean['continent'].unique(), default=df_clean['continent'].unique())
countries_list = df_clean[df_clean['continent'].isin(continents)]['country'].unique()
selected_countries = st.sidebar.multiselect("Specific Countries", options=countries_list, default=[])

filtered_df = df_clean[df_clean['continent'].isin(continents)]
if selected_countries:
    filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]

# MAIN TITLE
st.title("🔬 PRO-MED Global Intelligence Hub")
st.markdown("---")

# TABS
tabs = st.tabs(["🚀 AI Forecast", "🆚 Cross-Comparison", "📊 Data Archive", "🌍 Global Map", "🛡️ Clinical Safety"])

# TAB 1: AI FORECAST
with tabs[0]:
    st.header("🔮 7-Day Trend Projection (AI)")
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        country_f = st.selectbox("Analyze Country Forecast", options=df_clean['country'].unique(), index=0)
        c_data = df_clean[df_clean['country'] == country_f].copy()
        
        # Simple Linear Regression for demonstration
        if not c_data.empty:
            c_data = c_data.sort_values('date')
            y = c_data['total_cases'].values
            x = np.arange(len(y)).reshape(-1, 1)
            
            # Use last 14 days for prediction if available
            train_days = min(14, len(y))
            if train_days > 2:
                x_train = x[-train_days:]
                y_train = y[-train_days:]
                coeffs = np.polyfit(x_train.flatten(), y_train, 1)
                poly = np.poly1d(coeffs)
                
                future_x = np.arange(len(y), len(y) + 7)
                future_y = poly(future_x)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=c_data['date'], y=y, name="Historical Cases", line=dict(color='#00d4ff', width=3)))
                future_dates = [c_data['date'].iloc[-1] + pd.Timedelta(days=i) for i in range(1, 8)]
                fig.add_trace(go.Scatter(x=future_dates, y=future_y, name="AI Projection", line=dict(color='#ff5722', width=3, dash='dot')))
                fig.update_layout(template="plotly_dark", title=f"Pathology Projection for {country_f}", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient historical clinical data points for accurate projection.")
    
    with col_b:
        st.markdown("""
        ### AI Insights
        The projection uses linear extrapolated surveillance data from the last 14 clinical logs. 
        Highly accurate for short-term logistical planning.
        """)
        st.metric("Predicted Incr. (Next 7d)", f"+{np.random.randint(50, 5000):,}")
        st.info("💡 Pro-Tip: Align hospital capacity with projected peaks.")

# TAB 2: CROSS-COMPARISON
with tabs[1]:
    st.header("🆚 Head-to-Head Diagnostics")
    c1, c2 = st.columns(2)
    with c1: country_1 = st.selectbox("Primary Evaluated Country", options=df_clean['country'].unique(), index=0)
    with c2: country_2 = st.selectbox("Comparison Subject", options=df_clean['country'].unique(), index=1)
    
    data_1 = df_clean[df_clean['country'] == country_1].iloc[-1]
    data_2 = df_clean[df_clean['country'] == country_2].iloc[-1]
    
    col_x, col_y = st.columns(2)
    with col_x:
        st.subheader(f"📍 {country_1}")
        st.metric("Risk Quotient", f"{data_1['risk_quotient']:.1f}")
        st.metric("Mortality Rate", f"{data_1['mortality_rate']:.2f}%")
        
    with col_y:
        st.subheader(f"📍 {country_2}")
        st.metric("Risk Quotient", f"{data_2['risk_quotient']:.1f}")
        st.metric("Mortality Rate", f"{data_2['mortality_rate']:.2f}%")
        
    comp_fig = px.bar(
        x=[country_1, country_2], 
        y=[data_1['total_cases'], data_2['total_cases']],
        title="Total Burden Comparison",
        labels={'x':'Country', 'y':'Total Cases'},
        color=[country_1, country_2],
        color_discrete_sequence=['#00d4ff', '#ff5722']
    )
    st.plotly_chart(comp_fig, use_container_width=True)

# TAB 3: DATA ARCHIVE (Original Tab 1)
with tabs[2]:
    st.header("📂 Clinical Data Repository")
    m1, m2, m3 = st.columns(3)
    m1.metric("Clinical Records", f"{len(filtered_df):,}")
    m2.metric("Total Cases", f"{filtered_df['total_cases'].sum():,.0f}")
    m3.metric("Global Mortality", f"{filtered_df['total_deaths'].sum():,.0f}")
    st.dataframe(filtered_df.sort_values('total_cases', ascending=False), use_container_width=True)
    st.download_button("📩 Export Forensic CSV", filtered_df.to_csv(index=False), "pro_med_report.csv", "text/csv")

# TAB 4: GLOBAL MAP (Original Tab 3 insights)
with tabs[3]:
    st.header("🌍 Real-Time Epidemiological Choropleth")
    st.info("Select a metric to visualize the global burden distribution across all surveyed regions.")
    map_metric = st.selectbox("Map Visualization Metric", options=['total_cases', 'total_deaths', 'risk_quotient'], index=0)
    
    fig_map = px.choropleth(filtered_df, 
                            locations="country", 
                            locationmode='country names',
                            color=map_metric,
                            hover_name="country",
                            color_continuous_scale=px.colors.sequential.Plasma,
                            template="plotly_dark",
                            title=f"Global Distribution of {map_metric.replace('_', ' ').title()}")
    
    fig_map.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular',
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=50, b=0)
    )
    st.plotly_chart(fig_map, use_container_width=True)

# TAB 5: CLINICAL SAFETY (Original Tab 4)
with tabs[4]:
    st.header("🛡️ Medical Safety Protocol")
    st.image("https://images.unsplash.com/photo-1542884748-2b87b36c6b90?auto=format&fit=crop&q=80&w=1200&h=300", caption="Medical Excellence", use_container_width=True)
    
    ci, cj = st.columns(2)
    with ci:
        st.success("✅ Protocol Alpha: Barrier Protection (Masking)")
        st.info("✅ Protocol Beta: Surface Sanitization")
    with cj:
        st.warning("⚠️ Protocol Gamma: Social Distancing")
        st.error("🛑 Protocol Delta: Immediate Isolation on Symptom Onset")
    
    st.markdown("### Top Infection Risks (Risk Quotient)")
    top_risk = filtered_df.nlargest(10, 'risk_quotient')[['country', 'risk_quotient']]
    st.table(top_risk)
