import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ============================================================================
# KONFIGURASI PAGE
# ============================================================================
st.set_page_config(
    page_title="Dashboard Analisis Minyak & BBM Indonesia",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv('/mnt/user-data/outputs/data_master_minyak_bbm.csv')
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    return df

df = load_data()

# ============================================================================
# HEADER
# ============================================================================
st.markdown('<h1 class="main-header">⛽ Dashboard Analisis Minyak & BBM Indonesia</h1>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# SIDEBAR - FILTER
# ============================================================================
st.sidebar.header("🎛️ Filter Data")

# Date range filter
min_date = df['Tanggal'].min().date()
max_date = df['Tanggal'].max().date()

date_range = st.sidebar.date_input(
    "Pilih Rentang Tanggal",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Year filter
years = sorted(df['Tahun'].unique(), reverse=True)
selected_years = st.sidebar.multiselect(
    "Pilih Tahun",
    options=years,
    default=years
)

# Commodity filter
commodities = st.sidebar.multiselect(
    "Pilih Komoditas",
    options=['Brent', 'WTI', 'Pertalite', 'Pertamax', 'Solar'],
    default=['Brent', 'Pertalite', 'Pertamax']
)

# Filter data
if len(date_range) == 2:
    df_filtered = df[
        (df['Tanggal'].dt.date >= date_range[0]) &
        (df['Tanggal'].dt.date <= date_range[1]) &
        (df['Tahun'].isin(selected_years))
    ]
else:
    df_filtered = df[df['Tahun'].isin(selected_years)]

st.sidebar.markdown("---")
st.sidebar.info(f"📊 Data: {len(df_filtered):,} hari\n\n📅 Periode: {df_filtered['Tanggal'].min().date()} - {df_filtered['Tanggal'].max().date()}")

# ============================================================================
# TAB NAVIGATION
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 Analisis Trend",
    "🔗 Korelasi",
    "🔮 Prediksi",
    "💰 Business Intelligence"
])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================
with tab1:
    st.header("📊 Overview & Harga Terkini")
    
    # Metrics terkini
    latest = df.iloc[-1]
    prev = df.iloc[-30]  # 30 hari yang lalu
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        brent_change = ((latest['Harga_Brent_USD'] - prev['Harga_Brent_USD']) / prev['Harga_Brent_USD'] * 100)
        st.metric(
            "Brent Oil",
            f"${latest['Harga_Brent_USD']:.2f}/bbl",
            f"{brent_change:+.2f}% (30d)"
        )
    
    with col2:
        wti_change = ((latest['Harga_WTI_USD'] - prev['Harga_WTI_USD']) / prev['Harga_WTI_USD'] * 100)
        st.metric(
            "WTI Oil",
            f"${latest['Harga_WTI_USD']:.2f}/bbl",
            f"{wti_change:+.2f}% (30d)"
        )
    
    with col3:
        kurs_change = ((latest['Kurs_USD_IDR'] - prev['Kurs_USD_IDR']) / prev['Kurs_USD_IDR'] * 100)
        st.metric(
            "Kurs USD/IDR",
            f"Rp {latest['Kurs_USD_IDR']:,.0f}",
            f"{kurs_change:+.2f}% (30d)"
        )
    
    with col4:
        st.metric(
            "Pertalite",
            f"Rp {latest['Pertalite_Rp']:,.0f}/L",
            "Subsidi"
        )
    
    with col5:
        st.metric(
            "Pertamax",
            f"Rp {latest['Pertamax_Rp']:,.0f}/L",
            "Non-Subsidi"
        )
    
    st.markdown("---")
    
    # Chart harga historis
    st.subheader("📈 Pergerakan Harga Historis")
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Harga Minyak Mentah (USD/barrel)', 'Kurs USD/IDR', 'Harga BBM Indonesia (Rp/liter)'),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Minyak Mentah
    if 'Brent' in commodities:
        fig.add_trace(
            go.Scatter(x=df_filtered['Tanggal'], y=df_filtered['Harga_Brent_USD'],
                      name='Brent', line=dict(color='#FF6B6B', width=2)),
            row=1, col=1
        )
    if 'WTI' in commodities:
        fig.add_trace(
            go.Scatter(x=df_filtered['Tanggal'], y=df_filtered['Harga_WTI_USD'],
                      name='WTI', line=dict(color='#4ECDC4', width=2)),
            row=1, col=1
        )
    
    # Kurs
    fig.add_trace(
        go.Scatter(x=df_filtered['Tanggal'], y=df_filtered['Kurs_USD_IDR'],
                  name='Kurs', line=dict(color='#95E1D3', width=2), showlegend=False),
        row=2, col=1
    )
    
    # BBM
    if 'Pertalite' in commodities:
        fig.add_trace(
            go.Scatter(x=df_filtered['Tanggal'], y=df_filtered['Pertalite_Rp'],
                      name='Pertalite', line=dict(color='#F38181', width=2)),
            row=3, col=1
        )
    if 'Pertamax' in commodities:
        fig.add_trace(
            go.Scatter(x=df_filtered['Tanggal'], y=df_filtered['Pertamax_Rp'],
                      name='Pertamax', line=dict(color='#AA96DA', width=2)),
            row=3, col=1
        )
    if 'Solar' in commodities:
        fig.add_trace(
            go.Scatter(x=df_filtered['Tanggal'], y=df_filtered['Solar_Rp'],
                      name='Solar', line=dict(color='#FCBAD3', width=2)),
            row=3, col=1
        )
    
    fig.update_xaxes(title_text="Tanggal", row=3, col=1)
    fig.update_yaxes(title_text="USD/barrel", row=1, col=1)
    fig.update_yaxes(title_text="Rupiah", row=2, col=1)
    fig.update_yaxes(title_text="Rp/liter", row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistik Ringkasan
    st.subheader("📊 Statistik Ringkasan (Periode Terpilih)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stats_df = pd.DataFrame({
            'Komoditas': ['Brent Oil', 'WTI Oil', 'Kurs USD/IDR'],
            'Min': [
                f"${df_filtered['Harga_Brent_USD'].min():.2f}",
                f"${df_filtered['Harga_WTI_USD'].min():.2f}",
                f"Rp {df_filtered['Kurs_USD_IDR'].min():,.0f}"
            ],
            'Max': [
                f"${df_filtered['Harga_Brent_USD'].max():.2f}",
                f"${df_filtered['Harga_WTI_USD'].max():.2f}",
                f"Rp {df_filtered['Kurs_USD_IDR'].max():,.0f}"
            ],
            'Rata-rata': [
                f"${df_filtered['Harga_Brent_USD'].mean():.2f}",
                f"${df_filtered['Harga_WTI_USD'].mean():.2f}",
                f"Rp {df_filtered['Kurs_USD_IDR'].mean():,.0f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        stats_bbm = pd.DataFrame({
            'BBM': ['Pertalite', 'Pertamax', 'Solar'],
            'Min': [
                f"Rp {df_filtered['Pertalite_Rp'].min():,.0f}",
                f"Rp {df_filtered['Pertamax_Rp'].min():,.0f}",
                f"Rp {df_filtered['Solar_Rp'].min():,.0f}"
            ],
            'Max': [
                f"Rp {df_filtered['Pertalite_Rp'].max():,.0f}",
                f"Rp {df_filtered['Pertamax_Rp'].max():,.0f}",
                f"Rp {df_filtered['Solar_Rp'].max():,.0f}"
            ],
            'Rata-rata': [
                f"Rp {df_filtered['Pertalite_Rp'].mean():,.0f}",
                f"Rp {df_filtered['Pertamax_Rp'].mean():,.0f}",
                f"Rp {df_filtered['Solar_Rp'].mean():,.0f}"
            ]
        })
        st.dataframe(stats_bbm, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 2: ANALISIS TREND
# ============================================================================
with tab2:
    st.header("📈 Analisis Trend & Volatilitas")
    
    # Pilihan analisis
    analysis_type = st.selectbox(
        "Pilih Jenis Analisis",
        ["Perubahan Harian (%)", "Moving Average", "Volatilitas", "Spread Brent-WTI"]
    )
    
    if analysis_type == "Perubahan Harian (%)":
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_filtered['Tanggal'],
            y=df_filtered['Brent_Change_Pct'],
            name='Brent',
            line=dict(color='#FF6B6B', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=df_filtered['Tanggal'],
            y=df_filtered['Pertalite_Change_Pct'],
            name='Pertalite',
            line=dict(color='#4ECDC4', width=1)
        ))
        
        fig.update_layout(
            title="Perubahan Harga Harian (%)",
            xaxis_title="Tanggal",
            yaxis_title="Perubahan (%)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_type == "Moving Average":
        window = st.slider("Pilih Window (hari)", 7, 90, 30)
        
        df_filtered['Brent_MA'] = df_filtered['Harga_Brent_USD'].rolling(window=window).mean()
        df_filtered['Pertamax_MA'] = df_filtered['Pertamax_Rp'].rolling(window=window).mean()
        
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Brent Oil', 'Pertamax'))
        
        fig.add_trace(go.Scatter(x=df_filtered['Tanggal'], y=df_filtered['Harga_Brent_USD'],
                                name='Brent Actual', line=dict(color='lightgray', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_filtered['Tanggal'], y=df_filtered['Brent_MA'],
                                name=f'MA-{window}', line=dict(color='#FF6B6B', width=2)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df_filtered['Tanggal'], y=df_filtered['Pertamax_Rp'],
                                name='Pertamax Actual', line=dict(color='lightgray', width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_filtered['Tanggal'], y=df_filtered['Pertamax_MA'],
                                name=f'MA-{window}', line=dict(color='#AA96DA', width=2)), row=2, col=1)
        
        fig.update_layout(height=600, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_type == "Volatilitas":
        window = st.slider("Pilih Window (hari)", 7, 90, 30)
        
        df_filtered['Brent_Vol'] = df_filtered['Brent_Change_Pct'].rolling(window=window).std()
        df_filtered['Kurs_Vol'] = df_filtered['Kurs_Change_Pct'].rolling(window=window).std()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=df_filtered['Tanggal'], y=df_filtered['Brent_Vol'],
                                name='Volatilitas Brent', fill='tozeroy', line=dict(color='#FF6B6B')))
        fig.add_trace(go.Scatter(x=df_filtered['Tanggal'], y=df_filtered['Kurs_Vol'],
                                name='Volatilitas Kurs', fill='tozeroy', line=dict(color='#95E1D3')))
        
        fig.update_layout(title=f"Volatilitas (Rolling Std {window} hari)",
                         xaxis_title="Tanggal", yaxis_title="Std Deviasi (%)",
                         height=500, hovermode='x unified')
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # Spread Brent-WTI
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_filtered['Tanggal'],
            y=df_filtered['Spread_Brent_WTI'],
            fill='tozeroy',
            name='Spread Brent-WTI',
            line=dict(color='#667eea')
        ))
        
        fig.add_hline(y=df_filtered['Spread_Brent_WTI'].mean(),
                     line_dash="dash", line_color="red",
                     annotation_text=f"Rata-rata: ${df_filtered['Spread_Brent_WTI'].mean():.2f}")
        
        fig.update_layout(
            title="Spread Harga Brent vs WTI",
            xaxis_title="Tanggal",
            yaxis_title="Spread (USD/barrel)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Event Timeline
    st.subheader("🗓️ Event Penting")
    events_df = df[df['Event'] != ''].copy()
    if len(events_df) > 0:
        events_display = events_df[['Tanggal', 'Event', 'Harga_Brent_USD', 'Pertalite_Rp']].copy()
        events_display['Tanggal'] = events_display['Tanggal'].dt.date
        events_display.columns = ['Tanggal', 'Event', 'Brent (USD)', 'Pertalite (Rp)']
        st.dataframe(events_display, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 3: KORELASI
# ============================================================================
with tab3:
    st.header("🔗 Analisis Korelasi")
    
    # Correlation matrix
    st.subheader("📊 Matriks Korelasi")
    
    corr_cols = ['Harga_Brent_USD', 'Harga_WTI_USD', 'Kurs_USD_IDR', 
                 'Pertalite_Rp', 'Pertamax_Rp', 'Solar_Rp']
    corr_matrix = df_filtered[corr_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=['Brent', 'WTI', 'Kurs', 'Pertalite', 'Pertamax', 'Solar'],
        y=['Brent', 'WTI', 'Kurs', 'Pertalite', 'Pertamax', 'Solar'],
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Korelasi")
    ))
    
    fig.update_layout(
        title="Korelasi Antar Variabel",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot
    st.subheader("📈 Scatter Plot Korelasi")
    
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Pilih Variabel X", 
                            ['Harga_Brent_USD', 'Harga_WTI_USD', 'Kurs_USD_IDR'],
                            index=0)
    with col2:
        y_var = st.selectbox("Pilih Variabel Y",
                            ['Pertalite_Rp', 'Pertamax_Rp', 'Solar_Rp'],
                            index=1)
    
    fig = px.scatter(df_filtered, x=x_var, y=y_var,
                     trendline="ols",
                     color='Tahun',
                     title=f"Korelasi {x_var} vs {y_var}",
                     labels={x_var: x_var.replace('_', ' '), y_var: y_var.replace('_', ' ')})
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Lag Analysis
    st.subheader("⏱️ Analisis Lag Time")
    st.info("📌 Analisis berapa hari delay antara perubahan harga minyak dengan perubahan BBM")
    
    max_lag = st.slider("Max Lag (hari)", 1, 30, 14)
    
    correlations = []
    for lag in range(max_lag + 1):
        if lag == 0:
            corr = df_filtered['Harga_Brent_USD'].corr(df_filtered['Pertamax_Rp'])
        else:
            corr = df_filtered['Harga_Brent_USD'].iloc[:-lag].corr(df_filtered['Pertamax_Rp'].iloc[lag:])
        correlations.append({'Lag (hari)': lag, 'Korelasi': corr})
    
    lag_df = pd.DataFrame(correlations)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=lag_df['Lag (hari)'], y=lag_df['Korelasi'],
                        marker_color='#667eea'))
    
    fig.update_layout(
        title="Korelasi Brent Oil vs Pertamax dengan Berbagai Lag",
        xaxis_title="Lag (hari)",
        yaxis_title="Korelasi",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    best_lag = lag_df.loc[lag_df['Korelasi'].idxmax(), 'Lag (hari)']
    st.success(f"✨ **Optimal Lag**: {int(best_lag)} hari (Korelasi: {lag_df['Korelasi'].max():.3f})")

# ============================================================================
# TAB 4: PREDIKSI
# ============================================================================
with tab4:
    st.header("🔮 Prediksi Harga")
    
    st.info("📌 Model prediksi menggunakan Linear Regression dengan data historis")
    
    # Pilih komoditas untuk prediksi
    predict_commodity = st.selectbox(
        "Pilih Komoditas untuk Prediksi",
        ['Brent Oil', 'Pertamax', 'Pertalite']
    )
    
    forecast_days = st.slider("Jumlah Hari Prediksi", 7, 90, 30)
    
    # Prepare data
    if predict_commodity == 'Brent Oil':
        target_col = 'Harga_Brent_USD'
        title = 'Prediksi Harga Brent Oil'
    elif predict_commodity == 'Pertamax':
        target_col = 'Pertamax_Rp'
        title = 'Prediksi Harga Pertamax'
    else:
        target_col = 'Pertalite_Rp'
        title = 'Prediksi Harga Pertalite'
    
    # Model training
    df_model = df_filtered.copy()
    df_model['Days'] = (df_model['Tanggal'] - df_model['Tanggal'].min()).dt.days
    
    X = df_model[['Days']].values
    y = df_model[target_col].values
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions
    last_day = df_model['Days'].max()
    future_days = np.array([[last_day + i] for i in range(1, forecast_days + 1)])
    predictions = model.predict(future_days)
    
    # Create future dates
    last_date = df_model['Tanggal'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    
    # Plot
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df_model['Tanggal'],
        y=df_model[target_col],
        name='Data Historis',
        line=dict(color='#4ECDC4', width=2)
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        name='Prediksi',
        line=dict(color='#FF6B6B', width=2, dash='dash')
    ))
    
    # Confidence interval (simplified)
    std_error = np.std(y - model.predict(X))
    upper_bound = predictions + 2 * std_error
    lower_bound = predictions - 2 * std_error
    
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=list(upper_bound) + list(lower_bound)[::-1],
        fill='toself',
        fillcolor='rgba(255, 107, 107, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval',
        showlegend=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Tanggal",
        yaxis_title="Harga",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction table
    st.subheader("📊 Tabel Prediksi")
    
    pred_df = pd.DataFrame({
        'Tanggal': [d.date() for d in future_dates],
        'Prediksi': predictions.round(2),
        'Lower Bound': lower_bound.round(2),
        'Upper Bound': upper_bound.round(2)
    })
    
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
    
    # Model metrics
    from sklearn.metrics import r2_score, mean_absolute_error
    
    y_pred_train = model.predict(X)
    r2 = r2_score(y, y_pred_train)
    mae = mean_absolute_error(y, y_pred_train)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("Mean Absolute Error", f"{mae:.2f}")

# ============================================================================
# TAB 5: BUSINESS INTELLIGENCE
# ============================================================================
with tab5:
    st.header("💰 Business Intelligence & Insights")
    
    # Subsidi Analysis
    st.subheader("💵 Analisis Subsidi Pertalite")
    
    col1, col2 = st.columns(2)
    
    with col1:
        avg_subsidy = df_filtered['Estimasi_Subsidi_Pertalite_Rp'].mean()
        total_subsidy = df_filtered['Estimasi_Subsidi_Pertalite_Rp'].sum()
        max_subsidy = df_filtered['Estimasi_Subsidi_Pertalite_Rp'].max()
        
        st.metric("Rata-rata Subsidi per Liter", f"Rp {avg_subsidy:,.0f}")
        st.metric("Total Subsidi (Estimasi)", f"Rp {total_subsidy:,.0f}")
        st.metric("Subsidi Tertinggi", f"Rp {max_subsidy:,.0f}")
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_filtered['Tanggal'],
            y=df_filtered['Estimasi_Subsidi_Pertalite_Rp'],
            fill='tozeroy',
            name='Subsidi',
            line=dict(color='#FF6B6B')
        ))
        
        fig.update_layout(
            title="Estimasi Subsidi Pertalite per Liter",
            xaxis_title="Tanggal",
            yaxis_title="Subsidi (Rp)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Margin Analysis
    st.subheader("📊 Analisis Margin Pertamax")
    
    col1, col2 = st.columns(2)
    
    with col1:
        avg_margin = df_filtered['Estimasi_Margin_Pertamax_USD'].mean()
        margin_pct = (avg_margin / (df_filtered['Harga_Brent_USD'] / 159).mean() * 100)
        
        st.metric("Rata-rata Margin per Liter", f"${avg_margin:.3f}")
        st.metric("Margin (%)", f"{margin_pct:.1f}%")
        
        # Yearly comparison
        yearly_margin = df_filtered.groupby('Tahun')['Estimasi_Margin_Pertamax_USD'].mean().reset_index()
        yearly_margin['Estimasi_Margin_Pertamax_USD'] = yearly_margin['Estimasi_Margin_Pertamax_USD'].round(3)
        
        st.dataframe(yearly_margin, use_container_width=True, hide_index=True, 
                    column_config={
                        'Tahun': 'Tahun',
                        'Estimasi_Margin_Pertamax_USD': 'Avg Margin (USD)'
                    })
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_filtered['Tanggal'],
            y=df_filtered['Estimasi_Margin_Pertamax_USD'],
            name='Margin',
            line=dict(color='#AA96DA', width=2)
        ))
        
        fig.update_layout(
            title="Estimasi Margin Pertamax per Liter",
            xaxis_title="Tanggal",
            yaxis_title="Margin (USD)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Price Impact Analysis
    st.subheader("📈 Analisis Dampak Harga Minyak terhadap BBM")
    
    # Calculate price elasticity
    brent_change = df_filtered['Harga_Brent_USD'].pct_change() * 100
    pertamax_change = df_filtered['Pertamax_Rp'].pct_change() * 100
    
    # Remove outliers and NaN
    valid_idx = (~brent_change.isna()) & (~pertamax_change.isna()) & (abs(brent_change) < 50) & (abs(pertamax_change) < 50)
    
    if valid_idx.sum() > 0:
        elasticity = (pertamax_change[valid_idx] / brent_change[valid_idx]).mean()
        
        st.info(f"""
        📌 **Elastisitas Harga Pertamax terhadap Brent**: {elasticity:.2f}
        
        Artinya: Ketika harga Brent naik 1%, harga Pertamax cenderung naik {elasticity:.2f}%
        """)
    
    # Recommendations
    st.subheader("💡 Rekomendasi Strategis")
    
    latest_data = df.iloc[-1]
    
    recommendations = []
    
    # Check if Brent is trending up or down
    recent_brent = df.iloc[-30:]['Harga_Brent_USD']
    if recent_brent.iloc[-1] > recent_brent.mean():
        recommendations.append("⚠️ **Harga minyak sedang di atas rata-rata 30 hari** - Pertimbangkan hedging untuk mitigasi risiko")
    else:
        recommendations.append("✅ **Harga minyak di bawah rata-rata 30 hari** - Kondisi favorable untuk procurement")
    
    # Check subsidy level
    if latest_data['Estimasi_Subsidi_Pertalite_Rp'] > df['Estimasi_Subsidi_Pertalite_Rp'].quantile(0.75):
        recommendations.append("⚠️ **Beban subsidi tinggi** - Evaluasi kebijakan pricing Pertalite")
    
    # Check margin
    if latest_data['Estimasi_Margin_Pertamax_USD'] < df['Estimasi_Margin_Pertamax_USD'].quantile(0.25):
        recommendations.append("📉 **Margin Pertamax rendah** - Pertimbangkan penyesuaian harga atau efisiensi operasional")
    else:
        recommendations.append("✅ **Margin Pertamax sehat** - Maintain current pricing strategy")
    
    # Display recommendations
    for rec in recommendations:
        st.markdown(f"- {rec}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Dashboard Analisis Minyak & BBM Indonesia</strong></p>
    <p>Data Sources: FRED (Brent & WTI), Bank Indonesia (Kurs), Pertamina (BBM)</p>
    <p>Built with ❤️ for Pertamina Portfolio Project</p>
</div>
""", unsafe_allow_html=True)
