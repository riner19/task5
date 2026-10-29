import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from fpdf import FPDF
import tempfile
import os

SHEET_ID = "1p9xJTjMGfxcheSrRLJqPu3nXNrJWqDBF8FqjwQrBz0M"

@st.cache_data(ttl=60)
def load_data():
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
    raw_df = pd.read_csv(url)
    
    mine_names = raw_df.iloc[2, 1:].dropna().values.tolist()
    data_df = raw_df.iloc[11:, 8:(9 + len(mine_names))].copy()
    data_df.columns = ['Date'] + mine_names
    
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    data_df.set_index('Date', inplace=True)
    for col in mine_names:
        data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
    data_df = data_df.dropna(how='all')
    
    return data_df, mine_names

def detect_iqr_outliers(series, multiplier):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (multiplier * IQR)
    upper_bound = Q3 + (multiplier * IQR)
    return (series < lower_bound) | (series > upper_bound)

def detect_zscore_outliers(series, threshold):
    z_scores = np.abs(stats.zscore(series.dropna()))
    outliers = pd.Series(False, index=series.index)
    outliers.loc[series.dropna().index] = z_scores > threshold
    return outliers

def detect_ma_outliers(series, window, threshold_pct):
    ma = series.rolling(window=window, center=True).mean()
    ma = ma.fillna(series) 
    pct_diff = np.abs((series - ma) / ma) * 100
    return pct_diff > threshold_pct

def grubbs_stat(y):
    std_dev = np.std(y)
    avg_y = np.mean(y)
    abs_val_minus_avg = abs(y - avg_y)
    max_of_deviations = max(abs_val_minus_avg)
    max_ind = np.argmax(abs_val_minus_avg)
    Gcal = max_of_deviations / std_dev
    return Gcal, max_ind

def calculate_critical_value(size, alpha):
    t_dist = stats.t.ppf(1 - alpha / (2 * size), size - 2)
    numerator = (size - 1) * np.sqrt(np.square(t_dist))
    denominator = np.sqrt(size) * np.sqrt(size - 2 + np.square(t_dist))
    critical_value = numerator / denominator
    return critical_value

def detect_grubbs_outliers(series, alpha):
    outliers = pd.Series(False, index=series.index)
    y = series.dropna().values.copy()
    indices = series.dropna().index.copy()
    
    while len(y) > 2:
        Gcal, max_ind = grubbs_stat(y)
        Gcrit = calculate_critical_value(len(y), alpha)
        if Gcal > Gcrit:
            outlier_idx = indices[max_ind]
            outliers[outlier_idx] = True
            y = np.delete(y, max_ind)
            indices = np.delete(indices, max_ind)
        else:
            break
    return outliers


st.set_page_config(page_title="Weyland-Yutani Dashboard", layout="wide")
st.title("Weyland-Yutani Mining Operations Dashboard")

try:
    df, mine_names = load_data()
    df['Total Output'] = df.sum(axis=1)
    analysis_columns = mine_names + ['Total Output']

    st.sidebar.header("Dashboard Controls")
    st.sidebar.subheader("Visualization")
    chart_type = st.sidebar.selectbox("Chart Type", ["Line", "Bar", "Stacked Area"])
    trend_degree = st.sidebar.selectbox("Trendline Degree (Polynomial)", [1, 2, 3, 4])
    
    st.sidebar.subheader("Anomaly Parameters")
    iqr_multiplier = st.sidebar.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1)
    z_thresh = st.sidebar.slider("Z-Score Threshold", 1.0, 4.0, 3.0, 0.1)
    ma_window = st.sidebar.slider("Moving Avg Window (Days)", 2, 14, 7)
    ma_dist = st.sidebar.slider("MA Distance Threshold (%)", 5, 50, 20)
    grubbs_alpha = st.sidebar.slider("Grubbs' Test Alpha", 0.01, 0.10, 0.05, 0.01)

    st.header("Production Statistics")
    stats_df = pd.DataFrame({
        'Mean Daily Output': df.mean(),
        'Standard Deviation': df.std(),
        'Median': df.median(),
        'Interquartile Range (IQR)': df.apply(lambda x: x.quantile(0.75) - x.quantile(0.25))
    }).T
    st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)

    st.header("Production Visualization & Anomalies")
    selected_series = st.selectbox("Select Data to Visualize", analysis_columns)
    data_series = df[selected_series]
    
    outliers_iqr = detect_iqr_outliers(data_series, iqr_multiplier)
    outliers_z = detect_zscore_outliers(data_series, z_thresh)
    outliers_ma = detect_ma_outliers(data_series, ma_window, ma_dist)
    outliers_grubbs = detect_grubbs_outliers(data_series, grubbs_alpha)
    
    combined_outliers = outliers_iqr | outliers_z | outliers_ma | outliers_grubbs
    
    fig = go.Figure()
    if chart_type == "Line":
        fig.add_trace(go.Scatter(x=df.index, y=data_series, mode='lines', name=selected_series, line=dict(color='blue')))
    elif chart_type == "Bar":
        fig.add_trace(go.Bar(x=df.index, y=data_series, name=selected_series, marker_color='blue'))
    elif chart_type == "Stacked Area":
        fig.add_trace(go.Scatter(x=df.index, y=data_series, fill='tozeroy', mode='none', name=selected_series, fillcolor='rgba(0,0,255,0.3)'))

    x_num = np.arange(len(df.index))
    mask = ~data_series.isna()
    if mask.sum() > trend_degree:
        z = np.polyfit(x_num[mask], data_series[mask], trend_degree)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(x=df.index, y=p(x_num), mode='lines', name=f'Trend (Deg {trend_degree})', line=dict(color='orange', dash='dash')))
        
    if combined_outliers.any():
        fig.add_trace(go.Scatter(
            x=df.index[combined_outliers], 
            y=data_series[combined_outliers], 
            mode='markers', 
            name='Detected Outlier',
            marker=dict(color='red', size=10, symbol='x')
        ))

    fig.update_layout(title=f"{selected_series} Production Over Time", xaxis_title="Date", yaxis_title="Output")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Anomaly Breakdown: {selected_series}")
    if combined_outliers.any():
        anomaly_df = pd.DataFrame(index=df.index[combined_outliers])
        anomaly_df['Value'] = data_series[combined_outliers]
        anomaly_df['Failed IQR'] = outliers_iqr[combined_outliers]
        anomaly_df['Failed Z-Score'] = outliers_z[combined_outliers]
        anomaly_df['Failed MA Dist'] = outliers_ma[combined_outliers]
        anomaly_df['Failed Grubbs'] = outliers_grubbs[combined_outliers]
        st.dataframe(anomaly_df)
    else:
        st.info("No anomalies detected based on current parameters.")

    st.markdown("---")
    st.subheader("Report Export")
    
    if st.button("Generate Detailed PDF Report"):
        with st.spinner("Compiling report..."):
            pdf = FPDF()
            pdf.add_page()
            
            pdf.set_font("helvetica", "B", 16)
            pdf.cell(0, 10, f"Mining Operations Report: {selected_series}", ln=True, align="C")
            pdf.ln(5)
            
            pdf.set_font("helvetica", "B", 12)
            pdf.cell(0, 10, "1. Production Statistics", ln=True)
            pdf.set_font("helvetica", "", 10)
            stat_text = (
                f"Mean Daily Output: {stats_df.loc['Mean Daily Output', selected_series]:.2f}\n"
                f"Standard Deviation: {stats_df.loc['Standard Deviation', selected_series]:.2f}\n"
                f"Median: {stats_df.loc['Median', selected_series]:.2f}\n"
                f"IQR: {stats_df.loc['Interquartile Range (IQR)', selected_series]:.2f}"
            )
            pdf.multi_cell(0, 8, stat_text)
            pdf.ln(5)
            
            pdf.set_font("helvetica", "B", 12)
            pdf.cell(0, 10, "2. Visual Analysis & Trend", ln=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                fig.write_image(tmpfile.name, width=800, height=400)
                pdf.image(tmpfile.name, x=10, w=190)
                tmpfile_path = tmpfile.name
            pdf.ln(5)
            
            pdf.set_font("helvetica", "B", 12)
            pdf.cell(0, 10, "3. Detected Anomalies Breakdown", ln=True)
            pdf.set_font("helvetica", "", 10)
            
            if combined_outliers.any():
                mean_val = stats_df.loc['Mean Daily Output', selected_series]
                for idx, row in anomaly_df.iterrows():
                    date_str = idx.strftime('%Y-%m-%d')
                    val = row['Value']
                    direction = "Spike" if val > mean_val else "Drop"
                    
                    tests_failed = []
                    if row['Failed IQR']: tests_failed.append("IQR")
                    if row['Failed Z-Score']: tests_failed.append("Z-Score")
                    if row['Failed MA Dist']: tests_failed.append("Moving Avg")
                    if row['Failed Grubbs']: tests_failed.append("Grubbs")
                    
                    pdf.set_font("helvetica", "B", 10)
                    pdf.cell(0, 6, f"Date: {date_str} | Type: {direction} | Value: {val:.2f}", ln=True)
                    pdf.set_font("helvetica", "", 10)
                    pdf.cell(0, 6, f"   Tests Flagged: {', '.join(tests_failed)}", ln=True)
                    pdf.ln(2)
            else:
                pdf.cell(0, 10, "No anomalies detected within the current parameters.", ln=True)
            
            pdf_bytes = bytes(pdf.output())
            os.remove(tmpfile_path)
            
        st.success("Report generated successfully!")
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=f"Weyland_Yutani_Report_{selected_series}.pdf",
            mime="application/pdf"
        )

except Exception as e:
    st.error(f"Data connection failed. Error details: {e}")