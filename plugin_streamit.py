import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict

st.set_page_config(layout="wide", page_title="Oracle RMAN Backup Sizing")

def calculate_backup_sizes(
    db_size_gb: float,
    annual_growth_rate: float,
    log_gen_gb_per_day: float,
    daily_retention: int,
    weekly_retention: int,
    monthly_retention: int,
    yearly_retention: int,
    growth_projection_years: int = 3,
    compression_ratio: float = 0.5,
    incremental_change_rate: float = 0.1,
    use_compound_growth: bool = True
) -> Dict[str, float]:
    requirements: Dict[str, float] = {}
    requirements['full_backup_gb'] = db_size_gb * compression_ratio
    requirements['incremental_backup_gb'] = db_size_gb * incremental_change_rate * compression_ratio
    requirements['daily_incrementals_gb'] = daily_retention * requirements['incremental_backup_gb']
    requirements['weekly_fulls_gb'] = weekly_retention * requirements['full_backup_gb']
    requirements['monthly_fulls_gb'] = monthly_retention * requirements['full_backup_gb']
    requirements['yearly_fulls_gb'] = yearly_retention * requirements['full_backup_gb']

    max_log_days: int = max(
        daily_retention,
        weekly_retention * 7,
        monthly_retention * 30,
        yearly_retention * 365
    )
    requirements['logs_per_day_gb'] = log_gen_gb_per_day * compression_ratio
    requirements['total_logs_gb'] = max_log_days * requirements['logs_per_day_gb']

    requirements['total_backup_gb'] = (
        requirements['daily_incrementals_gb']
        + requirements['weekly_fulls_gb']
        + requirements['monthly_fulls_gb']
        + requirements['yearly_fulls_gb']
    )
    requirements['total_storage_gb'] = requirements['total_backup_gb'] + requirements['total_logs_gb']

    if annual_growth_rate > 0 and growth_projection_years > 0:
        if use_compound_growth:
            growth_factor: float = (1 + annual_growth_rate) ** growth_projection_years
        else:
            growth_factor = 1 + annual_growth_rate * growth_projection_years
        requirements['projected_storage_gb'] = requirements['total_storage_gb'] * growth_factor
        requirements['growth_years'] = growth_projection_years
        requirements['growth_factor'] = growth_factor
    else:
        requirements['projected_storage_gb'] = requirements['total_storage_gb']
        requirements['growth_years'] = 0
        requirements['growth_factor'] = 1.0

    return requirements

def generate_results_dataframe(
    db_size_gb: float,
    annual_growth_rate: float,
    log_gen_gb_per_day: float,
    daily_retention: int,
    weekly_retention: int,
    monthly_retention: int,
    yearly_retention: int,
    compression_ratio: float,
    incremental_change_rate: float,
    growth_projection_years: int,
    use_compound_growth: bool
) -> pd.DataFrame:
    months: int = growth_projection_years * 12
    time: np.ndarray = np.arange(1, months + 1)
    monthly_growth_rate: float = (1 + annual_growth_rate) ** (1/12) - 1
    max_log_days: int = max(
        daily_retention,
        weekly_retention * 7,
        monthly_retention * 30,
        yearly_retention * 365
    )

    if use_compound_growth:
        db_sizes: np.ndarray = db_size_gb * ((1 + monthly_growth_rate) ** time)
    else:
        db_sizes: np.ndarray = db_size_gb + (db_size_gb * annual_growth_rate) * (time / 12)

    full_backup_sizes: np.ndarray = db_sizes * compression_ratio
    incremental_backup_sizes: np.ndarray = db_sizes * incremental_change_rate * compression_ratio

    weekly_fulls: np.ndarray = weekly_retention * full_backup_sizes
    monthly_fulls: np.ndarray = monthly_retention * full_backup_sizes
    yearly_fulls: np.ndarray = yearly_retention * full_backup_sizes

    incremental_retention = math.floor(daily_retention / 7) * 6
    daily_incrementals: np.ndarray = incremental_retention * incremental_backup_sizes
    
    logs_growth: np.ndarray = log_gen_gb_per_day * compression_ratio * max_log_days
    if use_compound_growth:
        logs_growth = logs_growth * ((1 + monthly_growth_rate) ** time)
    else:
        logs_growth = logs_growth + (logs_growth * annual_growth_rate) * (time / 12)

    total_capacity: np.ndarray = (
        weekly_fulls + monthly_fulls + yearly_fulls + daily_incrementals + logs_growth
    )

    df: pd.DataFrame = pd.DataFrame({
        'Month': time,
        'Weekly Full Backups': weekly_fulls,
        'Monthly Full Backups': monthly_fulls,
        'Yearly Full Backups': yearly_fulls,
        'Daily Incrementals': daily_incrementals,
        'Logs': logs_growth,
        'Total Capacity': total_capacity
    })
    return df

# Sidebar for inputs
st.sidebar.title("Oracle RMAN Backup Sizing")

db_size_gb: float = st.sidebar.number_input("Database Size (GB)", value=1000, min_value=1)
annual_growth_rate: float = st.sidebar.slider("Annual Growth Rate", 0, 100, 20, 1) / 100
compression_ratio: float = ( 1 - (st.sidebar.slider("Compression Ratio", 0, 99, 50, 1) / 100)) 
incremental_change_rate: float = st.sidebar.slider("Incremental Change Rate", 0, 99, 50, 1) / 100
growth_projection_years: int = st.sidebar.slider("Growth Projection (years)", 1, 10, 3)
log_gen_gb_per_day: float = st.sidebar.number_input("Log Generation per Day (GB)", value=50, min_value=1)
daily_retention: int = st.sidebar.number_input("Daily Retention (days)", value=30, min_value=0)
weekly_retention: int = st.sidebar.number_input("Weekly Retention (weeks)", value=13, min_value=0)
monthly_retention: int = st.sidebar.number_input("Monthly Retention (months)", value=12, min_value=0)
yearly_retention: int = st.sidebar.number_input("Yearly Retention (years)", value=0, min_value=0)

max_retention: int = max(
    math.ceil(daily_retention / 365),
    math.ceil(weekly_retention / 52),
    math.ceil(monthly_retention / 12),
    yearly_retention
)
if max_retention > growth_projection_years:
    st.sidebar.warning("Retention periods exceed growth projection years. Please adjust your inputs.")

# Main page toggles
unit: str = st.radio("Select unit for display:", ("GB", "TB"), horizontal=True)
conversion_factor: float = 1 if unit == "GB" else 1 / 1024

use_compound_growth: bool = st.toggle("Use compound growth", value=True)

# Calculate results
results: Dict[str, float] = calculate_backup_sizes(
    db_size_gb, annual_growth_rate, log_gen_gb_per_day,
    daily_retention, weekly_retention, monthly_retention, yearly_retention,
    growth_projection_years, compression_ratio, incremental_change_rate,
    use_compound_growth
)

# Generate dataframe for visualization
df: pd.DataFrame = generate_results_dataframe(
    db_size_gb, annual_growth_rate, log_gen_gb_per_day,
    daily_retention, weekly_retention, monthly_retention, yearly_retention,
    compression_ratio, incremental_change_rate, growth_projection_years,
    use_compound_growth
)

# Apply conversion to dataframe for display
df_display: pd.DataFrame = df.copy()
for col in df_display.columns:
    if col != 'Month':
        df_display[col] = df_display[col] * conversion_factor

# Main content
st.title("Oracle RMAN Backup Storage Requirements")

# Create two columns for table and chart
table_col, chart_col = st.columns(2)

with table_col:
    st.subheader(f"Storage Growth Over Time ({unit})")
    st.dataframe(df_display, use_container_width=True, height=400)

with chart_col:
    st.subheader(f"Storage Growth Visualization ({unit})")
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(df['Month'], df['Weekly Full Backups'] * conversion_factor, label='Weekly Full Backups', color='blue')
    plt.plot(df['Month'], df['Monthly Full Backups'] * conversion_factor, label='Monthly Full Backups', color='purple')
    plt.plot(df['Month'], df['Yearly Full Backups'] * conversion_factor, label='Yearly Full Backups', color='brown')
    plt.plot(df['Month'], df['Daily Incrementals'] * conversion_factor, label='Daily Incrementals', color='green')
    plt.plot(df['Month'], df['Logs'] * conversion_factor, label='Logs', color='orange')
    plt.plot(df['Month'], df['Total Capacity'] * conversion_factor, label='Total Capacity', color='red', linewidth=2)
    plt.xlabel('Months')
    plt.ylabel(f'Storage Size ({unit})')
    plt.title(f'Oracle RMAN Backup Storage Growth Over Time ({unit})')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

# Summary table at the bottom
st.subheader(f"Summary of Current Storage Requirements ({unit})")
summary_data: Dict[str, list] = {
    "Component": [
        "Single Full Backup", "Single Incremental Backup", "Daily Incrementals", 
        "Weekly Fulls", "Monthly Fulls", "Yearly Fulls", "Total Backup Storage", "Archived Logs Storage",
        "Total Current Storage", f"Projected Storage ({growth_projection_years} years)"
    ],
    "Size": [
        results['full_backup_gb'] * conversion_factor,
        results['incremental_backup_gb'] * conversion_factor,
        results['daily_incrementals_gb'] * conversion_factor,
        results['weekly_fulls_gb'] * conversion_factor,
        results['monthly_fulls_gb'] * conversion_factor,
        results['yearly_fulls_gb'] * conversion_factor,
        results['total_backup_gb'] * conversion_factor,
        results['total_logs_gb'] * conversion_factor,
        results['total_storage_gb'] * conversion_factor,
        results['projected_storage_gb'] * conversion_factor
    ]
}
summary_df: pd.DataFrame = pd.DataFrame(summary_data)
st.table(summary_df)

# Veeam Repository sizing recommendation
st.subheader("Repository Sizing Recommendation")
full_backup_channels: int = 3 * 4  # 3 parallel backups with 4 channels each
log_shipping_channels: int = 7     # 7 systems shipping logs with 1 channel each
total_channels: int = full_backup_channels + log_shipping_channels

cpu_cores: int = max(1, total_channels // 5)  # 1 CPU core per 5 channels
memory_gb: int = max(1, total_channels // 5)  # 1 GB RAM per 5 channels
memory_with_headroom: float = memory_gb * 1.15

repo_data: Dict[str, list] = {
    "Resource": ["CPU Cores", "Memory", "Storage", "RMAN Compression", "Network"],
    "Recommendation": [
        f"{cpu_cores} cores (1 core per 5 channels)",
        f"{memory_gb} GB + 15% = {memory_with_headroom:.1f} GB",
        f"{results['projected_storage_gb'] * conversion_factor:.2f} {unit} (with {growth_projection_years}-year growth)",
        "MEDIUM (best balance of performance and size)",
        "Enable Jumbo Frames, 10 Gbps or better"
    ]
}
repo_df: pd.DataFrame = pd.DataFrame(repo_data)
st.table(repo_df)