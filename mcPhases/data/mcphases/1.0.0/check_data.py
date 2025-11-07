import pandas as pd
import numpy as np

# Load glucose data
glucose_df = pd.read_csv('glucose.csv')
print("=" * 60)
print("GLUCOSE DATA")
print("=" * 60)
print(f'Total rows: {len(glucose_df):,}')
print(f'Participants: {glucose_df["id"].nunique()}')
print(f'Study intervals: {glucose_df["study_interval"].unique()}')
print(f'Day range: {glucose_df["day_in_study"].min()} to {glucose_df["day_in_study"].max()} days')
print(f'\nGlucose value range: {glucose_df["glucose_value"].min():.2f} to {glucose_df["glucose_value"].max():.2f}')
print(f'Glucose value mean: {glucose_df["glucose_value"].mean():.2f}')
print('\nSample data:')
print(glucose_df.head())

# Load hormones and self report
hormones_df = pd.read_csv('hormones_and_selfreport.csv')
print("\n" + "=" * 60)
print("HORMONES & SELF-REPORT DATA")
print("=" * 60)
print(f'Total rows: {len(hormones_df):,}')
print(f'Participants: {hormones_df["id"].nunique()}')
print(f'Columns: {list(hormones_df.columns)}')
print(f'\nPhases: {hormones_df["phase"].unique()}')
print('\nSample data:')
print(hormones_df.head())

# Load HRV data
hrv_df = pd.read_csv('heart_rate_variability_details.csv')
print("\n" + "=" * 60)
print("HEART RATE VARIABILITY DATA")
print("=" * 60)
print(f'Total rows: {len(hrv_df):,}')
print(f'Participants: {hrv_df["id"].nunique()}')
print(f'Columns: {list(hrv_df.columns)}')

# Load height and weight
hw_df = pd.read_csv('height_and_weight.csv')
print("\n" + "=" * 60)
print("HEIGHT & WEIGHT DATA")
print("=" * 60)
print(f'Total rows: {len(hw_df):,}')
print(f'Participants: {hw_df["id"].nunique()}')
print(f'Columns: {list(hw_df.columns)}')
print('\nSample data:')
print(hw_df.head())

# Summary statistics
print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f'Total unique participants across datasets: {glucose_df["id"].nunique()}')
print(f'Glucose measurements per participant: {len(glucose_df) / glucose_df["id"].nunique():.0f} avg')
print(f'Study duration per participant: ~{glucose_df.groupby("id")["day_in_study"].max().mean():.0f} days avg')
