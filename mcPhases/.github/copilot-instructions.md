# mcPhases Glucose Analysis Project - AI Agent Instructions

## Project Overview
This is a medical datathon project analyzing continuous glucose monitoring (CGM) data across menstrual cycle phases. The core goal is to build glucose variability metrics and composite scores to understand glucose regulation patterns during different hormonal phases.

## Data Architecture & Key Entities

### Core Data Structure
- **Primary keys**: `id` (participant), `day_in_study`, `study_interval` (2022 vs 2024)
- **Main datasets**: `glucose.csv` (837K+ continuous readings), `hormones_and_selfreport.csv` (daily hormone levels + phase labels)
- **Glucose readings**: 5-minute intervals, require unit conversion (>15 = mg/dL → multiply by 0.05556 for mmol/L)
- **Phase categories**: "Follicular", "Fertility", "Luteal", "Menstrual"

### Standard Data Processing Pipeline
```python
# 1. Load and merge datasets
glucose_df = pd.read_csv('glucose.csv')
phases_df = pd.read_csv('hormones_and_selfreport.csv')

# 2. Unit conversion (critical!)
glucose_df.loc[glucose_df['glucose_value'] > 15, 'glucose_value'] *= 0.05556

# 3. Aggregate glucose to daily lists
glist = glucose_df.groupby(["id", "study_interval", "day_in_study"], as_index=False).agg(glucose_values=("glucose_value", list))
glist["glucose_times"] = glucose_df.groupby(["id", "study_interval", "day_in_study"])["timestamp"].agg(list).values

# 4. Merge with hormones/phases
combined = phases_df.merge(glist, on=["id", "study_interval", "day_in_study"], how="left")

# 5. Explode for time-series analysis
long_df = combined.explode(['glucose_times', 'glucose_values']).reset_index(drop=True)
```

## Core Analysis Functions (Reusable Library)

### Glucose Metrics Framework
The project implements a comprehensive glucose analysis library with these key functions:

1. **`prepare_glucose(df)`**: Standardizes data format, converts timestamps to minutes since midnight
2. **`overnight_basal(df, start="03:00:00", end="05:00:00")`**: Calculates fasting glucose (3-5 AM window)
3. **`variability_metrics(df)`**: Computes SD, CV, and MAGE (Mean Amplitude of Glycemic Excursions)
4. **`postprandial_auc(df)`**: Measures post-meal glucose response using inferred meal detection
5. **`decay_slope_k(df)`**: Estimates glucose clearance rate after peaks

### Composite Scoring System
The "Cremaster Score" combines multiple glucose metrics:
- Higher = worse glucose regulation
- Components: `overnight_mean + cv_glucose + mage + pp_auc_0_120 - k_decay`
- Supports within-person z-score normalization vs population-level

## Project-Specific Conventions

### Time Handling
- **Always convert timestamps**: Use `_to_minutes()` helper for "HH:MM:SS" → minutes since midnight
- **Sort data**: `df.sort_values(["id", "day_in_study", "minutes"])` before time-series operations
- **Handle missing times**: Use `dropna(subset=["minutes", "glucose_values"])`

### Data Grouping Patterns
- **Daily metrics**: Group by `["id", "day_in_study", "phase"]`
- **Phase comparisons**: Group by `["id", "phase"]` then aggregate across days
- **Statistical analysis**: Use repeated-measures ANOVA or Friedman tests for phase comparisons

### Visualization Standards
- **Phase color palette**: Follicular (#A097F3), Fertility (#4BB45C), Luteal (#F37F7D), Menstrual (#311C3B)
- **Time-series plots**: Show mean ± SEM, not SD
- **Multi-panel layouts**: Use `math.ceil(n_phases / 2)` rows for phase comparisons

## File Organization
- `data/mcphases/1.0.0/`: Contains all CSV datasets and analysis notebooks
- **Exploration notebooks**: `explore.00.ipynb`, `score_explore.ipynb`, `score_explore.01.ipynb`, `score_explore.02.ipynb`
- **Key data files**: `glucose.csv`, `hormones_and_selfreport.csv`, plus 20+ other physiological metrics

## Critical Implementation Details

### Memory Management
- **Large datasets**: `glucose.csv` has 837K+ rows - use chunking or filtering for single-participant analysis
- **List columns**: After groupby operations, glucose data becomes nested lists - handle with `.explode()` for individual values

### Statistical Considerations
- **Repeated measures**: Same participants across multiple phases - use appropriate statistical tests
- **Missing data**: Not all participants have data for all phases - filter accordingly
- **Outlier detection**: Glucose values outside physiological range (e.g., >20 mmol/L) may indicate sensor errors

### Performance Patterns
- **Pre-filter by participant**: For individual analysis, filter by `id` early to reduce memory usage
- **Vectorized operations**: Use pandas aggregations instead of loops for time-series calculations
- **Caching**: Store processed `df_ready` and daily metrics to avoid recomputation

## Development Workflow
- **Notebook-driven**: Primary development in Jupyter notebooks with function definitions at top
- **Iterative refinement**: Multiple versions (`.01`, `.02`) show evolution of analysis approach
- **No formal testing**: Exploratory research project - validate results through visualization and sanity checks

## External Dependencies
- Standard data science stack: `pandas`, `numpy`, `matplotlib`, `seaborn`
- Statistics: `scipy.stats`, `statsmodels` for repeated-measures ANOVA
- Time-series: Custom time conversion functions (no specialized libraries)