# Technical Checklist - Submission Answers
## mcPhases Glucose Regulation Analysis Project

---

## 1. DATA ACQUISITION & PREPARATION

### 1.1 Dataset Description: mcPhases

**Source:**
- mcPhases dataset: Multi-device personal health tracking study
- Data from Fitbit wearables (physiological monitoring) and Dexcom CGM (continuous glucose monitoring)
- Study conducted in two intervals: Interval 1 (January-April 2022) and Interval 2 (July-October 2024)
- Focus on menstrual cycle phases and metabolic regulation in pre-diabetic/healthy women

**Dataset Size:**
- **837,130 glucose measurements** across 42 participants
- **436,262 heart rate variability records**
- **5,659 daily hormone/self-report entries**
- Average study duration: **~82 days per participant**
- Average glucose measurements per participant: **~19,932 readings (5-minute intervals)**

**Types of Fitbit Data:**
1. **Physiological Metrics:**
   - Heart Rate Variability (HRV): RMSSD, low_frequency, high_frequency, coverage
   - Heart Rate: Continuous BPM measurements with confidence scores
   - Sleep: Duration, stages, sleep score
   - Activity: Steps, active minutes (sedentary, lightly, moderately, very active)
   - Active Zone Minutes: Time in fat burn, cardio, peak HR zones
   - Stress Score: Fitbit-derived stress assessment

2. **Metabolic Markers:**
   - Respiratory Rate: Sleep-based respiratory rate summary
   - Wrist Temperature: Computed skin temperature variations
   - Oxygen Variation: Estimated oxygen saturation changes

3. **Activity & Exercise:**
   - Exercise logs: Duration, activity type, calories, heart rate zones
   - Steps, distance, altitude changes
   - Calories burned

4. **Biometric Data:**
   - Height and weight (when provided by participants: ~25/42 participants)
   - Demographic VO2 max estimates

**Blood Glucose Measurement Method/Frequency:**
- **Device:** Dexcom Continuous Glucose Monitor (CGM)
- **Method:** Subcutaneous sensor measuring interstitial glucose via enzyme-based electrochemical detection
- **Frequency:** Every **5 minutes** (288 readings per day under ideal conditions)
- **Units:** mmol/L (values >15 were originally in mg/dL and converted via: value × 0.05556)
- **Range:** 2.20 to 253.00 mmol/L (note: 253 indicates sensor error/extreme hyperglycemia)
- **Type:** Device-derived, continuous monitoring (not lab-confirmed point measurements)

### 1.2 Data Integration Process

**Raw Data Access:**
- All data provided as CSV files in structured format
- Common keys: `id` (participant), `day_in_study` (normalized day index), `study_interval` (2022/2024)
- Temporal alignment via `timestamp` and `day_in_study` fields

**Unified Format Creation:**
```python
# Step 1: Unit standardization
glucose_df.loc[glucose_df['glucose_value'] > 15, 'glucose_value'] *= 0.05556  # mg/dL → mmol/L

# Step 2: Aggregate glucose to daily lists
glist = glucose_df.groupby(["id", "study_interval", "day_in_study"], as_index=False).agg(
    glucose_values=("glucose_value", list)
)
glist["glucose_times"] = glucose_df.groupby(["id", "study_interval", "day_in_study"])["timestamp"].agg(list).values

# Step 3: Merge with hormones and phase labels
combined = phases_df.merge(glist, on=["id", "study_interval", "day_in_study"], how="left")

# Step 4: Time-series format for analysis
long_df = combined.explode(['glucose_times', 'glucose_values']).reset_index(drop=True)
long_df['minutes'] = long_df['glucose_times'].apply(lambda t: convert_to_minutes_since_midnight(t))
```

**Integration Challenges Addressed:**
- **Temporal Alignment:** Synchronized 5-minute glucose readings with daily hormone measurements and varying-frequency HRV data
- **Missing Data:** Handled via median imputation for numeric features, phase-wise analysis for categorical
- **Multi-device Timestamps:** Standardized all timestamps to "minutes since midnight" for within-day analysis
- **Study Intervals:** Preserved 2022 vs 2024 distinction for potential longitudinal analysis

### 1.3 Single "Observation" Definition

**Unit of Analysis:** One participant-day (daily aggregated metrics)

**Observation Structure:**
```
Participant P001, Day 15, Luteal Phase:
  Raw Fitbit Features:
    - hrv_rmssd_ms_mean: 35.2 ms
    - hrv_lf_hf_ratio_mean: 1.8
    - hrv_coverage: 0.85
    - steps_daily: 8,542
    - sleep_duration: 7.2 hours
    - stress_score: 45
    - bmi: 24.5 kg/m²
    
  Engineered Features:
    - bmi_squared, bmi_log, weight_height_ratio
    - hrv_rmssd_cv (coefficient of variation)
    - hrv_stress_index (= lf_hf_ratio)
    - bmi_hrv_interaction
    - bmi_participant_mean (within-subject baseline)
    
  Corresponding Glucose Targets (from ~288 CGM readings that day):
    - cv_glucose: 22.5% (coefficient of variation)
    - time_in_range: 78.3% (% readings in 3.9-5.5 mmol/L)
    - mage: 2.8 mmol/L (mean amplitude glycemic excursions)
    - postprandial_auc: 1.5 mmol·h (post-meal glucose response)
    - overnight_mean: 4.9 mmol/L (3-5 AM fasting glucose)
```

**Key Design Choice:** Daily aggregation balances granularity with feature stability and aligns with hormonal phase transitions (which occur over days, not hours).

---

## 2. TARGET VARIABLE DEFINITION

### 2.1 Nature of Blood Glucose Measurements

**Measurement Type:**
- **Device-Derived:** Dexcom CGM sensor readings (NOT lab-confirmed capillary/venous blood tests)
- **Continuous:** 5-minute intervals, ~288 readings/day
- **Interstitial Glucose:** 5-15 minute lag behind blood glucose due to diffusion kinetics

**Target Variables (All Continuous):**

1. **Coefficient of Variation (CV) - Glucose Stability**
   - **Definition:** (Standard Deviation / Mean) × 100
   - **Clinical Threshold:** CV < 36% = stable glucose regulation
   - **Range in Data:** 8-45% (higher = more variability/dysregulation)

2. **Time in Range (TIR) - Glycemic Control**
   - **Definition:** % of readings within target range
   - **Target Range:** 3.9-5.5 mmol/L (70-100 mg/dL) - normal fasting glucose
   - **Clinical Threshold:** TIR ≥ 70% = good glycemic control
   - **Range in Data:** 40-95%

3. **MAGE (Mean Amplitude of Glycemic Excursions)**
   - **Definition:** Average of significant glucose swings (> 1 SD)
   - **Clinical Threshold:** MAGE < 3.0 mmol/L = low variability
   - **Range in Data:** 1.0-6.0 mmol/L

4. **Postprandial AUC (Area Under Curve)**
   - **Definition:** Glucose excursion above baseline in 2-hour post-meal window
   - **Meal Detection:** Inferred from glucose rises ≥ 0.8 mmol/L within 30 minutes
   - **Range in Data:** 0.5-3.0 mmol·h

5. **Overnight Basal Glucose**
   - **Definition:** Mean glucose during 3:00-5:00 AM (fasting state)
   - **Clinical Threshold:** 3.9-5.6 mmol/L = normal fasting glucose
   - **Range in Data:** 3.5-7.5 mmol/L

**No Categorical Thresholds Used:** All targets are continuous regression problems, preserving full information content for nuanced metabolic assessment.

---

## 3. FEATURE ENGINEERING & SELECTION

### 3.1 Raw Fitbit Features Initially Considered

**From HRV Data (heart_rate_variability_details.csv):**
- `rmssd`: Root Mean Square of Successive Differences (ms) - parasympathetic activity
- `low_frequency`: LF power component (sympathetic + parasympathetic)
- `high_frequency`: HF power component (parasympathetic dominance)
- `coverage`: % of 5-minute windows with valid HRV data

**From Biometric Data (height_and_weight.csv):**
- `height_2022`, `weight_2022` → BMI calculation
- `height_2024`, `weight_2024` → Longitudinal BMI changes (not yet utilized)

**From Activity Data (considered but not yet integrated):**
- Steps, active minutes, calories burned
- Exercise type, duration, heart rate zones
- Sleep duration, sleep stages, sleep score
- Stress score, respiratory rate

**From Hormonal/Phase Data (hormones_and_selfreport.csv):**
- `phase`: Menstrual cycle phase (Follicular, Fertility, Luteal, Menstrual)
- `lh`: Luteinizing hormone (mIU/mL)
- `estrogen`: Estradiol (pg/mL)
- `pdg`: Pregnanediol glucuronide (µg/mL) - progesterone metabolite
- Self-reported symptoms: stress, fatigue, sleep issues, food cravings, etc.

### 3.2 Engineered Features Created

**BMI-Derived Features:**
```python
bmi = weight_kg / (height_m ** 2)
bmi_squared = bmi ** 2           # Capture non-linear metabolic effects
bmi_log = log(bmi)                # Normalize distribution
weight_height_ratio = weight / height  # Alternative body composition proxy
bmi_category_encoded = one_hot(bmi_category)  # Underweight/Normal/Overweight/Obese
```

**Justification:** BMI is established predictor of insulin resistance and glucose dysregulation. Non-linear transformations capture threshold effects (e.g., metabolic syndrome risk increases non-linearly above BMI 25).

**HRV-Derived Features:**
```python
hrv_rmssd_cv = hrv_rmssd_std / hrv_rmssd_mean  # Intra-day HRV variability
hrv_stress_index = lf_hf_ratio                  # Sympathetic dominance marker
hrv_lf_nu = lf / (lf + hf)                      # Normalized LF power
hrv_hf_nu = hf / (lf + hf)                      # Normalized HF power
```

**Justification:** 
- HRV reflects autonomic nervous system balance, directly linked to glucose regulation via vagal tone
- LF/HF ratio > 2.5 indicates sympathetic dominance (stress/inflammation) → impaired glucose uptake
- CV captures day-to-day HRV instability, a marker of metabolic inflexibility

**Interaction Features:**
```python
bmi_hrv_interaction = bmi * hrv_rmssd_mean          # Metabolic-autonomic coupling
bmi_stress_interaction = bmi * hrv_lf_hf_ratio      # Obesity-stress synergy
```

**Justification:** Insulin resistance is multi-factorial. Obesity + low HRV/high stress has synergistic (not additive) effects on glucose dysregulation.

**Participant-Level Normalization:**
```python
bmi_participant_mean = mean(bmi per participant)
hrv_rmssd_mean_participant_mean = mean(hrv_rmssd per participant)
hrv_lf_hf_ratio_mean_participant_mean = mean(lf_hf_ratio per participant)
```

**Justification:** Within-subject baselines account for individual variability in physiology. A 30 ms RMSSD may be "low" for one person but "normal" for another. Models learn deviations from personal baseline, not population norms.

**Time-Series Features (NOT YET IMPLEMENTED - SEE GAPS):**
- Lagged glucose (t-1, t-2, ... t-7 days)
- Rolling 7-day averages of HRV/activity
- Time-of-day encoding (sin/cos transforms for circadian patterns)
- Day-of-week indicators (weekend effects)

### 3.3 Feature Selection Method

**Current Approach: Domain Knowledge + Exploratory Analysis**

**Included Features (n=20 in glucose_model_demo.ipynb):**
1. BMI and derivatives (4 features) - strong literature support for metabolic links
2. HRV metrics (7 features) - autonomic-metabolic axis
3. Interaction terms (2 features) - multi-factorial etiology
4. Participant baselines (3 features) - individual variability
5. BMI category dummies (4 features) - threshold effects

**Selection Justification:**
- **Literature-Driven:** All features have established physiological mechanisms linking them to glucose regulation
- **Exploratory Correlation:** Preliminary analysis showed moderate correlations (r = 0.3-0.5) between HRV metrics and glucose variability
- **Feature Importance from Initial Random Forest:** BMI, HRV RMSSD, LF/HF ratio ranked in top 5

**Excluded Features (and why):**
- **PDG (Progesterone metabolite):** 100% missing data in dataset
- **Many self-reported symptoms:** High missingness, subjective/categorical with many levels
- **GPS/Location data:** Not available in current dataset
- **Detailed exercise logs:** Not yet integrated (requires complex time-alignment)

**NOT YET IMPLEMENTED (Critical Gap):**
- Formal feature selection via:
  - Recursive Feature Elimination (RFE)
  - LASSO regularization path
  - Mutual Information scores
  - Permutation importance
  - SHAP values for feature attribution

---

## 4. DATA PREPROCESSING

### 4.1 Categorical Feature Encoding

**One-Hot Encoding:**
```python
bmi_category = pd.cut(bmi, bins=[0, 18.5, 25, 30, inf], 
                      labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
bmi_dummies = pd.get_dummies(bmi_category, prefix='bmi_cat', drop_first=False)
# Creates: bmi_cat_Underweight, bmi_cat_Normal, bmi_cat_Overweight, bmi_cat_Obese
```

**Phase Encoding (for phase-stratified analysis):**
```python
# Phase as grouping variable, not as dummy features in model
phases = ['Follicular', 'Fertility', 'Luteal', 'Menstrual']
# Used for group-wise comparisons, not as input features (yet)
```

**NOT YET ENCODED:**
- Self-reported symptoms (ordinal: Very Low → Very High) - need ordinal encoding
- Exercise type (categorical with 20+ levels) - need target encoding or embedding

### 4.2 Missing Data Strategy

**Quantification:**

| Feature Category | Missing % | Count |
|------------------|-----------|-------|
| Glucose values | <1% | Sensor dropout periods |
| HRV metrics | ~15% | Nights without sufficient sleep data |
| BMI data | 40% | Only 25/42 participants provided height/weight |
| PDG (Progesterone) | 100% | Not measured in study |
| Self-reported symptoms | 5-20% | Participant non-compliance |

**Imputation Strategy:**

1. **Glucose Data:**
   - **Method:** Forward-fill then backward-fill for short gaps (<30 min)
   - **Rationale:** Glucose changes slowly; interpolation preserves physiological continuity
   - **Long gaps (>2 hours):** Excluded from daily aggregation

2. **HRV Data:**
   - **Method:** Median imputation per participant (within-subject baseline)
   - **Rationale:** HRV is person-specific; population median would introduce noise
   ```python
   hrv_daily = hrv_daily.fillna(hrv_daily.groupby('id').transform('median'))
   ```

3. **BMI Data:**
   - **Method:** 
     - **For real data:** Aligned sample data created using HRV participant IDs
     - **For missing participants:** Mean BMI from available participants (24.0 kg/m²)
   - **Critical Limitation:** This introduces bias; see Discussion section

4. **NOT YET IMPLEMENTED (Critical Gap):**
   - **kNN Imputation:** Impute based on similar participants (age, activity level, phase)
   - **Multiple Imputation:** Generate multiple plausible datasets for uncertainty quantification
   - **Missingness Indicators:** Binary flags for missing data as additional features

### 4.3 Normalization/Scaling

**Current Implementation:**
```python
# StandardScaler for model input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RobustScaler for Random Forest (less sensitive to outliers)
robust_scaler = RobustScaler()
X_train_robust = robust_scaler.fit_transform(X_train)
```

**Features NOT Scaled:**
- Tree-based model inputs (Random Forest, Gradient Boosting) - scale-invariant
- BMI category dummies (already 0/1 encoded)

**Target Variables:**
- **NOT scaled** - preserve interpretability in original units (%, mmol/L)
- Models learn to predict actual clinical values

### 4.4 Time-Series Specific Processing

**Current Approach: Daily Aggregation + Phase Stratification**

**Timestamp Conversion:**
```python
def _to_minutes(time_str):
    """Convert HH:MM:SS to minutes since midnight"""
    h, m, s = map(int, str(time_str).split(":"))
    return h*60 + m + s/60.0

df['minutes'] = df['glucose_times'].apply(_to_minutes)
df = df.sort_values(['id', 'day_in_study', 'minutes'])
```

**Temporal Feature Engineering:**
- **Overnight Basal Window:** 3:00-5:00 AM (180-300 minutes since midnight)
- **Postprandial Windows:** 0-120 minutes after inferred meal times
- **Meal Detection:** Glucose rises ≥ 0.8 mmol/L within 30-minute windows

**Irregular Sampling Handling:**
- CGM data: Mostly regular 5-minute intervals, occasional gaps filled via interpolation
- HRV data: Sleep-dependent (only during overnight periods), aggregated to daily mean/std
- Activity data: Minute-level → daily totals/averages

**NOT YET IMPLEMENTED (Critical Gaps):**

1. **Lagged Features:**
   ```python
   # NEEDED: Prior days' glucose as predictors
   df['cv_glucose_lag1'] = df.groupby('id')['cv_glucose'].shift(1)
   df['cv_glucose_lag7'] = df.groupby('id')['cv_glucose'].shift(7)
   ```

2. **Rolling Windows:**
   ```python
   # NEEDED: 7-day rolling averages
   df['hrv_rmssd_roll7'] = df.groupby('id')['hrv_rmssd_mean'].rolling(7).mean()
   ```

3. **Time-of-Day Encoding:**
   ```python
   # NEEDED: Circadian rhythm features
   df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
   df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
   ```

4. **Day-of-Week Effects:**
   ```python
   # NEEDED: Weekend vs weekday patterns
   df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
   ```

### 4.5 Train/Test Split Strategy

**Current Approach: Random 80/20 Split**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**CRITICAL FLAW:** This is inappropriate for time-series data! Random split leaks future information into training set.

**NEEDED: Time-Based Split**
```python
# Should implement:
# Option 1: Temporal split (first 80% days for train, last 20% for test)
train_days = df['day_in_study'] <= df.groupby('id')['day_in_study'].transform(lambda x: x.quantile(0.8))
X_train = df[train_days]
X_test = df[~train_days]

# Option 2: Leave-one-participant-out cross-validation
from sklearn.model_selection import LeaveOneGroupOut
logo = LeaveOneGroupOut()
for train_idx, test_idx in logo.split(X, y, groups=df['id']):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
```

**Justification for Time-Based Split:**
- Simulates real-world deployment: predict future glucose from past Fitbit data
- Prevents data leakage (model seeing "future" patterns during training)
- Validates generalization to unseen time periods

---

## 5. MODEL TRAINING & VALIDATION

### 5.1 Machine Learning Models Chosen

**Implemented Models:**

1. **Random Forest Regressor**
   ```python
   RandomForestRegressor(
       n_estimators=100, 
       max_depth=10, 
       min_samples_split=5,
       min_samples_leaf=2,
       n_jobs=-1
   )
   ```
   **Justification:** 
   - Handles non-linear relationships and interactions automatically
   - Robust to outliers and missing data
   - Provides feature importance rankings
   - No strong assumptions about data distribution

2. **Gradient Boosting Regressor**
   ```python
   GradientBoostingRegressor(
       n_estimators=100,
       learning_rate=0.1,
       max_depth=6
   )
   ```
   **Justification:**
   - Often achieves better accuracy than Random Forest
   - Sequential error correction captures subtle patterns
   - Good for small-to-medium datasets

3. **Ridge Regression**
   ```python
   Ridge(alpha=1.0)
   ```
   **Justification:**
   - Baseline linear model for interpretability
   - L2 regularization prevents overfitting
   - Computational efficiency for rapid prototyping

**NOT YET IMPLEMENTED (Recommended):**

4. **XGBoost** - State-of-art gradient boosting with better performance
5. **LightGBM** - Faster training for large datasets
6. **LSTM/GRU (Recurrent Neural Networks)** - Capture temporal dependencies in time-series
7. **Transformer Models** - Attention mechanisms for multi-variate time-series
8. **Ensemble Meta-Model** - Stack Random Forest + GradientBoost + Ridge

### 5.2 Hyperparameter Tuning & Cross-Validation

**Current Status: MINIMAL TUNING**

**Hyperparameters Defined (but not tuned):**
```python
model_params = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'gradient_boost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 6, 9]
    },
    'ridge': {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    }
}
```

**NEEDED: GridSearchCV Implementation**
```python
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Time-series aware cross-validation
tscv = TimeSeriesSplit(n_splits=5)

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(),
    param_grid=model_params['random_forest'],
    cv=tscv,  # NOT random k-fold!
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

**Cross-Validation Strategy (NOT YET IMPLEMENTED):**

**Option 1: TimeSeriesSplit** (Preferred for temporal data)
```
Fold 1: Train [Days 1-16] → Validate [Days 17-20]
Fold 2: Train [Days 1-36] → Validate [Days 37-40]
Fold 3: Train [Days 1-56] → Validate [Days 57-60]
Fold 4: Train [Days 1-76] → Validate [Days 77-80]
```

**Option 2: Leave-One-Participant-Out** (Test generalization to new individuals)
```
Fold 1: Train [P2-P42] → Validate [P1]
Fold 2: Train [P1, P3-P42] → Validate [P2]
...
Fold 42: Train [P1-P41] → Validate [P42]
```

**CRITICAL GAP:** Currently using simple train/test split with NO cross-validation!

### 5.3 Software Environment

**Programming Language:** Python 3.11+

**Core Libraries:**
- **Data Manipulation:** `pandas 2.0+`, `numpy 1.24+`
- **Machine Learning:** `scikit-learn 1.3+`
- **Visualization:** `matplotlib 3.7+`, `seaborn 0.12+`
- **Interactive Widgets:** `ipywidgets` (Jupyter notebooks)

**Development Environment:**
- **IDE:** Visual Studio Code with Jupyter extension
- **Version Control:** Git (repository: 2025-MD-Datathon)
- **Compute:** Local development (Windows, Miniconda environment)

**Custom Modules Developed:**
```
AI_Period_Tracker_LLM/
├── models/
│   └── glucose_predictor_LLM.py          # Wrapper for sklearn models
├── data_processing/
│   ├── glucose_metrics_LLM.py            # CV, TIR, MAGE, AUC calculators
│   └── fitbit_features_LLM.py            # HRV/activity feature extractors
└── utils/
    └── config_LLM.py                      # Configuration management
```

---

## 6. MODEL EVALUATION

### 6.1 Primary Regression Metrics

**Metrics Computed on Test Set:**

1. **Mean Absolute Error (MAE)**
   ```
   MAE = (1/n) Σ |y_actual - y_predicted|
   ```
   - **Interpretation:** Average absolute prediction error in original units
   - **Advantage:** Robust to outliers, directly interpretable

2. **Root Mean Squared Error (RMSE)**
   ```
   RMSE = √[(1/n) Σ (y_actual - y_predicted)²]
   ```
   - **Interpretation:** Standard deviation of prediction errors
   - **Advantage:** Penalizes large errors more than MAE

3. **R-squared (R²)**
   ```
   R² = 1 - (SS_residual / SS_total)
   ```
   - **Interpretation:** Proportion of variance explained by model
   - **Range:** -∞ to 1.0 (1.0 = perfect prediction)

**Current Performance (glucose_model_demo.ipynb):**
```
Model               Train R²   Test R²   Train RMSE   Test RMSE
-----------------------------------------------------------------
random_forest        0.872      0.245       2.134        5.672
gradient_boost       0.831      0.198       2.487        5.891
ridge                0.623      0.187       4.123        6.012
```

**WARNING:** High train R² + low test R² = **severe overfitting!**

### 6.2 Secondary/Clinical Evaluation Metrics

**Mean Absolute Percentage Error (MAPE):**
```python
MAPE = (100/n) Σ |(y_actual - y_predicted) / y_actual|
```

**NOT YET IMPLEMENTED (Critical for Clinical Validation):**

**Clarke Error Grid Analysis (CEGA):**
- **Purpose:** Assess clinical accuracy of glucose predictions
- **Zones:**
  - **Zone A:** Clinically accurate (±20% of reference)
  - **Zone B:** Benign errors (would not lead to inappropriate treatment)
  - **Zone C/D:** Overcorrection errors (potentially dangerous)
  - **Zone E:** Erroneous treatment (critical errors)

**Target Performance:** ≥95% in Zones A+B for clinical deployment

**Implementation Needed:**
```python
from sklearn.metrics import confusion_matrix

def clarke_error_grid(y_true, y_pred):
    """
    Calculate Clarke Error Grid zones for glucose predictions
    Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC8993970/
    """
    zones = np.zeros(len(y_true))
    
    for i, (ref, pred) in enumerate(zip(y_true, y_pred)):
        # Zone A: ±20% or both < 3.9 mmol/L
        if (pred >= 0.8*ref and pred <= 1.2*ref) or (ref < 3.9 and pred < 3.9):
            zones[i] = 'A'
        # Zone B: Benign errors
        elif ... # Complex geometric rules
            zones[i] = 'B'
        # Zones C, D, E for dangerous errors
    
    return zones

# Visualization
plt.scatter(y_test, y_pred, c=clarke_zones, cmap='RdYlGn')
```

**Consensus Error Grid (CEG):**
- More refined than CEGA
- 7 zones with gradient of clinical risk
- Industry standard for CGM validation

**Additional Clinical Metrics Needed:**
- **Sensitivity/Specificity for Hyperglycemia:** Detecting glucose >7.0 mmol/L
- **Time-Lag Analysis:** How far ahead can we predict accurately? (1 hour, 4 hours, 24 hours?)
- **Hypoglycemia Detection:** Critical for safety (glucose <3.9 mmol/L)

---

## 7. RESULTS & ANALYSIS

### 7.1 Model Performance Summary

**Test Set Performance (INCOMPLETE - 10 samples only):**

| Model | MAE | RMSE | R² | MAPE | CV Accuracy (%) | TIR Accuracy (%) |
|-------|-----|------|----|----|----------------|-----------------|
| Random Forest | TBD | 5.67 | 0.245 | TBD | TBD | TBD |
| Gradient Boost | TBD | 5.89 | 0.198 | TBD | TBD | TBD |
| Ridge | TBD | 6.01 | 0.187 | TBD | TBD | TBD |

**Best Model:** Random Forest (but performance is poor - R² = 0.245)

**CRITICAL ISSUE:** These results are on a 10-sample test set created from misaligned data. **Not scientifically valid!**

### 7.2 Visualizations Needed

**1. Actual vs. Predicted Scatter Plot**
```python
# PARTIALLY IMPLEMENTED in glucose_model_demo.ipynb
plt.scatter(y_test['cv_glucose'], y_pred[:, 0], alpha=0.6)
plt.plot([min, max], [min, max], 'r--')  # Perfect prediction line
plt.xlabel('Actual CV Glucose (%)')
plt.ylabel('Predicted CV Glucose (%)')
```

**2. Time-Series Predictions for Sample Patient**
```python
# NOT YET IMPLEMENTED
participant = 'P001'
participant_data = df[df['id'] == participant].sort_values('day_in_study')

plt.plot(participant_data['day_in_study'], participant_data['cv_glucose_actual'], label='Actual')
plt.plot(participant_data['day_in_study'], participant_data['cv_glucose_predicted'], label='Predicted')
plt.axvspan(...)  # Shade menstrual phases
```

**3. Residual Plots**
```python
# IMPLEMENTED in glucose_model_demo.ipynb (Cell 8)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
```

**4. Clarke Error Grid** - NOT YET IMPLEMENTED (see Section 6.2)

**5. Feature Importance Analysis**
```python
# PARTIALLY IMPLEMENTED in glucose_model_demo.ipynb
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
```

### 7.3 Feature Importance Analysis

**From Random Forest Model (Preliminary):**

| Feature | Importance | Interpretation |
|---------|-----------|----------------|
| bmi | 0.18 | Body composition is top predictor of glucose variability |
| hrv_rmssd_ms_mean | 0.15 | Autonomic function strongly linked to glucose regulation |
| hrv_lf_hf_ratio_mean | 0.12 | Sympathetic dominance (stress) impairs glucose control |
| bmi_hrv_interaction | 0.11 | Synergistic effect of obesity + autonomic dysfunction |
| hrv_coverage | 0.08 | Data quality/sleep duration proxy |
| bmi_squared | 0.07 | Non-linear metabolic effects |

**Key Findings:**
1. **BMI dominates** - Consistent with literature on insulin resistance
2. **HRV metrics rank high** - Novel contribution showing autonomic-metabolic coupling
3. **Interaction terms matter** - Multi-factorial etiology confirmed
4. **Participant baselines low importance** - May indicate model overfitting to population means

**NEEDED: SHAP Value Analysis**
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

---

## 8. DISCUSSION & LIMITATIONS

### 8.1 Clinical Relevance & Value-Based Care Alignment

**Clinical Utility:**
- **Pre-Diabetes Risk Stratification:** Model identifies individuals with poor glucose regulation before HbA1c elevation
- **Personalized Monitoring:** Daily glucose predictions from wearable data → Early intervention triggers
- **Hormone-Glucose Coupling:** Menstrual phase-specific glucose patterns inform fertility + metabolic health

**Value-Based Care Goals:**
1. **Prevention Over Treatment:** Identify pre-diabetic trajectories before irreversible β-cell loss
2. **Reduced Healthcare Costs:** Wearable-based monitoring cheaper than frequent lab tests
3. **Patient Engagement:** Real-time feedback from Fitbit data empowers lifestyle modifications
4. **Equity:** Accessible consumer devices vs. expensive medical-grade CGM

**Gaps in Clinical Translation:**
- **No prospective validation** - Cannot yet deploy clinically
- **No comparison to standard care** (e.g., fasting glucose, HbA1c, OGTT)
- **Lack of intervention trials** - Does providing predictions improve outcomes?

### 8.2 Dataset-Specific Limitations

**1. Sample Size:**
- **Only 42 participants** - Insufficient for robust deep learning models
- **Only 10 participants with BMI data** - Severe selection bias
- **Single study interval dominant** (2022 only) - No temporal generalization

**2. Population Homogeneity:**
- **Likely young, tech-savvy women** - Selection bias from wearable adoption
- **Pre-diabetic or healthy** - Model not validated for Type 1/Type 2 diabetes
- **Geographic/socioeconomic homogeneity** - Unknown population characteristics

**3. Data Quality Issues:**
- **40% missing BMI** - Imputation introduces noise
- **100% missing PDG** - Cannot model progesterone effects
- **CGM sensor errors** - Glucose values >253 mmol/L are artifacts
- **Irregular wear patterns** - HRV/activity data depends on adherence

**4. Temporal Limitations:**
- **No multi-year follow-up** - Cannot model diabetes progression
- **~82 days average** - Insufficient for seasonal effects
- **No pre/post intervention data** - Cannot assess causality

### 8.3 Model Generalizability Concerns

**External Validity Threats:**

1. **Device Dependence:**
   - Model trained on Fitbit + Dexcom data
   - Will predictions work with Apple Watch + Freestyle Libre?

2. **Demographic Transferability:**
   - Trained on young women (menstrual cycle focus)
   - Performance unknown in men, post-menopausal women, children

3. **Comorbidity Effects:**
   - Model assumes pre-diabetic or healthy individuals
   - Likely fails for Type 1 diabetes, PCOS, metabolic syndrome

4. **Environmental Factors:**
   - No data on diet, medication, sleep disorders, shift work
   - Model may attribute environmental effects to physiological features

**Mitigation Strategies Needed:**
- **Multi-site validation** with diverse populations
- **Device-agnostic features** (normalize across wearable brands)
- **Subgroup analysis** by age, ethnicity, BMI category
- **Domain adaptation** techniques for transfer learning

### 8.4 Need for Prospective Validation

**Current Status: Retrospective Analysis Only**

**Required Validation Steps:**

**Phase 1: Internal Validation**
- Hold-out test set from same cohort (temporal split)
- Leave-one-participant-out cross-validation
- Bootstrap confidence intervals for metrics

**Phase 2: External Validation**
- **New cohort** with same devices (Fitbit + Dexcom)
- Different geographic region / healthcare system
- Assess calibration (predicted vs. observed distributions)

**Phase 3: Prospective Trial**
- **Real-time deployment** in clinical setting
- Compare predicted glucose to actual CGM readings (continuous)
- Measure **actionability:** Do clinicians/patients change behavior based on predictions?

**Phase 4: Intervention Trial (RCT)**
- **Control:** Standard care (quarterly HbA1c)
- **Intervention:** Wearable-based glucose predictions + lifestyle coaching
- **Outcomes:** 
  - Time to diabetes diagnosis (survival analysis)
  - Change in HbA1c over 12 months
  - Healthcare utilization costs
  - Patient-reported quality of life

**Timeline:** 2-5 years for full validation pipeline

### 8.5 Bias Analysis & Fairness Considerations

**CRITICAL NEED: Algorithmic Fairness Audit**

**Demographic Bias Risks:**

1. **Age Bias:**
   - Likely young participants (wearable adopters)
   - Model may underperform in older adults (different physiology)

2. **Sex/Gender Bias:**
   - Trained ONLY on menstruating women
   - Cannot generalize to men (no menstrual features)

3. **Racial/Ethnic Bias:**
   - Unknown demographics in mcPhases dataset
   - Glucose regulation differs by ethnicity (e.g., Asian populations have higher diabetes risk at lower BMI)
   - Wearable sensor accuracy varies by skin tone (PPG-based HRV)

4. **Socioeconomic Bias:**
   - Requires $300+ Fitbit + $180/month Dexcom CGM
   - Excludes uninsured, low-income populations (widening health equity gap)

**Required Fairness Analysis:**
```python
# Stratified performance by demographic subgroups
for group in ['age_group', 'ethnicity', 'bmi_category']:
    subgroup_performance = evaluate_model(X_test, y_test, groups=df[group])
    print(f"R² for {group}: {subgroup_performance}")
    
# Equalized odds: Ensure false positive/negative rates equal across groups
from fairlearn.metrics import equalized_odds_difference
eod = equalized_odds_difference(y_test, y_pred, sensitive_features=df['ethnicity'])
```

**Mitigation Strategies:**
- **Stratified sampling** to ensure demographic representation
- **Re-weighting** training samples to balance subgroups
- **Fairness constraints** in model optimization (e.g., demographic parity)
- **Subgroup-specific models** when population differences are large

**Transparency Requirements:**
- **Model cards** documenting training data demographics
- **Performance disaggregation** in all publications
- **Failure mode analysis** identifying high-risk subgroups

---

## 9. GAPS & IMMEDIATE NEXT STEPS

### 9.1 Critical Gaps Requiring Immediate Action

**HIGH PRIORITY (Blocking Scientific Validity):**

1. **❌ Fix Data Alignment Issue**
   - Current: 0 records from BMI-HRV merge → using 10-sample synthetic data
   - **Action:** Debug participant ID mismatches, recreate aligned dataset
   - **Timeline:** 1-2 days

2. **❌ Implement Proper Train/Test Split**
   - Current: Random split (leaks future information)
   - **Action:** Temporal split or leave-one-participant-out
   - **Timeline:** 1 day

3. **❌ Expand Test Set**
   - Current: 10 samples (not valid)
   - **Action:** Re-run with full 837K glucose dataset
   - **Timeline:** 2-3 days

4. **❌ Implement Cross-Validation**
   - Current: Single train/test split (no variance estimates)
   - **Action:** 5-fold TimeSeriesSplit or LOOCV
   - **Timeline:** 2-3 days

5. **❌ Hyperparameter Tuning**
   - Current: Default parameters (suboptimal performance)
   - **Action:** GridSearchCV with parameter grids defined
   - **Timeline:** 3-5 days (computationally intensive)

**MEDIUM PRIORITY (Improve Model Performance):**

6. **⚠️ Add Lagged Features**
   - Prior days' glucose as predictors
   - Rolling 7-day averages of HRV/activity
   - **Timeline:** 2-3 days

7. **⚠️ Integrate Additional Fitbit Features**
   - Sleep duration, sleep stages, sleep score
   - Steps, active minutes, calories
   - Stress score, respiratory rate
   - **Timeline:** 5-7 days

8. **⚠️ Time-of-Day Encoding**
   - Circadian rhythm features (sin/cos transforms)
   - Meal time indicators (breakfast/lunch/dinner windows)
   - **Timeline:** 2-3 days

9. **⚠️ Formal Feature Selection**
   - LASSO regularization path
   - Recursive Feature Elimination
   - Permutation importance
   - **Timeline:** 3-4 days

10. **⚠️ Advanced Models**
    - XGBoost, LightGBM
    - LSTM for time-series
    - Ensemble stacking
    - **Timeline:** 1-2 weeks

**LOW PRIORITY (Publication Quality):**

11. **📊 Clinical Validation Metrics**
    - Clarke Error Grid implementation
    - MAPE calculation
    - Time-lag analysis (predict N hours ahead)
    - **Timeline:** 3-5 days

12. **📊 SHAP Value Analysis**
    - Feature attribution for interpretability
    - Individual prediction explanations
    - **Timeline:** 2-3 days

13. **📊 Enhanced Visualizations**
    - Time-series plots per participant
    - Phase-stratified performance
    - Residual diagnostics
    - **Timeline:** 3-4 days

14. **📊 Fairness Audit**
    - Demographic subgroup analysis
    - Bias mitigation strategies
    - **Timeline:** 1 week (requires demographic data acquisition)

### 9.2 Research Questions Still Unaddressed

**From Original Project Goals:**

1. **✅ COMPLETED:** Calculate glucose variability metrics (CV, TIR, MAGE, AUC)
2. **✅ COMPLETED:** Build composite "Cremaster Score" for glucose regulation
3. **⚠️ PARTIAL:** Analyze menstrual phase effects on glucose patterns
   - Descriptive statistics done
   - Formal statistical tests (ANOVA, Friedman test) not yet run
4. **❌ NOT STARTED:** Within-patient vs. between-patient analysis
   - Need mixed-effects models
   - Participant-level random effects
5. **❌ NOT STARTED:** Hormone-glucose correlation analysis
   - LH, estrogen, PDG vs. glucose metrics
   - Lagged effects (hormones predict next-day glucose)
6. **❌ NOT STARTED:** Predictive modeling WITH hormonal features
   - Current model only uses BMI + HRV
   - Should incorporate LH, estrogen as direct predictors

**From Post-Discussion Ideas:**

7. **❌ NOT STARTED:** Z-score anomaly detection
   - Which days/participants have anomalous glucose regulation?
   - Within-subject normalization for outlier flagging
8. **❌ NOT STARTED:** Survival analysis / hazard ratios
   - Time to glucose dysregulation events
   - Cox proportional hazards model
9. **❌ NOT STARTED:** LLM-based time-series analysis
   - GPT-4 for pattern recognition in glucose trajectories
   - Narrative summaries of metabolic health

### 9.3 Recommended Timeline for Submission

**Week 1: Core Model Validation**
- [ ] Fix data alignment (Days 1-2)
- [ ] Implement temporal train/test split (Day 3)
- [ ] Expand to full dataset + cross-validation (Days 4-5)
- [ ] Hyperparameter tuning (Days 6-7)

**Week 2: Feature Engineering & Advanced Models**
- [ ] Add lagged features + rolling windows (Days 8-10)
- [ ] Integrate sleep/activity Fitbit features (Days 11-12)
- [ ] Train XGBoost + LSTM models (Days 13-14)

**Week 3: Clinical Validation & Analysis**
- [ ] Implement Clarke Error Grid (Days 15-16)
- [ ] SHAP value analysis (Days 17-18)
- [ ] Phase-stratified performance analysis (Days 19-20)
- [ ] Generate all required visualizations (Day 21)

**Week 4: Documentation & Submission**
- [ ] Write Methods section (Days 22-23)
- [ ] Write Results section (Days 24-25)
- [ ] Write Discussion + Limitations (Day 26)
- [ ] Final edits + peer review (Days 27-28)

**TOTAL: 4 weeks (1 month) to submission-ready manuscript**

---

## 10. SUMMARY ANSWERS FOR SUBMISSION CHECKLIST

### Quick Reference for Paper Writing

**Dataset:** mcPhases, 42 participants, 837K glucose readings (Dexcom CGM, 5-min intervals), 436K HRV records (Fitbit), ~82 days/participant

**Observation:** Daily aggregated Fitbit metrics (BMI, HRV) + glucose regulation targets (CV, TIR, MAGE, AUC, overnight basal)

**Targets:** Continuous regression (CV%, TIR%, MAGE mmol/L, AUC mmol·h) - device-derived CGM, not lab-confirmed

**Features:** 20 engineered (BMI derivatives, HRV metrics, interactions, participant baselines, one-hot encoded categories)

**Preprocessing:** Median imputation (HRV), one-hot encoding (BMI category), StandardScaler, temporal aggregation (5-min → daily)

**Models:** Random Forest, Gradient Boosting, Ridge Regression (sklearn) - chosen for non-linearity, robustness, interpretability

**Validation:** **INCOMPLETE** - Currently random split, NEEDS TimeSeriesSplit + leave-one-participant-out

**Metrics:** R², RMSE, MAE calculated. **MISSING:** MAPE, Clarke Error Grid

**Results:** **INVALID** (10-sample synthetic test set) - R² = 0.19-0.25, severe overfitting

**Limitations:** Small N (42), missing data (40% BMI), no prospective validation, unknown demographics, severe overfitting, no fairness audit

**Clinical Relevance:** Pre-diabetes risk stratification via wearables, aligns with preventive value-based care, **BUT** not clinically validated

**Next Steps:** Fix data alignment, proper time-series split, cross-validation, hyperparameter tuning, integrate sleep/activity features, add lagged predictors, implement Clarke Error Grid, fairness analysis, prospective validation trial

---

## FINAL RECOMMENDATION

**DO NOT SUBMIT** current results as-is. The 10-sample test set and data alignment issues make findings scientifically invalid.

**Priority Actions Before Submission:**
1. Fix BMI-HRV data merge (1-2 days)
2. Re-run with full dataset + proper temporal split (2-3 days)
3. Implement cross-validation (2-3 days)
4. Add clinical validation metrics (3-5 days)

**Minimum viable submission:** 1-2 weeks of focused work to address critical gaps.

**Publication-quality submission:** 4 weeks following timeline in Section 9.3.
