import io
import numpy as np
import pandas as pd

REQUIRED_EDUCATIONAL_COLUMNS = ['attendance', 'marks', 'lms_activity', 'previous_performance']

COLUMN_ALIASES = {
    'att': 'attendance',
    'attendance_rate': 'attendance',
    'presence': 'attendance',
    'score': 'marks',
    'grade_marks': 'marks',
    'exam_score': 'marks',
    'lms': 'lms_activity',
    'lms_score': 'lms_activity',
    'engagement': 'lms_activity',
    'prev_perf': 'previous_performance',
    'previous_marks': 'previous_performance',
    'prev_performance': 'previous_performance'
}


def validate_dataset(df):
    """
    Validate dataset structure, required feature schema, and numeric integrity
    against the EduPredict Big Data specification.
    """
    if df is None or df.empty:
        return {"valid": False, "message": "The uploaded dataset contains zero rows or is completely empty."}

    # Normalize column names
    normalized_cols = {col: col.strip().lower().replace(" ", "_") for col in df.columns}
    df_renamed = df.rename(columns=normalized_cols)

    # Resolve column aliases
    for alias, target in COLUMN_ALIASES.items():
        if alias in df_renamed.columns and target not in df_renamed.columns:
            df_renamed.rename(columns={alias: target}, inplace=True)

    missing = [col for col in REQUIRED_EDUCATIONAL_COLUMNS if col not in df_renamed.columns]
    if missing:
        return {
            "valid": False,
            "message": f"Dataset schema missing required attributes: {', '.join(missing)}. "
                       f"Required features are: {', '.join(REQUIRED_EDUCATIONAL_COLUMNS)}."
        }

    # Verify numeric readability for required columns
    non_numeric = []
    for col in REQUIRED_EDUCATIONAL_COLUMNS:
        valid_numeric = pd.to_numeric(df_renamed[col], errors='coerce').notna().sum()
        if valid_numeric / len(df_renamed) < 0.5:
            non_numeric.append(col)

    if non_numeric:
        return {
            "valid": False,
            "message": f"Attributes must contain numeric percentage scores: {', '.join(non_numeric)}"
        }

    return {"valid": True, "message": "Dataset schema successfully validated.", "df": df_renamed}


def clean_dataset(df):
    """
    Clean dataset using statistical mean/mode imputation so models
    and parallel MapReduce jobs do not fail on missing data.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df_cleaned = df.copy()
    df_cleaned.drop_duplicates(inplace=True)
    df_cleaned.dropna(how='all', inplace=True)

    # Impute numeric columns with median
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = df_cleaned[col].median()
        df_cleaned[col] = df_cleaned[col].fillna(median_val if pd.notna(median_val) else 0)

    # Impute categorical columns with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_series = df_cleaned[col].mode()
        mode_val = mode_series[0] if not mode_series.empty else 'Nominal'
        df_cleaned[col] = df_cleaned[col].fillna(mode_val)

    return df_cleaned


def get_column_types(df):
    """Detect attribute data types for feature selection (X) and target (Y) mapping."""
    types = {}
    for col in df.columns:
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        valid_numeric_count = numeric_series.notna().sum()
        if len(df) > 0 and (valid_numeric_count / len(df)) >= 0.7:
            types[col] = 'numeric'
        else:
            types[col] = 'categorical'
    return types


def prepare_features(df, x_columns, y_column):
    """
    Extract independent features (X) and dependent target (Y)
    for model training and evaluation.
    """
    df_clean = clean_dataset(df)
    X = df_clean[x_columns].copy()
    y = df_clean[y_column].copy()

    for col in x_columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    y_numeric = pd.to_numeric(y, errors='coerce')
    if y_numeric.notna().sum() / max(1, len(y)) >= 0.7:
        y = y_numeric.fillna(y_numeric.median())
    else:
        y = y.astype(str).fillna("Unknown")

    return X, y


def generate_sample_data(n=1000):
    """
    Generate synthetic educational records across attendance, continuous marks,
    LMS clickstream telemetry, and historical indices.
    """
    np.random.seed(42)

    attendance = np.random.normal(loc=78, scale=14, size=n).clip(30, 100)
    marks = np.random.normal(loc=72, scale=16, size=n).clip(25, 100)
    lms_activity = np.random.normal(loc=75, scale=18, size=n).clip(20, 100)
    previous_performance = np.random.normal(loc=70, scale=15, size=n).clip(30, 100)

    # Weighted risk formulation
    risk_scores = (
        (100.0 - attendance) * 0.35 +
        (100.0 - marks) * 0.35 +
        (100.0 - lms_activity) * 0.15 +
        (100.0 - previous_performance) * 0.15
    )

    risk_levels = []
    grades = []
    for score, m in zip(risk_scores, marks):
        if score >= 45:
            risk_levels.append('High')
        elif score >= 28:
            risk_levels.append('Medium')
        else:
            risk_levels.append('Low')

        if m >= 80:
            grades.append('A')
        elif m >= 65:
            grades.append('B')
        elif m >= 50:
            grades.append('C')
        else:
            grades.append('F')

    return pd.DataFrame({
        'attendance': attendance.round(1),
        'marks': marks.round(1),
        'lms_activity': lms_activity.round(1),
        'previous_performance': previous_performance.round(1),
        'risk': risk_levels,
        'grade': grades
    })


def process_csv(file_content):
    """Read CSV from a path, raw string, or BytesIO stream."""
    if isinstance(file_content, (str, bytes)) and not os.path.exists(str(file_content)[:255] if isinstance(file_content, str) else ""):
        if isinstance(file_content, str):
            return pd.read_csv(io.StringIO(file_content))
        return pd.read_csv(io.BytesIO(file_content))
    return pd.read_csv(file_content)