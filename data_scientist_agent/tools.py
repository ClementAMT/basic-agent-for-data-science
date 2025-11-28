from langchain_core.tools import tool
import pandas as pd


@tool("summarize_dataframe")
def summarize_dataframe(file_path: str):
    """Return basic statistics and a data preview."""
    df = pd.read_csv(file_path)
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "head": df.head().to_dict(),
        "describe": df.describe().to_dict()
    }

@tool("detect_types")
def detect_types(file_path: str):
    """Detect data types of each column."""
    df = pd.read_csv(file_path)
    return df.dtypes.astype(str).to_dict()


@tool("missing_values_report")
def missing_values_report(file_path: str):
    """Report missing values per column."""
    df = pd.read_csv(file_path)
    return df.isnull().sum().to_dict()

@tool("correlations")
def correlations(file_path: str):
    """Compute the correlation matrix among numerical columns."""
    df = pd.read_csv(file_path)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    return df[num_cols].corr().to_dict()

@tool("outliers")
def outliers(file_path: str, column: str):
    """Detect outliers in a specified column using the IQR method."""
    df = pd.read_csv(file_path)
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    return df[(df[column] < q1 - 1.5*iqr) | (df[column] > q3 + 1.5*iqr)][column].tolist()