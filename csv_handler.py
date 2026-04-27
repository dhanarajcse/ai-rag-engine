import pandas as pd
import re

def handle_csv_query(query, file_path):
    try:
        # -------------------------
        # READ CSV (safe)
        # -------------------------
        df = pd.read_csv(file_path, on_bad_lines='skip')

        # -------------------------
        # CLEANING
        # -------------------------
        df.columns = df.columns.str.strip().str.lower()

        # Convert numeric-like columns safely
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        query_lower = query.lower()

        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        # -------------------------
        # SMART TOTAL / SUM
        # -------------------------
        if numeric_cols:
            # column-specific total
            for col in numeric_cols:
                if col in query_lower and any(
                    word in query_lower for word in ["total", "sum", "overall", "combined"]
                ):
                    total = df[col].sum(skipna=True)
                    return f"{col} total = {round(total, 2)}"

            # generic total
            if any(word in query_lower for word in ["total", "sum", "overall", "combined"]):
                results = []
                for col in numeric_cols:
                    total = df[col].sum(skipna=True)
                    results.append(f"{col} total = {round(total, 2)}")
                return "\n".join(results)

        # -------------------------
        # FIRST N ROWS (e.g., first 5)
        # -------------------------
        match = re.search(r"first\s+(\d+)", query_lower)
        if match:
            n = int(match.group(1))
            df_subset = df.head(n).reset_index(drop=True)

            output = []
            for i, row in df_subset.iterrows():
                row_data = ", ".join(
                    f"{col}: {row[col]}" for col in df.columns
                )
                output.append(f"Row {i+1} → {row_data}")

            return "\n".join(output)

        # -------------------------
        # FILTER ROW (e.g., Ride 2, ID 5)
        # -------------------------
        for col in df.columns:
            if any(key in col for key in ["id", "no", "ride"]):
                for val in df[col].dropna():
                    if str(int(val)) in query_lower:
                        row = df[df[col] == val].iloc[0]
                        return ", ".join(
                            f"{c}: {row[c]}" for c in df.columns
                        )

        # -------------------------
        # SHOW ALL ROWS
        # -------------------------
        if any(word in query_lower for word in ["show", "list", "all", "rows"]):
            df = df.reset_index(drop=True)

            output = []
            for i, row in df.iterrows():
                row_data = ", ".join(
                    f"{col}: {row[col]}" for col in df.columns
                )
                output.append(f"Row {i+1} → {row_data}")

            return "\n".join(output)

        # -------------------------
        # SPECIFIC COLUMN
        # -------------------------
        for col in df.columns:
            if col in query_lower:
                return "\n".join(
                    f"{col}: {val}" for val in df[col]
                )

        # -------------------------
        # FALLBACK
        # -------------------------
        return "Query not supported for this CSV"

    except Exception as e:
        return f"Error processing CSV: {str(e)}"