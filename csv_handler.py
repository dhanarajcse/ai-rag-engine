import pandas as pd

def handle_csv_query(query, file_path):
    try:
        df = pd.read_csv(file_path)

        # normalize column names
        df.columns = df.columns.str.strip()

        # identify numeric columns automatically
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if not numeric_cols:
            return "No numeric data found in CSV"

        # -------------------------
        # TOTAL / SUM QUERY
        # -------------------------
        if any(word in query.lower() for word in ["total", "sum"]):
            results = []
            for col in numeric_cols:
                total = df[col].sum()
                results.append(f"{col} total = {round(total, 2)}")

            return "\n".join(results)

        # -------------------------
        # ROW-WISE DISPLAY
        # -------------------------
        if any(word in query.lower() for word in ["show", "list", "all"]):
            output = []

            for i, row in df.iterrows():
                row_data = ", ".join(
                    f"{col}: {row[col]}" for col in df.columns
                )
                output.append(f"Row {i+1} → {row_data}")

            return "\n".join(output)

        # -------------------------
        # SPECIFIC COLUMN QUERY
        # -------------------------
        for col in df.columns:
            if col.lower() in query.lower():
                return "\n".join(
                    f"{col}: {val}" for val in df[col]
                )

        return "Query not supported for this CSV"

    except Exception as e:
        return f"Error processing CSV: {str(e)}"