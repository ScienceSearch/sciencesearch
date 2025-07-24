import pandas as pd
import pandas as pd
import ast
from typing import List, Dict, Optional, Union, Tuple


class KeywordComparator:
    """
    A flexible class for comparing keyword extraction results between different conditions.
    Supports both simple 2-way comparisons and complex multi-condition comparisons.
    """

    def __init__(self):
        pass

    def _to_set(self, value):
        """Parsing of keyword columns that handles various formats.

        Attributes:
            value (str, list, or None): The keyword value to parse

        Returns:
            set (set): Set of cleaned keywords
        """

        if pd.isna(value) or value is None:
            return set()

        # If already a list
        if isinstance(value, list):
            return set(str(item).strip() for item in value if item)

        # Convert to string
        value = str(value).strip()

        # Handle empty strings
        if not value or value.lower() in ["nan", "none", ""]:
            return set()

        # Try to parse as Python literal (for lists stored as strings)
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return set(str(item).strip() for item in parsed if item)
        except (ValueError, SyntaxError):
            pass

        # Handle string representations of lists
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]  # Remove brackets

        # Clean and split
        value = value.replace("'", "").replace('"', "")

        if "," in value:
            split_list = [item.strip() for item in value.split(",")]
        else:
            split_list = [value.strip()]

        # Filter out empty strings
        return set(item for item in split_list if item and item.lower() != "nan")

    def _parse_keyword_column(self, value: Union[str, list, None]) -> set:
        """Alternative parsing method for more complex formats."""
        return self._to_set(value)  # Use the same logic for consistency

    def diff(
        self,
        csv_condition1,
        csv_condition2,
        csv_additional=None,
        condition1_name="condition1",
        condition2_name="condition2",
        experiment_col="experiment_name",
        keyword_col="predicted",
        similarity_metrics=False,
    ):
        """
        Simple 2-way comparison of keywords.

        Args:
            csv_condition1 (str): Path to first condition CSV
            csv_condition2 (str) Path to second condition CSV
            csv_additional (str, optional): Path to additional data CSV (like acronyms)
            condition1_name (str): Name for first condition
            condition2_name (str): Name for second condition
            experiment_col (str):  Name of experiment identifier column
            keyword_col (str): Name of keyword column
            similarity_metrics (bool): If comparison should include simialrity metrics

        Returns:
            pd.DataFrame: Comparison results
        """
        # Load the data
        df_1 = pd.read_csv(csv_condition1)
        df_2 = pd.read_csv(csv_condition2)

        # Merge the two main datasets
        merged = pd.merge(
            df_1[[experiment_col, keyword_col]],
            df_2[[experiment_col, keyword_col]],
            on=experiment_col,
            suffixes=(f"_{condition1_name}", f"_{condition2_name}"),
        )

        # Add additional data if provided
        if csv_additional:
            try:
                df_additional = pd.read_csv(csv_additional)
                # Try to merge with any matching columns
                common_cols = set(merged.columns) & set(df_additional.columns)
                if experiment_col in common_cols:
                    merged = pd.merge(
                        merged, df_additional, on=experiment_col, how="left"
                    )
            except Exception as e:
                print(f"Warning: Could not load additional file {csv_additional}: {e}")

        # Apply set operations
        results = []
        for _, row in merged.iterrows():
            set_1 = self._to_set(row[f"{keyword_col}_{condition1_name}"])
            set_2 = self._to_set(row[f"{keyword_col}_{condition2_name}"])

            # Calculate set operations
            common = set_1 & set_2
            unique_to_1 = set_1 - set_2
            unique_to_2 = set_2 - set_1
            all_keywords = set_1 | set_2

            result_row = {
                experiment_col: row[experiment_col],
                f"all keywords {condition1_name}": ", ".join(sorted(set_1)),
                f"all keywords {condition2_name}": ", ".join(sorted(set_2)),
                "common keywords": ", ".join(sorted(common)),
                f"unique to {condition1_name}": ", ".join(sorted(unique_to_1)),
                f"unique to {condition2_name}": ", ".join(sorted(unique_to_2)),
                f"count {condition1_name}": len(set_1),
                f"count {condition2_name}": len(set_2),
                "common count": len(common),
                f"unique count {condition1_name}": len(unique_to_1),
                f"unique count {condition2_name}": len(unique_to_2),
                "total unique keywords": len(all_keywords),
            }

            if similarity_metrics:  # Add similarity metrics
                jaccard = len(common) / len(all_keywords) if all_keywords else 0
                overlap = (
                    len(common) / min(len(set_1), len(set_2))
                    if min(len(set_1), len(set_2)) > 0
                    else 0
                )
                dice = (
                    (2 * len(common)) / (len(set_1) + len(set_2))
                    if (len(set_1) + len(set_2)) > 0
                    else 0
                )

                result_row["jaccard_similarity"] = jaccard
                result_row["overlap_coefficient"] = overlap
                result_row["dice_coefficient"] = dice

            # Copy any additional columns
            for col in merged.columns:
                if (
                    col not in result_row
                    and col != experiment_col
                    and not col.startswith(keyword_col)
                ):
                    result_row[col] = row[col]

            results.append(result_row)

        return pd.DataFrame(results)

    def load_and_prepare_data(
        self,
        csv_files: Dict[str, str],
        experiment_col: str = "experiment_name",
        keyword_col: str = "predicted",
        additional_cols: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Load and merge multiple CSV files for multi-condition comparison.

        Args:
            csv_files (dict):  Dictionary mapping condition names to CSV file paths
            experiment_col (str): Name of the column containing experiment identifiers
            keyword_col (str): Name of the column containing keywords
            additional_cols (dict, optional): Additional columns to include {new_name: column_name}

        Returns:
            pd.DataFrame: Merged dataframe ready for comparison
        """
        dataframes = {}

        # Load each CSV file
        for condition_name, file_path in csv_files.items():
            try:
                df = pd.read_csv(file_path)

                # Validate required columns
                if experiment_col not in df.columns:
                    raise ValueError(
                        f"Column '{experiment_col}' not found in {file_path}"
                    )
                if keyword_col not in df.columns:
                    raise ValueError(f"Column '{keyword_col}' not found in {file_path}")

                # Select relevant columns
                cols_to_keep = [experiment_col, keyword_col]
                if additional_cols:
                    for new_name, orig_name in additional_cols.items():
                        if orig_name in df.columns:
                            cols_to_keep.append(orig_name)

                df_subset = df[cols_to_keep].copy()
                dataframes[condition_name] = df_subset

                print(
                    f"Loaded {condition_name}: {len(df_subset)} experiments from {file_path}"
                )

            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue

        if len(dataframes) < 2:
            raise ValueError("Need at least 2 valid CSV files for comparison")

        # Merge dataframes
        condition_names = list(dataframes.keys())
        merged_df = dataframes[condition_names[0]].copy()

        # Rename keyword column for first condition
        merged_df = merged_df.rename(
            columns={keyword_col: f"{keyword_col} {condition_names[0]}"}
        )

        # Merge with other conditions
        for condition in condition_names[1:]:
            df_to_merge = dataframes[condition][[experiment_col, keyword_col]].copy()
            df_to_merge = df_to_merge.rename(
                columns={keyword_col: f"{keyword_col} {condition}"}
            )

            merged_df = pd.merge(merged_df, df_to_merge, on=experiment_col, how="outer")

        # Handle additional columns if specified
        if additional_cols:
            for condition in condition_names:
                for new_name, orig_name in additional_cols.items():
                    if orig_name in dataframes[condition].columns:
                        additional_data = dataframes[condition][
                            [experiment_col, orig_name]
                        ].copy()
                        additional_data = additional_data.rename(
                            columns={orig_name: f"{new_name}_{condition}"}
                        )
                        merged_df = pd.merge(
                            merged_df, additional_data, on=experiment_col, how="left"
                        )

        return merged_df

    def compare_conditions(
        self,
        merged_df: pd.DataFrame,
        conditions: List[str],
        experiment_col: str = "experiment_name",
        keyword_col_prefix: str = "predicted",
    ) -> pd.DataFrame:
        """
        Compare keyword sets between multiple conditions.

        Args:
            merged_df (pd.DataFrame): Merged dataframe from load_and_prepare_data
            conditions (list): List of condition names to compare
            experiment_col (str): Name of experiment identifier column
            keyword_col_prefix (str): Prefix for keyword columns

        Returns:
        pd.DataFrame: Comparison results with set operations
        """
        results = []

        for _, row in merged_df.iterrows():
            # Parse keyword sets for each condition
            keyword_sets = {}
            for condition in conditions:
                col_name = f"{keyword_col_prefix} {condition}"
                if col_name in row:
                    keyword_sets[condition] = self._parse_keyword_column(row[col_name])
                else:
                    keyword_sets[condition] = set()

            # Calculate set operations
            result_row = {experiment_col: row[experiment_col]}

            # Calculate pairwise comparisons for all condition pairs
            for i, cond1 in enumerate(conditions):
                for cond2 in conditions[i + 1 :]:
                    set1, set2 = keyword_sets[cond1], keyword_sets[cond2]

                    # Set operations
                    common = set1 & set2
                    unique_to_1 = set1 - set2
                    unique_to_2 = set2 - set1
                    all_keywords = set1 | set2

                    # Add to results
                    comparison_key = f"{cond1} vs. {cond2}"
                    result_row[f"common keywords"] = ", ".join(sorted(common))
                    result_row[f"common keywords count"] = len(common)

                    result_row[f"unique to {cond1}"] = ", ".join(sorted(unique_to_1))
                    result_row[f"unique count {cond1}"] = len(unique_to_1)

                    result_row[f"unique to {cond2}"] = ", ".join(sorted(unique_to_2))
                    result_row[f"unique count {cond2}"] = len(unique_to_2)
                    result_row[f"total unique keywords"] = len(all_keywords)

                    # Similarity metrics
                    jaccard = len(common) / len(all_keywords) if all_keywords else 0
                    overlap = (
                        len(common) / min(len(set1), len(set2))
                        if min(len(set1), len(set2)) > 0
                        else 0
                    )

                    result_row[f"jaccard similarity {comparison_key}"] = jaccard
                    result_row[f"overlap coefficient {comparison_key}"] = overlap

            # Copy additional columns
            for col in merged_df.columns:
                if col not in result_row and col != experiment_col:
                    result_row[col] = row[col]

            results.append(result_row)

        comparison_df = pd.DataFrame(results)
        return comparison_df

    def diff_acronyms(self, csv_acronym, csv_default, csv_acronyms_found):
        """
        Perform diff
        """
        return self.diff(
            csv_acronym,
            csv_default,
            csv_acronyms_found,
            condition1_name="with_expansion",
            condition2_name="without_expansion",
        )
