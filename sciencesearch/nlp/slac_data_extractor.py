import sqlite3
from sqlite3 import Cursor
import json
from pathlib import Path
import pandas as pd
from sciencesearch.nlp.preprocessing import Preprocessor


class SLACDatabaseDataExtractor:
    def __init__(self, config_file: str, replace_abbrv: bool = False):
        conf = json.load(open(config_file))
        training = conf["training"]
        self.training_file_path = Path(training["directory"])
        replacement_fp = Path(training["replacement_dict"])
        self.fp = conf["database"]
        self.connected = False
        with open("../private_data/queries_info.json", "r") as f:
            self._query_info = json.load(f)
        self.replace_abbrv = replace_abbrv
        if replace_abbrv:
            print(replacement_fp)
            self.preprocessor = Preprocessor()
            self.preprocessor._setup_replacement_dict(replacement_fp)
        else:
            self.preprocessor = Preprocessor()

    def create_connection(self) -> Cursor:
        self.connection = sqlite3.connect(self.fp)
        self.connected = True
        self.cursor = self.connection.cursor()

    def close_connection(self):
        # self.connection.close()
        pass

    def get_tables(self):
        self.create_connection()
        self.cursor.execute(
            """
        SELECT name FROM sqlite_master 
        WHERE type='table';
        """
        )

        result = self.cursor.fetchall()

        self.close_connection()
        return [x[0] for x in result]

    def join_runs_by_experiment(self, df: pd.DataFrame):
        experiment_summaries = (
            df.groupby("experiment_name")["content"]
            .apply(lambda x: " ".join(x.dropna().astype(str)))
            .reset_index()
        )
        return experiment_summaries

    def preprocess_and_save_files(self, df: pd.DataFrame, col_to_save: str):

        folder_path = Path(self.training_file_path)
        folder_path.mkdir(parents=True, exist_ok=True)

        for index, row in df.iterrows():
            experiment_name = row["experiment_name"]
            content_cleaned = self.preprocessor.process_string(
                row[col_to_save], replace_abbrv=self.replace_abbrv
            )
            with open(f"{folder_path}/{experiment_name}.txt", "w") as f:
                f.write(content_cleaned)

    def create_pattern_matching_sql(self):
        patterns = self._query_info["parameter_patterns"]
        patterns_list = patterns.split(",")
        pattern_sql = "LOWER(TRIM(content)) LIKE "
        pattern_sql += " OR LOWER(TRIM(content)) LIKE ".join(patterns_list)
        return pattern_sql

    def remove_html(self):
        self.create_connection()
        html_tags = self._query_info["html_tags"]
        drop_table_query = f""" DROP TABLE IF EXISTS logbook_reduced;"""
        create_table_query = f"""
        CREATE TABLE logbook_reduced AS
            SELECT * from logbook
            WHERE TAGS NOT IN ({html_tags}) OR tags IS NULL;"""
        self.cursor.execute(drop_table_query)
        self.cursor.execute(create_table_query)
        self.close_connection()

    def process_elogs(self):  # all elogs
        self.create_connection()
        self.remove_html()

        query = "SELECT * FROM logbook_reduced"

        df_elogs = pd.read_sql(query, self.connection)
        experiment_summaries = self.join_runs_by_experiment(df_elogs)

        self.preprocess_and_save_files(experiment_summaries, "content")
        self.close_connection()
        return experiment_summaries

    def process_experiment_descriptions(self):  # descriptions
        self.create_connection()
        query = "SELECT * FROM experiments"
        df_experiments = pd.read_sql(query, self.connection)

        self.preprocess_and_save_files(df_experiments, "description")
        for index, row in df_experiments.iterrows():
            experiment_id = row["experiment_name"]
            description = row["description"]
            clean = self.preprocessor.process_string(description)

            with open(f"../private_data/descriptions/{experiment_id}.txt", "w") as f:
                f.write(clean)
        self.close_connection()

    def process_experiment_elog_parameters(self):  # params
        param_tags = self._query_info["parameter_tags"]
        pattern_sql = self.create_pattern_matching_sql()

        self.create_connection()

        if "logbook_reduced" not in self.get_tables():
            self.remove_html()

        query_params_drop = """DROP TABLE IF EXISTS logbook_parameters"""
        query_params = f"""CREATE TABLE logbook_parameters AS
        SELECT * FROM logbook_reduced
        WHERE tags IN ({param_tags}
        )
        OR {pattern_sql}
        ;"""

        self.cursor.execute(query_params_drop)
        self.cursor.execute(query_params)

        query = "SELECT * FROM logbook_parameters"
        df_logbook_params = pd.read_sql(query, self.connection)

        experiment_summaries = self.join_runs_by_experiment(df_logbook_params)

        self.preprocess_and_save_files(experiment_summaries, "content")

        self.close_connection()

    def process_experiment_elog_commentary(self):  # commentary
        self.create_connection()
        if "logbook_reduced" not in self.get_tables():
            self.remove_html()

        param_tags = self._query_info["parameter_tags"]
        pattern_sql = self.create_pattern_matching_sql()

        query_commentary_drop = """DROP TABLE IF EXISTS logbook_commentary"""

        query_commentary = f"""CREATE TABLE logbook_commentary AS
        SELECT * FROM logbook_reduced
        WHERE (
        tags IS NULL 
        OR 
        tags NOT IN ({param_tags})
        )
        AND NOT (
            {pattern_sql}
        );"""

        self.cursor.execute(query_commentary_drop)
        self.cursor.execute(query_commentary)

        query = "SELECT * FROM logbook_commentary"
        df_logbook_not_params = pd.read_sql(query, self.connection)

        experiment_summaries = self.join_runs_by_experiment(df_logbook_not_params)
        self.preprocess_and_save_files(experiment_summaries, "content")

        self.close_connection()


def main():
    de = SLACDatabaseDataExtractor(
        config_file="examples/config_files/slac_config_params.json"
    )
    de.process_experiment_elog_parameters()
