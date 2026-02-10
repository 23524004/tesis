import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from typing import List, Dict, Optional

import yake
import re


class TopicLabeler:
    """
    Graph-based topic labeling using YAKE + TextRank.
    """

    def __init__(
        self,
        text_columns: List[str] = ["title", "abstract"],
        language: str = "en",
        top_n_keywords: int = 10,
        min_docs_per_cluster: int = 5,
        yake_params: Optional[Dict] = None,
        textrank_params: Optional[Dict] = None,
        score_combination: str = "weighted",  # {"yake", "textrank", "weighted"}
        alpha: float = 0.5,
        verbose: bool = True,
        # random_state: int = 42,
    ):
        self.text_columns = text_columns
        self.language = language
        self.top_n_keywords = top_n_keywords
        self.min_docs_per_cluster = min_docs_per_cluster
        self.score_combination = score_combination
        self.alpha = alpha
        self.verbose = verbose
        # self.random_state = random_state

        self.yake_params = yake_params or {
            "n": 3,
            "dedupLim": 0.9,
            "top": 50,
            "features": None,
        }

        self.textrank_params = textrank_params or {
            "window_size": 5
        }

        self._topic_info = []
        self._keyword_table = []

        # np.random.seed(self.random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "TopicLabeler":
        self._validate_input(df)

        self._topic_info.clear()
        self._keyword_table.clear()

        for cluster_id, cluster_df in df.groupby("cluster"):
            if len(cluster_df) < self.min_docs_per_cluster:
                if self.verbose:
                    print(f"[SKIP] Cluster {cluster_id} (n={len(cluster_df)})")
                continue

            if self.verbose:
                print(f"[LABELING] Cluster {cluster_id} (n={len(cluster_df)})")

            result = self.label_cluster(cluster_df)

            self._topic_info.append({
                "topic_id": cluster_id,
                "topic_label": result["topic_label"],
                "keywords": result["keywords"],
                "n_docs": len(cluster_df),
            })

            keyword_df = result["keyword_scores"]
            keyword_df["topic_id"] = cluster_id
            self._keyword_table.append(keyword_df)

        return self

    def label_cluster(self, cluster_df: pd.DataFrame) -> Dict:
        texts = self._prepare_cluster_text(cluster_df)

        yake_scores = self._extract_yake_candidates(texts)
        if not yake_scores:
            return {
                "topic_label": "",
                "keywords": [],
                "keyword_scores": pd.DataFrame()
            }

        graph = self._build_textrank_graph(texts, list(yake_scores.keys()))
        textrank_scores = self._run_textrank(graph)

        keyword_df = self._combine_scores(yake_scores, textrank_scores)
        keyword_df = keyword_df.sort_values("final_score", ascending=False)

        top_keywords = keyword_df.head(self.top_n_keywords)["keyword"].tolist()
        topic_label = top_keywords[0] if top_keywords else ""

        return {
            "topic_label": topic_label,
            "keywords": top_keywords,
            "keyword_scores": keyword_df.reset_index(drop=True),
        }

    def get_topic_info(self) -> pd.DataFrame:
        return pd.DataFrame(self._topic_info)

    def get_keyword_table(self) -> pd.DataFrame:
        if not self._keyword_table:
            return pd.DataFrame()
        return pd.concat(self._keyword_table, ignore_index=True)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        topic_map = (
            self.get_topic_info()
            .set_index("topic_id")["topic_label"]
            .to_dict()
        )
        df = df.copy()
        df["topic_label"] = df["cluster"].map(topic_map)
        return df

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _validate_input(self, df: pd.DataFrame) -> None:
        required_cols = {"cluster"} | set(self.text_columns)
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _prepare_cluster_text(self, cluster_df: pd.DataFrame) -> List[str]:
        texts = []
        for _, row in cluster_df.iterrows():
            parts = []
            for col in self.text_columns:
                if pd.notna(row[col]):
                    parts.append(str(row[col]))
            text = ". ".join(parts).strip()
            if text:
                texts.append(text)
        return texts

    def _extract_yake_candidates(self, texts: List[str]) -> Dict[str, float]:
        kw_extractor = yake.KeywordExtractor(
            lan=self.language,
            **self.yake_params
        )

        scores = {}
        for text in texts:
            for kw, score in kw_extractor.extract_keywords(text):
                kw = kw.lower().strip()
                if kw not in scores or score < scores[kw]:
                    scores[kw] = score

        return scores

    def _build_textrank_graph(
        self,
        texts: List[str],
        candidates: List[str]
    ) -> nx.Graph:
        window_size = self.textrank_params.get("window_size", 5)
        graph = nx.Graph()

        candidate_set = set(candidates)
        tokenized_docs = [self._simple_tokenize(t) for t in texts]

        for tokens in tokenized_docs:
            for i, token in enumerate(tokens):
                if token not in candidate_set:
                    continue
                for j in range(i + 1, min(i + window_size, len(tokens))):
                    other = tokens[j]
                    if other not in candidate_set:
                        continue
                    if token != other:
                        graph.add_edge(token, other, weight=1.0)

        return graph

    def _run_textrank(self, graph: nx.Graph) -> Dict[str, float]:
        if graph.number_of_nodes() == 0:
            return {}
        scores = nx.pagerank(graph, weight="weight")
        return scores

    def _combine_scores(
        self,
        yake_scores: Dict[str, float],
        textrank_scores: Dict[str, float]
    ) -> pd.DataFrame:
        keywords = set(yake_scores) | set(textrank_scores)

        rows = []
        for kw in keywords:
            y = yake_scores.get(kw, np.nan)
            t = textrank_scores.get(kw, 0.0)
            rows.append((kw, y, t))

        df = pd.DataFrame(
            rows,
            columns=["keyword", "yake_score", "textrank_score"]
        )

        # Normalize
        if df["yake_score"].notna().any():
            df["yake_norm"] = 1 - (
                (df["yake_score"] - df["yake_score"].min()) /
                (df["yake_score"].max() - df["yake_score"].min() + 1e-9)
            )
        else:
            df["yake_norm"] = 0.0

        if df["textrank_score"].max() > 0:
            df["textrank_norm"] = (
                df["textrank_score"] / df["textrank_score"].max()
            )
        else:
            df["textrank_norm"] = 0.0

        if self.score_combination == "yake":
            df["final_score"] = df["yake_norm"]
        elif self.score_combination == "textrank":
            df["final_score"] = df["textrank_norm"]
        else:
            df["final_score"] = (
                self.alpha * df["textrank_norm"]
                + (1 - self.alpha) * df["yake_norm"]
            )

        return df.sort_values("final_score", ascending=False)

    @staticmethod
    def _simple_tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if len(t) > 2]
