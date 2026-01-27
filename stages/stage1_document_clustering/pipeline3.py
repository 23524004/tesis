import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from itertools import product
import random
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

import umap
import hdbscan
import matplotlib.pyplot as plt


class DocumentClusteringPipeline:
    """
    End-to-end document clustering pipeline:
    DataFrame / List[str]
    → SBERT Embeddings
    → UMAP
    → HDBSCAN
    """

    # --------------------------------------------------
    # DEFAULT PARAMETERS
    # --------------------------------------------------
    DEFAULT_UMAP_PARAMS = {
        "n_neighbors": [15],
        "n_components": [5],
        "min_dist": [0.0],
        "metric": ["cosine"]
    }

    DEFAULT_HDBSCAN_PARAMS = {
        "min_cluster_size": [10],
        "metric": ["euclidean"],
        "cluster_selection_method": ["eom"]
    }

    # --------------------------------------------------
    # INIT
    # --------------------------------------------------
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_device: str = "cpu",
        random_state: int = 42
    ):
        self.model_name = model_name
        self.embedding_device = embedding_device
        self.random_state = random_state

        # raw storage
        self.df = None
        self.documents = None     # Abstract (list[str])
        self.titles = None
        self.ids = None

        # results
        self.embeddings = None
        self.reduced_embeddings = None
        self.labels = None
        self.probabilities = None
        self._hdbscan_clusterer = None

        # metrics
        self.umap_params = None
        self.hdbscan_params = None
        self.n_clusters = None
        self.noise_ratio = None
        self.silhouette = None
        self.dbcv = None
        self.gridsearch_log = None

    # --------------------------------------------------
    # FIT
    # --------------------------------------------------
    def fit(
        self,
        documents: Union[List[str], pd.DataFrame],
        text_column: Optional[str] = None,
        title_column: Optional[str] = None,
        id_column: Optional[str] = None,
        umap_params: Optional[Dict] = None,
        hdbscan_params: Optional[Dict] = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        verbose: bool = True
    ):

        # ------------------------------------
        # 0. HANDLE INPUT
        # ------------------------------------
        if isinstance(documents, pd.DataFrame):
            if text_column is None:
                raise ValueError(
                    "text_column must be provided when documents is a DataFrame"
                )

            self.df = documents.reset_index(drop=True)

            self.documents = (
                self.df[text_column]
                .astype(str)
                .tolist()
            )

            self.titles = (
                self.df[title_column].astype(str).tolist()
                if title_column and title_column in self.df.columns
                else None
            )

            self.ids = (
                self.df[id_column].astype(str).tolist()
                if id_column and id_column in self.df.columns
                else None
            )

        else:
            # backward compatibility
            self.df = None
            self.documents = documents
            self.titles = None
            self.ids = None

        if verbose:
            print("=" * 60)
            print("DOCUMENT CLUSTERING PIPELINE")
            print(f"Documents: {len(self.documents)}")
            print("=" * 60)

        # ------------------------------------
        # PARAM VALIDATION
        # ------------------------------------
        umap_params = self._validate_or_default_params(
            umap_params,
            self.DEFAULT_UMAP_PARAMS,
            "UMAP",
            verbose
        )

        hdbscan_params = self._validate_or_default_params(
            hdbscan_params,
            self.DEFAULT_HDBSCAN_PARAMS,
            "HDBSCAN",
            verbose
        )

        # ------------------------------------
        # 1. EMBEDDINGS
        # ------------------------------------
        if verbose:
            print("\n[1/3] Generating SBERT embeddings...")

        model = SentenceTransformer(
            self.model_name,
            device=self.embedding_device
        )

        embeddings = model.encode(
            self.documents,
            batch_size=batch_size,
            show_progress_bar=verbose
        )

        if normalize_embeddings:
            embeddings = normalize(embeddings)

        self.embeddings = embeddings

        if verbose:
            print(f"✔ Embeddings shape: {embeddings.shape}")

        # ------------------------------------
        # 2. GRID SEARCH UMAP + HDBSCAN
        # ------------------------------------
        if verbose:
            print("\n[2/3] GridSearch UMAP + HDBSCAN")

        umap_keys, umap_values = zip(*umap_params.items())
        hdb_keys, hdb_values = zip(*hdbscan_params.items())

        best_score = -1
        best_result = None
        logs = []

        total_runs = (
            len(list(product(*umap_values))) *
            len(list(product(*hdb_values)))
        )

        run_id = 1

        for umap_combo in product(*umap_values):
            umap_cfg = dict(zip(umap_keys, umap_combo))

            reducer = umap.UMAP(
                random_state=self.random_state,
                **umap_cfg
            )

            X_umap = reducer.fit_transform(embeddings)

            for hdb_combo in product(*hdb_values):
                hdb_cfg = dict(zip(hdb_keys, hdb_combo))

                if verbose:
                    print(
                        f"Run {run_id}/{total_runs} | "
                        f"UMAP={umap_cfg} | HDBSCAN={hdb_cfg}"
                    )

                clusterer = hdbscan.HDBSCAN(
                    gen_min_span_tree=True,
                    prediction_data=True,
                    **hdb_cfg
                )

                labels = clusterer.fit_predict(X_umap)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise_ratio = np.mean(labels == -1)

                if n_clusters <= 1:
                    run_id += 1
                    continue

                try:
                    sil = silhouette_score(X_umap, labels)
                except Exception:
                    sil = -1

                logs.append({
                    "umap_params": umap_cfg,
                    "hdbscan_params": hdb_cfg,
                    "n_clusters": n_clusters,
                    "noise_ratio": noise_ratio,
                    "silhouette": sil
                })

                if sil > best_score:
                    best_score = sil
                    best_result = {
                        "X_umap": X_umap,
                        "labels": labels,
                        "probabilities": clusterer.probabilities_,
                        "clusterer": clusterer,
                        "umap_params": umap_cfg,
                        "hdbscan_params": hdb_cfg,
                        "n_clusters": n_clusters,
                        "noise_ratio": noise_ratio,
                        "silhouette": sil,
                        "dbcv": clusterer.relative_validity_
                    }

                run_id += 1

        if best_result is None:
            raise RuntimeError("No valid clustering found")

        # ------------------------------------
        # 3. STORE BEST RESULT
        # ------------------------------------
        self.reduced_embeddings = best_result["X_umap"]
        self.labels = best_result["labels"]
        self.probabilities = best_result["probabilities"]
        self._hdbscan_clusterer = best_result["clusterer"]

        self.umap_params = best_result["umap_params"]
        self.hdbscan_params = best_result["hdbscan_params"]
        self.n_clusters = best_result["n_clusters"]
        self.noise_ratio = best_result["noise_ratio"]
        self.silhouette = best_result["silhouette"]
        self.dbcv = best_result["dbcv"]

        self.gridsearch_log = pd.DataFrame(logs)

        if verbose:
            print("\n" + "=" * 60)
            print("BEST RESULT")
            print(f"UMAP params     : {self.umap_params}")
            print(f"HDBSCAN params  : {self.hdbscan_params}")
            print(f"Clusters        : {self.n_clusters}")
            print(f"Noise ratio     : {self.noise_ratio:.2f}")
            print(f"Silhouette      : {self.silhouette:.4f}")
            print(f"DBCV            : {self.dbcv:.4f}")
            print("=" * 60)

    # --------------------------------------------------
    # INSPECT CLUSTERS
    # --------------------------------------------------
    def inspect_clusters(
        self,
        n_samples: int = 5,
        n_clusters: Optional[int] = None,
        include_noise: bool = False,
        random_state: Optional[int] = None,
        show_abstract: bool = True,
        abstract_max_chars: int = 100
    ) -> str:
        """
        Inspect clusters with formatted output.
    
        Format:
        === Cluster k (N documents) ===
        1. [ID] Title
            Abstract snippet (optional)
        """
    
        if self.labels is None:
            raise RuntimeError("Run fit() before inspecting clusters")
    
        rng = random.Random(
            self.random_state if random_state is None else random_state
        )
    
        # ------------------------------------
        # GROUP DOCUMENTS BY CLUSTER
        # ------------------------------------
        cluster_docs = defaultdict(list)
    
        for i, label in enumerate(self.labels):
            cluster_docs[label].append({
                "id": self.ids[i] if self.ids else None,
                "title": self.titles[i] if self.titles else None,
                "abstract": self.documents[i]
            })
    
        # ------------------------------------
        # SELECT CLUSTERS
        # ------------------------------------
        labels_available = list(cluster_docs.keys())
    
        if not include_noise and -1 in labels_available:
            labels_available.remove(-1)
    
        if n_clusters is not None and n_clusters < len(labels_available):
            selected_labels = rng.sample(labels_available, n_clusters)
        else:
            selected_labels = sorted(labels_available)
    
        # ------------------------------------
        # BUILD OUTPUT
        # ------------------------------------
        output_lines = []
    
        for label in selected_labels:
            docs = cluster_docs[label]
            cluster_size = len(docs)
    
            output_lines.append(
                f"=== Cluster {label} ({cluster_size} documents) ==="
            )
    
            sampled_docs = rng.sample(
                docs,
                min(n_samples, cluster_size)
            )
    
            for i, doc in enumerate(sampled_docs, 1):
                doc_id = doc["id"] or "N/A"
                title = doc["title"] or "(No Title)"
    
                output_lines.append(
                    f"{i}. [{doc_id}] {title}"
                )
    
                if show_abstract:
                    abstract = (
                        doc["abstract"]
                        .replace("\n", " ")
                        .strip()
                    )
    
                    if len(abstract) > abstract_max_chars:
                        abstract = (
                            abstract[:abstract_max_chars].rstrip() + "..."
                        )
    
                    output_lines.append(
                        f"    {abstract}"
                    )
    
                output_lines.append("")  # blank line between samples
    
        return "\n".join(output_lines)



    # --------------------------------------------------
    # TO DATAFRAME
    # --------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:

        if any(v is None for v in [
            self.documents,
            self.embeddings,
            self.reduced_embeddings,
            self.labels
        ]):
            raise RuntimeError("Pipeline not fully fitted. Run fit() first.")
        
        n_docs = len(self.documents)
        if not (
            n_docs == self.embeddings.shape[0] ==
            self.reduced_embeddings.shape[0] ==
            len(self.labels)
        ):
            raise ValueError(
                "documents, embeddings, reduced_embeddings, "
                "and labels must have the same length"
            )
            
        df = pd.DataFrame({
            "cluster": self.labels,
            "probability": self.probabilities,
            "embedding": self.embeddings.tolist(),
            "embedding_umap": self.reduced_embeddings.tolist(),
            "abstract": self.documents
        })

        if self.titles is not None:
            df["title"] = self.titles

        if self.ids is not None:
            df["id"] = self.ids

        return df

    # ------------------------------------------------------------------
    # [TAMBAHAN] HDBSCAN CONDENSED TREE
    # ------------------------------------------------------------------
    def visualize_hdbscan_tree(
        self,
        select_clusters: bool = True,
        figsize: tuple = (8, 6)
    ):
        """
        Visualize HDBSCAN condensed tree (cluster stability).
        """

        if self._hdbscan_clusterer is None:
            raise RuntimeError("Run fit() before visualizing HDBSCAN tree")

        plt.figure(figsize=figsize)

        self._hdbscan_clusterer.condensed_tree_.plot(
            select_clusters=select_clusters
        )

        plt.title("HDBSCAN Condensed Tree (Cluster Stability)")
        plt.xlabel("Clusters")
        plt.ylabel("Lambda (Density)")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # [TAMBAHAN] HDBSCAN MST
    # ------------------------------------------------------------------
    def visualize_hdbscan_mst(
        self,
        figsize: tuple = (6, 4)
    ):
        """
        Visualize HDBSCAN minimum spanning tree.
        """

        if self._hdbscan_clusterer is None:
            raise RuntimeError("Run fit() before visualizing HDBSCAN MST")

        plt.figure(figsize=figsize)

        self._hdbscan_clusterer.minimum_spanning_tree_.plot()

        plt.title("HDBSCAN Minimum Spanning Tree")
        plt.show()

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------
    def _ensure_list(self, v):
        return v if isinstance(v, list) else [v]

    def _validate_or_default_params(
        self,
        params: Optional[Dict],
        default_params: Dict,
        name: str,
        verbose: bool
    ) -> Dict:
        if params is None:
            if verbose:
                print(f"⚠ {name} params not provided → using default")
            return default_params.copy()

        return {k: self._ensure_list(v) for k, v in params.items()}
























# CARA PAKAI
# topic_model = DocumentClusteringPipeline()

# topic_model.fit(
#     documents=documents,
#     umap_params=None,      # otomatis default
#     hdbscan_params=None,   # otomatis default
#     verbose=True
# )


# CARA PAKAI CONTOH GRID SEARCH UMAP SAJA
# pipeline.fit(
#     documents=docs,
#     umap_params={
#         "n_neighbors": [10, 15, 30],
#         "n_components": [5],
#         "min_dist": [0.0, 0.1],
#         "metric": ["cosine"]
#     },
#     hdbscan_params=None,  # default
#     verbose=True
# )

# # CARA PAKAI CONTOH GRID SEARCH HDBSCAN SAJA
# pipeline.fit(
#     documents=docs,
#     umap_params=None,  # default
#     hdbscan_params={
#         "min_cluster_size": [5, 10, 20],
#         "metric": ["euclidean"],
#         "cluster_selection_method": ["eom", "leaf"]
#     },
#     verbose=True
# )

# # CARA PAKAI CONTOH GRID SEARCH PENUH (UMAP + HDBSCAN)
# pipeline.fit(
#     documents=docs,
#     umap_params={
#         "n_neighbors": [10, 15, 30],
#         "n_components": [5, 10],
#         "min_dist": [0.0, 0.1],
#         "metric": ["cosine"]
#     },
#     hdbscan_params={
#         "min_cluster_size": [5, 10, 20],
#         "metric": ["euclidean"],
#         "cluster_selection_method": ["eom"]
#     },
#     verbose=True
# )

# # CARA PAKAI CONTOH CAMPURAN (AMAN & RINGKAS)
# pipeline.fit(
#     documents=docs,
#     umap_params={
#         "n_neighbors": [15, 30],
#         "n_components": 5,      # ← boleh
#         "min_dist": [0.0, 0.1],
#         "metric": "cosine"      # ← boleh
#     },
#     hdbscan_params={
#         "min_cluster_size": [10, 20],
#         "cluster_selection_method": "eom"
#     },
#     verbose=True
# )

# MELIHAT HASIL
# pipeline.gridsearch_log.sort_values(
#     by="silhouette",
#     ascending=False
# ).head(10)



# CARA CEPAT
# df = pd.DataFrame({
#     "document": pipeline.documents,
#     "cluster": pipeline.labels,
#     "probability": pipeline.probabilities
# })

# # Ambil hanya dokumen yang confident
# df_high_conf = df[
#     (df.cluster != -1) &
#     (df.probability > 0.7)
# ]
