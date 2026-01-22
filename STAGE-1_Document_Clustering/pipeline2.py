import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from itertools import product
import random

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

import umap
import hdbscan
import matplotlib.pyplot as plt


class DocumentClusteringPipeline:
    """
    End-to-end document clustering pipeline:
    Embeddings → UMAP → HDBSCAN
    (single-call, self-contained)
    """

    # --------------------------------------------------
    # DEFAULT PARAMETERS (GRID-READY)
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

        self.documents = None
        self.embeddings = None
        self.reduced_embeddings = None
        self.labels = None
        self._hdbscan_clusterer = None

        self.umap_params = None
        self.hdbscan_params = None
        self.n_clusters = None
        self.noise_ratio = None
        self.silhouette = None
        self.dbcv = None
        self.gridsearch_log = None


    # --------------------------------------------------
    # METHOD
    # --------------------------------------------------
    def fit(
        self,
        documents: List[str],
        umap_params: Optional[Dict] = None,
        hdbscan_params: Optional[Dict] = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Full pipeline:
        Documents → SBERT Embeddings → UMAP → HDBSCAN (grid search)
        """

        # --------------------------------------------------
        # 0. STORE & VALIDATE PARAMS
        # --------------------------------------------------
        self.documents = documents

        if verbose:
            print("=" * 60)
            print("DOCUMENT CLUSTERING PIPELINE")
            print(f"Documents: {len(documents)}")
            print("=" * 60)

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

        # --------------------------------------------------
        # 1. EMBEDDINGS (SBERT)
        # --------------------------------------------------
        if verbose:
            print("\n[1/3] Generating SBERT embeddings...")

        model = SentenceTransformer(
            self.model_name,
            device=self.embedding_device
        )

        embeddings = model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=verbose
        )

        if normalize_embeddings:
            embeddings = normalize(embeddings)

        self.embeddings = embeddings

        if verbose:
            print(f"✔ Embeddings shape: {embeddings.shape}")

        # --------------------------------------------------
        # 2. GRID SEARCH UMAP + HDBSCAN
        # --------------------------------------------------
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

            if verbose:
                print(f"\n→ UMAP params: {umap_cfg}")

            reducer = umap.UMAP(
                random_state=self.random_state,
                **umap_cfg
            )
            X_umap = reducer.fit_transform(embeddings)

            for hdb_combo in product(*hdb_values):
                hdb_cfg = dict(zip(hdb_keys, hdb_combo))

                if verbose:
                    print(
                        f"  HDBSCAN params: {hdb_cfg} "
                        f"[{run_id}/{total_runs}]"
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
                    if verbose:
                        print("    ✘ Skip (≤1 cluster)")
                    run_id += 1
                    continue

                try:
                    sil = silhouette_score(X_umap, labels)
                except Exception:
                    sil = -1

                if verbose:
                    print(
                        f"    ✔ clusters={n_clusters}, "
                        f"noise={noise_ratio:.2f}, "
                        f"silhouette={sil:.4f}"
                    )

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


        # --------------------------------------------------
        # 3. STORE BEST RESULT
        # --------------------------------------------------
        self.reduced_embeddings = best_result["X_umap"]
        self.labels = best_result["labels"]
        self.probabilities = best_result["probabilities"]
        self._hdbscan_clusterer = best_result["clusterer"]

        self.labels_original = best_result["labels"]          # debugging

        self.umap_params = best_result["umap_params"]
        self.hdbscan_params = best_result["hdbscan_params"]
        self.n_clusters = best_result["n_clusters"]           # self.n_clusters = len(self.topic_sizes)
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
        # 4. SORT TOPIC IDS BY CLUSTER SIZE
        # --------------------------------------------------
        sorted_labels, topic_mapping, topic_sizes = \
            self._sort_topic_ids_by_size(self.labels)
        
        self.labels = sorted_labels
        self.topic_mapping = topic_mapping
        self.topic_sizes = topic_sizes
        
        if verbose:
            print("\n[3/3] Remapping by cluster size...")
            print("Topic mapping (old → new):", topic_mapping)
            print("Topic sizes:", topic_sizes)

    
        # return best_result


    # ------------------------------------------------------------------
    # [TAMBAHAN] INSPECTION
    # ------------------------------------------------------------------
    def inspect_clusters(
        self,
        n_samples: int = 5,
        n_clusters: Optional[int] = None,
        include_noise: bool = False,
        random_state: Optional[int] = None
    ) -> str:
        """
        Sample representative documents from randomly selected clusters.

        Parameters
        ----------
        n_samples : int
            Number of documents sampled per cluster
        n_clusters : int or None
            Number of clusters to sample randomly.
            If None, all clusters are shown.
        include_noise : bool
            Whether to include noise cluster (-1)
        random_state : int or None
            Random seed for reproducibility

        Returns
        -------
        str
            Formatted text showing sample documents per cluster
        """

        if self.documents is None or self.labels is None:
            raise RuntimeError("Run fit() before inspecting clusters")

        if len(self.documents) != len(self.labels):
            raise ValueError("documents and labels must have the same length")

        rng = random.Random(
            self.random_state if random_state is None else random_state
        )

        from collections import defaultdict
        cluster_docs = defaultdict(list)

        for doc, label in zip(self.documents, self.labels):
            cluster_docs[label].append(doc)

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
        # SAMPLE DOCUMENTS
        # ------------------------------------
        output_lines = []

        for label in sorted(selected_labels):
            output_lines.append(f"=== Cluster {label} ===")

            docs = cluster_docs[label]
            sampled_docs = (
                rng.sample(docs, min(n_samples, len(docs)))
                if n_samples > 0
                else docs
            )

            for i, doc in enumerate(sampled_docs, start=1):
                doc_clean = doc.strip().replace("\n", " ")
                output_lines.append(f"{i}. {doc_clean}")

            output_lines.append("")  # blank line between clusters

        return "\n".join(output_lines)


    # ------------------------------------------------------------------
    # [TAMBAHAN] TO DATAFRAME
    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """
        Combine clustering results into a single pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with document, cluster, embedding, and UMAP embedding
        """

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
            "document": self.documents,
            "cluster": self.labels,
            "embedding": self.embeddings.tolist(),
            "embedding_umap": self.reduced_embeddings.tolist()
        })

        return df



    # ------------------------------------------------------------------
    # [TAMBAHAN] HDBSCAN CONDENSED TREE
    # ------------------------------------------------------------------
    def visualize_hdbscan_tree(
        self,
        select_clusters: bool = True,
        figsize: tuple = (10, 6)
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
        figsize: tuple = (8, 6)
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


    

    # -------------------------------
    # HELPER
    # -------------------------------
    def _ensure_list(self, v):
        return v if isinstance(v, list) else [v]

    def _validate_or_default_params(
        self,
        params: Optional[Dict],
        default_params: Dict,
        name: str,
        verbose: bool
    ) -> Dict:
        if params is None or len(params) == 0:
            if verbose:
                print(f"⚠ {name} params not provided → using default")
            return default_params.copy()

        return {k: self._ensure_list(v) for k, v in params.items()}


    def _sort_topic_ids_by_size(self, labels: np.ndarray):
        """
        Reorder cluster labels by descending cluster size.
        Noise label (-1) is preserved.
    
        Returns
        -------
        new_labels : np.ndarray
            Remapped labels
        topic_mapping : Dict[int, int]
            Old topic id → new topic id
        topic_sizes : Dict[int, int]
            New topic id → size
        """
        labels = np.asarray(labels)
    
        # Count topic sizes (exclude noise)
        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        topic_sizes_old = dict(zip(unique, counts))
    
        # Sort topics by size (descending)
        sorted_topics = sorted(
            topic_sizes_old.items(),
            key=lambda x: x[1],
            reverse=True
        )
    
        # Build mapping: old_topic → new_topic
        topic_mapping = {-1: -1}
        for new_id, (old_id, _) in enumerate(sorted_topics):
            topic_mapping[old_id] = new_id
    
        # Apply mapping
        new_labels = np.array([topic_mapping[label] for label in labels])
    
        # New topic sizes
        topic_sizes_new = {
            topic_mapping[old]: size
            for old, size in topic_sizes_old.items()
        }
    
        return new_labels, topic_mapping, topic_sizes_new




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

