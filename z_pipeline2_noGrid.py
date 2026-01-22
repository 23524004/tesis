import random
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import hdbscan
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer


class DocumentClusteringPipeline:
    """
    Pipeline mirip BERTopic:
    SBERT Embeddings → UMAP → HDBSCAN → Sort clusters by size
    """

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

        # Data placeholders
        self.documents = None
        self.embeddings = None
        self.reduced_embeddings = None
        self.labels = None
        self.probabilities = None
        self._hdbscan_clusterer = None

        # Cluster info
        self.topic_mapping = None
        self.topic_sizes = None

    # --------------------------------------------------
    # FIT METHOD
    # --------------------------------------------------
    def fit(
        self,
        documents: List[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        verbose: bool = True
    ):
        """
        Fit the pipeline: Embeddings → UMAP → HDBSCAN → Sort clusters
        """

        # 1️⃣ SBERT Embeddings
        if verbose:
            print("[1/4] Generating embeddings...")
        model = SentenceTransformer(self.model_name, device=self.embedding_device)
        embeddings = model.encode(documents, batch_size=batch_size, show_progress_bar=verbose)
        if normalize_embeddings:
            embeddings = normalize(embeddings)
        self.embeddings = embeddings
        self.documents = documents
        if verbose:
            print(f"✔ Embeddings shape: {embeddings.shape}")

        # 2️⃣ UMAP Reduction
        if verbose:
            print("[2/4] Reducing dimensionality with UMAP...")
        reducer = umap.UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=self.random_state
        )
        X_umap = reducer.fit_transform(embeddings)
        self.reduced_embeddings = X_umap
        if verbose:
            print(f"✔ UMAP reduced shape: {X_umap.shape}")

        # 3️⃣ HDBSCAN Clustering
        if verbose:
            print("[3/4] Clustering with HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=10,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
            gen_min_span_tree=True
        )
        labels = clusterer.fit_predict(X_umap)
        probabilities = clusterer.probabilities_
        self._hdbscan_clusterer = clusterer

        # 3a️⃣ Probabilistic threshold for noise (like BERTopic)
        threshold = 0.015
        labels = np.where(probabilities < threshold, -1, labels)
        self.probabilities = probabilities

        # 4️⃣ Sort clusters by size
        sorted_labels, topic_mapping, topic_sizes = self._sort_topic_ids_by_size(labels)
        self.labels = sorted_labels
        self.topic_mapping = topic_mapping
        self.topic_sizes = topic_sizes

        if verbose:
            print("[4/4] Clusters sorted by size:")
            print("Topic mapping (old → new):", topic_mapping)
            print("Topic sizes:", topic_sizes)

    # --------------------------------------------------
    # INSPECT CLUSTERS
    # --------------------------------------------------
    def inspect_clusters(
        self,
        n_samples: int = 5,
        n_clusters: Optional[int] = None,
        include_noise: bool = False,
        random_state: Optional[int] = None
    ) -> str:
        """
        Sample documents per cluster
        """
        if self.documents is None or self.labels is None:
            raise RuntimeError("Run fit() first.")

        rng = random.Random(self.random_state if random_state is None else random_state)
        from collections import defaultdict
        cluster_docs = defaultdict(list)
        for doc, label in zip(self.documents, self.labels):
            cluster_docs[label].append(doc)

        # Select clusters
        labels_available = list(cluster_docs.keys())
        if not include_noise and -1 in labels_available:
            labels_available.remove(-1)
        if n_clusters is not None and n_clusters < len(labels_available):
            selected_labels = rng.sample(labels_available, n_clusters)
        else:
            selected_labels = sorted(labels_available)

        # Sample documents
        output_lines = []
        for label in sorted(selected_labels):
            output_lines.append(f"=== Cluster {label} ===")
            docs = cluster_docs[label]
            sampled_docs = rng.sample(docs, min(n_samples, len(docs))) if n_samples > 0 else docs
            for i, doc in enumerate(sampled_docs, start=1):
                doc_clean = doc.strip().replace("\n", " ")
                output_lines.append(f"{i}. {doc_clean}")
            output_lines.append("")
        return "\n".join(output_lines)

    # --------------------------------------------------
    # TO DATAFRAME
    # --------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        if any(v is None for v in [self.documents, self.embeddings, self.reduced_embeddings, self.labels]):
            raise RuntimeError("Pipeline not fitted yet.")

        df = pd.DataFrame({
            "document": self.documents,
            "cluster": self.labels,
            "embedding": self.embeddings.tolist(),
            "embedding_umap": self.reduced_embeddings.tolist()
        })
        return df

    # --------------------------------------------------
    # HDBSCAN Visualizations
    # --------------------------------------------------
    def visualize_hdbscan_tree(self, select_clusters: bool = True, figsize: tuple = (10, 6)):
        if self._hdbscan_clusterer is None:
            raise RuntimeError("Run fit() first.")
        plt.figure(figsize=figsize)
        self._hdbscan_clusterer.condensed_tree_.plot(select_clusters=select_clusters)
        plt.title("HDBSCAN Condensed Tree")
        plt.xlabel("Clusters")
        plt.ylabel("Lambda (Density)")
        plt.tight_layout()
        plt.show()

    def visualize_hdbscan_mst(self, figsize: tuple = (8, 6)):
        if self._hdbscan_clusterer is None:
            raise RuntimeError("Run fit() first.")
        plt.figure(figsize=figsize)
        self._hdbscan_clusterer.minimum_spanning_tree_.plot()
        plt.title("HDBSCAN MST")
        plt.show()

    # --------------------------------------------------
    # HELPER: Sort clusters by size
    # --------------------------------------------------
    def _sort_topic_ids_by_size(self, labels: np.ndarray):
        labels = np.asarray(labels)
        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        topic_sizes_old = dict(zip(unique, counts))
        sorted_topics = sorted(topic_sizes_old.items(), key=lambda x: x[1], reverse=True)

        topic_mapping = {-1: -1}
        for new_id, (old_id, _) in enumerate(sorted_topics):
            topic_mapping[old_id] = new_id

        new_labels = np.array([topic_mapping[label] for label in labels])
        topic_sizes_new = {topic_mapping[old]: size for old, size in topic_sizes_old.items()}

        return new_labels, topic_mapping, topic_sizes_new
