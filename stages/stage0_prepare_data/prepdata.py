import re
import unicodedata
import pandas as pd


class ArxivPreprocessor:
    """
    Preprocessor khusus untuk abstrak penelitian ArXiv.

    Tujuan utama class ini:
    - Membersihkan noise non-linguistik (LaTeX, simbol matematika, citation)
    - Menjaga struktur dan makna semantik teks
    - Menyiapkan teks agar optimal untuk embedding berbasis SBERT
      (sentence-transformers/all-MiniLM-L6-v2)

    Catatan:
    - Tidak melakukan stopword removal
    - Tidak melakukan stemming atau lemmatization
    - Preprocessing dibuat ringan agar tidak merusak konteks ilmiah
    """

    def __init__(
        self,
        min_words: int = 40,
        max_words: int = 500
    ):
        """
        Inisialisasi parameter filtering dokumen.

        Parameters
        ----------
        min_words : int
            Jumlah kata minimum sebuah abstrak agar dianggap informatif.
            Abstrak terlalu pendek cenderung menjadi noise dalam clustering.

        max_words : int
            Jumlah kata maksimum sebuah abstrak.
            Abstrak terlalu panjang dapat mendominasi embedding dan centroid cluster.
        """
        self.min_words = min_words
        self.max_words = max_words
        

    def clean_text(self, text: str) -> str:
        """
        Pipeline pembersihan teks untuk satu dokumen abstrak.

        Urutan pembersihan:
        1. Hapus LaTeX command
        2. Hapus simbol matematika
        3. Hapus sitasi
        4. Normalisasi teks

        Method ini digunakan sebelum embedding.
        """
        if not isinstance(text, str):
            return ""

        text = self._abstract_logic_formulas(text)
        text = self._remove_latex(text)
        text = self._remove_math_symbols(text)
        text = self._remove_citations(text)
        text = self._normalize_text(text)
        text = self._normalize_numbers(text)
        text = self._remove_latex_citations(text)
        return text

    def filter_by_length(self, df: pd.DataFrame, text_col: str):
        """
        Memfilter dokumen berdasarkan panjang (jumlah kata).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame berisi dokumen abstrak.

        text_col : str
            Nama kolom yang berisi teks abstrak.

        Returns
        -------
        pd.DataFrame
            DataFrame yang hanya berisi dokumen
            dengan panjang kata dalam rentang yang ditentukan.
        """
        word_counts = df[text_col].str.split().str.len()

        return df[
            (word_counts >= self.min_words) &
            (word_counts <= self.max_words)
        ].reset_index(drop=True)

    def preprocess_dataframe(self, df: pd.DataFrame, text_col: str):
        """
        Menjalankan preprocessing lengkap pada DataFrame.

        Langkah:
        - Salin DataFrame (menghindari side-effect)
        - Bersihkan teks abstrak
        - Filter dokumen ekstrem (terlalu pendek / panjang)

        Method ini adalah entry point utama
        sebelum proses embedding dan document clustering.
        """
        df = df.copy()
        df[text_col] = df[text_col].apply(self.clean_text)
        df = self.filter_by_length(df, text_col)
        return df

    
    # -------------------------------
    # HELPER
    # -------------------------------
    def _remove_latex(self, text: str) -> str:
        """
        Menghapus command LaTeX umum yang sering muncul pada abstrak ArXiv.

        Contoh yang dihapus:
        - \\theta
        - \\sqrt{...}
        - \\frac{a}{b}

        Alasan:
        Model embedding tidak memahami notasi LaTeX dan
        menganggapnya sebagai token tidak bermakna (noise).
        """
        text = re.sub(r"\\[a-zA-Z]+(\{.*?\})?", " ", text)
        return text

    def _remove_math_symbols(self, text: str) -> str:
        """
        Menghapus simbol matematika yang tersisa setelah pembersihan LaTeX.

        Contoh simbol:
        =, <, >, ±, ×, ∑, √, ∞, ≈

        Alasan:
        Simbol ini jarang berkontribusi pada makna topik,
        tetapi dapat mengganggu embedding similarity.
        """
        text = re.sub(r"[=<>±×∑√∈∂∞≈~]", " ", text)
        return text

    def _remove_citations(self, text: str) -> str:
        """
        Menghapus pola sitasi sederhana yang sering muncul di abstrak.

        Contoh:
        - [12]
        - [1, 3, 5]
        - (Smith et al., 2020)

        Alasan:
        Sitasi tidak membawa informasi topikal,
        tetapi sering berulang dan menambah noise semantik.
        """
        text = re.sub(r"\[[0-9, ]+\]", " ", text)
        text = re.sub(r"\([A-Za-z\s]+et al\.,?\s*\d{4}\)", " ", text)
        return text

    def _normalize_text(self, text: str) -> str:
        """
        Normalisasi teks agar konsisten dan stabil untuk embedding.

        Langkah:
        - Unicode normalization (NFKC)
        - Lowercasing
        - Perbaikan escape character (contoh: \\% → %)
        - Normalisasi spasi

        Alasan:
        Variasi unicode dan whitespace dapat mempengaruhi tokenisasi
        dan kualitas embedding.
        """
        text = unicodedata.normalize("NFKC", text)
        # text = text.lower()
        text = text.replace("\\%", "%")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _normalize_numbers(self, text: str) -> str:
        """
        Mengganti angka dengan token generik agar
        informasi kuantitatif tidak hilang sepenuhnya.
        """
        text = re.sub(r"\b\d+(\.\d+)?\b", " NUM ", text)
        return text
    
    def _remove_latex_citations(self, text: str) -> str:
        """
        Menghapus citation berbasis LaTeX seperti \\cite{...} atau \\cite ...
        """
        text = re.sub(r"\\cite\{.*?\}", " ", text)
        text = re.sub(r"\\cite\s+[A-Za-z0-9_]+", " ", text)
        return text

    def _abstract_logic_formulas(self, text: str) -> str:
        """
        Mengabstraksikan notasi logika / teori kompleksitas
        menjadi token semantik yang stabil untuk embedding.
    
        Aman untuk dokumen non-formal:
        - Jika pola tidak ditemukan → teks tidak berubah
        """
    
        replacements = {
            # Complexity theory
            r"\bP\s*=\s*NP\b": " p_np_problem ",
            r"\bP\s*≠\s*NP\b": " p_np_problem ",
    
            # SAT family
            r"\bXSAT\b": " xsat_problem ",
            r"\bSAT\b": " sat_problem ",
            r"one[- ]in[- ]three": " one_in_three ",
    
            # Logic symbols
            r"[∧∨¬⊙→←↔≡]": " logical_operator ",
    
            # Common logical objects
            r"[φψ]": " boolean_formula ",
            r"O\([^)]+\)": " time_complexity ",
        }
    
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
    
        return text
