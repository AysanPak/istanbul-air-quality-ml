# 04_literature_analysis.py
"""
Performs a comprehensive machine learning analysis on a corpus of academic
research papers related to air quality.

This script processes text files extracted from papers, applies NLP techniques,
and uses unsupervised learning to uncover themes, relationships, and anomalies.

Steps:
1.  Loads paper metadata from a CSV and full texts from a directory.
2.  Preprocesses text data: cleaning, tokenization, stopword removal, and stemming.
3.  Creates a TF-IDF matrix to represent papers numerically.
4.  Performs t-SNE analysis (with PCA pre-reduction) to visualize paper relationships.
5.  Performs Local Outlier Factor (LOF) analysis to identify anomalous or novel papers.
6.  Performs K-Means clustering to group papers into thematic clusters.
7.  Generates and saves visualizations and analysis reports.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import warnings
from pathlib import Path

# Scikit-learn and NLTK imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

warnings.filterwarnings('ignore')

# --- Configuration ---
# TODO: Update these paths to match your project structure
# Directory containing the full text of papers (e.g., 'W12345.txt')
EXTRACTED_TEXTS_DIR = "data/research_data/texts/"
# The CSV file from OpenAlex containing paper metadata
METADATA_CSV_FILE = "data/research_data/openalex_metadata.csv"
# Directory where analysis results (CSVs, reports) will be saved
OUTPUT_DIR = "results/literature_analysis"
# Directory where visualizations (PNGs) will be saved
VISUALIZATIONS_DIR = "results/visualizations"


class ResearchPapersAnalyzer:
    def __init__(self):
        """Initializes the analyzer with necessary components."""
        self.papers_df = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.feature_names = []
        self.stemmer = PorterStemmer()
        self._setup_nltk()
        self._setup_stopwords()
        
        # Create output directories if they don't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

    def _setup_nltk(self):
        """Downloads required NLTK data if not already present."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading required NLTK data (punkt, stopwords)...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)

    def _setup_stopwords(self):
        """Creates a custom stopword list for academic air quality papers."""
        english_stopwords = set(stopwords.words('english'))
        custom_stopwords = {
            'abstract', 'introduction', 'conclusion', 'results', 'discussion',
            'method', 'methods', 'study', 'research', 'paper', 'article',
            'figure', 'table', 'fig', 'tab', 'et', 'al', 'doi', 'elsevier',
            'copyright', 'journal', 'vol', 'pp', 'page', 'author'
        }
        self.stopwords = english_stopwords.union(custom_stopwords)

    def load_and_preprocess(self):
        """Loads paper texts and metadata, then preprocesses them."""
        print("--- Step 1: Loading and Preprocessing Research Papers ---")
        
        # Load metadata
        metadata_df = pd.read_csv(METADATA_CSV_FILE)
        metadata_df['paper_id'] = metadata_df['id'].str.replace('https://openalex.org/', '', regex=False)
        id_to_metadata = metadata_df.set_index('paper_id').to_dict('index')

        # Load and process text files
        papers_data = []
        text_files = list(Path(EXTRACTED_TEXTS_DIR).glob("*.txt"))
        
        for text_file in text_files:
            paper_id = text_file.stem
            if paper_id in id_to_metadata:
                with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                processed_text = self._preprocess_text(content)
                if len(processed_text.strip()) > 200: # Ensure meaningful content
                    papers_data.append({
                        'paper_id': paper_id,
                        'text': processed_text,
                        **id_to_metadata[paper_id]
                    })
        
        self.papers_df = pd.DataFrame(papers_data)
        print(f"Successfully loaded and processed {len(self.papers_df)} papers.\n")

    def _preprocess_text(self, text):
        """Cleans and standardizes a single paper's text."""
        # Normalize and remove non-essential elements
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S*@\S*\s?', '', text)
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text) # Citations like [1, 2]
        text = re.sub(r'\s+', ' ', text) # Excess whitespace
        
        # Tokenize, stem, and remove stopwords
        tokens = [
            self.stemmer.stem(word) for word in nltk.word_tokenize(text.lower())
            if word.isalpha() and len(word) > 2 and word not in self.stopwords
        ]
        return ' '.join(tokens)

    def run_tfidf(self, max_features=5000):
        """Creates a TF-IDF matrix from the processed texts."""
        print("--- Step 2: Creating TF-IDF Matrix ---")
        texts = self.papers_df['text'].tolist()
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=3,
            max_df=0.9,
            ngram_range=(1, 2) # Use unigrams and bigrams
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        print(f"TF-IDF matrix created with shape: {self.tfidf_matrix.shape}\n")

    def run_tsne_and_visualize(self):
        """Performs t-SNE for dimensionality reduction and visualization."""
        print("--- Step 3: Performing t-SNE Analysis and Visualization ---")
        
        # PCA pre-reduction for better t-SNE performance and stability
        pca = PCA(n_components=50, random_state=42)
        tfidf_pca = pca.fit_transform(self.tfidf_matrix.toarray())
        
        n_samples = self.tfidf_matrix.shape[0]
        perplexity = min(30, n_samples - 1)
        
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(tfidf_pca)
        
        self.papers_df['tsne_x'] = tsne_results[:, 0]
        self.papers_df['tsne_y'] = tsne_results[:, 1]

        # Visualization
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            self.papers_df['tsne_x'], self.papers_df['tsne_y'],
            c=np.log1p(self.papers_df['cited_by_count']),
            cmap='viridis', alpha=0.7
        )
        plt.colorbar(scatter, label='log(1 + Citation Count)')
        plt.title('t-SNE Visualization of Research Papers')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        
        save_path = os.path.join(VISUALIZATIONS_DIR, "tsne_visualization.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"t-SNE visualization saved to {save_path}\n")

    def run_lof_anomaly_detection(self):
        """Finds anomalous papers using Local Outlier Factor."""
        print("--- Step 4: Running LOF Anomaly Detection ---")
        
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        outlier_preds = lof.fit_predict(self.tfidf_matrix.toarray())
        
        self.papers_df['is_outlier'] = (outlier_preds == -1)
        self.papers_df['lof_score'] = lof.negative_outlier_factor_
        
        outliers_df = self.papers_df[self.papers_df['is_outlier']].sort_values('lof_score')
        print(f"Found {len(outliers_df)} anomalous papers.")
        print("Top 5 most anomalous papers:")
        for _, row in outliers_df.head(5).iterrows():
            print(f"  - (Score: {row['lof_score']:.3f}) {row['title']}")
        print("")
        
    def run_kmeans_clustering(self, n_clusters=5):
        """Groups papers into thematic clusters using K-Means."""
        print(f"--- Step 5: Running K-Means Clustering (K={n_clusters}) ---")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self.papers_df['cluster'] = kmeans.fit_predict(self.tfidf_matrix)

        print("Top terms for each cluster:")
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        for i in range(n_clusters):
            top_terms = [self.feature_names[ind] for ind in order_centroids[i, :10]]
            print(f"Cluster {i}: {', '.join(top_terms)}")
        print("")
    
    def save_results(self):
        """Saves the final DataFrame with all analysis results to a CSV."""
        print("--- Step 6: Saving All Analysis Results ---")
        save_path = os.path.join(OUTPUT_DIR, "full_literature_analysis.csv")
        self.papers_df.to_csv(save_path, index=False)
        print(f"Full results DataFrame saved to {save_path}")

def main():
    """Main function to execute the full analysis pipeline."""
    analyzer = ResearchPapersAnalyzer()
    analyzer.load_and_preprocess()
    analyzer.run_tfidf()
    analyzer.run_tsne_and_visualize()
    analyzer.run_lof_anomaly_detection()
    analyzer.run_kmeans_clustering()
    analyzer.save_results()
    print("\nLiterature analysis complete!")


if __name__ == "__main__":
    main()