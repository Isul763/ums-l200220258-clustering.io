from metaflow import FlowSpec, step
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

class ManyKMeansFlow(FlowSpec):

    @step
    def start(self):
        """Step to load the data and perform initial setup."""
        # Load the data
        file_path = 'data_group.csv'  # Update this path if needed
        self.data = pd.read_csv(file_path)['Konten'].dropna()
        print(f"Loaded {len(self.data)} messages.")
        self.next(self.preprocess)

    @step
    def preprocess(self):
        """Step to preprocess the text data."""
        # Convert the text into TF-IDF features
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = vectorizer.fit_transform(self.data)
        self.feature_names = vectorizer.get_feature_names_out()
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        self.next(self.cluster)

    @step
    def cluster(self):
        """Step to perform K-Means clustering."""
        # Perform K-Means clustering (3 clusters as suggested)
        self.n_clusters = 3
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans.fit(self.tfidf_matrix)

        # Extract cluster terms
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        self.top_terms_per_cluster = []
        for i in range(self.n_clusters):
            top_terms = [self.feature_names[ind] for ind in order_centroids[i, :3]]  # Top 3 terms
            self.top_terms_per_cluster.append(top_terms)
        print("Clustering completed.")
        self.next(self.end)

    @step
    def end(self):
        """Final step to display clustering results."""
        print("Top terms per cluster:")
        for i, terms in enumerate(self.top_terms_per_cluster):
            print(f"Cluster {i+1}: {', '.join(terms)}")

if __name__ == '__main__':
    ManyKMeansFlow()
