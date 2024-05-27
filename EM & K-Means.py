import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load the Iris dataset
df = pd.read_csv("Iris.csv")
# Define features and labels
x = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm",
"PetalWidthCm"]]
label = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = [label[c] for c in df["Species"]]
# KMeans clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3).fit(x)
kmeans_colors = np.array(["blue", "yellow", "green"])
# Plot KMeans clusters
plt.scatter(df["PetalLengthCm"], df["PetalWidthCm"],
c=kmeans_colors[kmeans.labels_])
plt.title("KMeans Clustering")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()
# Gaussian Mixture Model (GMM)
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3, random_state=0).fit(x)
gmm_prediction = gmm.predict(x)
# Plot GMM classification
plt.scatter(df["PetalLengthCm"], df["PetalWidthCm"],
c=kmeans_colors[gmm_prediction])
plt.title("GMM Classification")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()
# Calculate accuracies
from sklearn import metrics
accuracy_kmeans = metrics.accuracy_score(y, kmeans.labels_)
accuracy_gmm = metrics.accuracy_score(y, gmm_prediction)
print("Accuracy of KMeans:", accuracy_kmeans)
print("Accuracy of GMM:", accuracy_gmm)
