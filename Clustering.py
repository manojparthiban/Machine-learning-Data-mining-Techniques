import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("orbit classification for prediction.csv")

def recalculate_clusters(X, centroids, k):
    clusters = {}
    for i in range(k):
        clusters[i] = []
    for data in X:
        euc_dist = []
        for j in range(k):
            euc_dist.append(np.linalg.norm(data[1] - centroids[j][1]))
        clusters[euc_dist.index(min(euc_dist))].append(data)
    return clusters

def recalculate_centroids(clusters, k):
    centroids = []
    for i in range(k):
        xval = yval = 0
        for j in clusters[i]:
            xval +=  j[0]
            yval +=  j[1]
        if (len(clusters[i]) != 0):        
            xval /= len(clusters[i])
            yval /= len(clusters[i])
        centroids.append([xval,yval])
    return centroids

def plot_clusters(X, clusters, centroids):
    for i in range(len(X)):  
        j=0
        for b in clusters.values():
            for c in b:
                if X[i][0] == c[0] and X[i][1] == c[1]:
                    plt.plot(X[i][0],X[i][1],color[j])
            j += 1
    for i in range(len(centroids)):
        plt.plot(centroids[i][0],centroids[i][1],'ro')
    plt.show()

col = df.columns.tolist()
plt.scatter(x=df.index,y=df[col[0]])
plt.title(col[0])
plt.show()

color = ['bo','go','yo','mo']
k = 4
clusters = {}
X = []
for i in range(len(df.index)):
    X.append([df.index[i],df[col[0]][i]])

for i in range(k):
    clusters[i] = []
    
for i in range(k):
    ul = int((len(X)/k)*i)
    ll = int((len(X)/k)*(i+1))
    for j in range(ul,ll):
        clusters[i].append(X[j])

centroids = recalculate_centroids(clusters, k)

for a in range(10):
    plot_clusters(X, clusters, centroids)
    centroids = recalculate_centroids(clusters, k)
    clusters = recalculate_clusters(X,centroids,k)
