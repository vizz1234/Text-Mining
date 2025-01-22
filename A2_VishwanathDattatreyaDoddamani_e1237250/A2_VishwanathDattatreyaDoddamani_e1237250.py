import numpy as np



def euclidean_distances(X, Y):
    """
    Learn vocabulary and idf from training set.    
    X, Y are 2d numpy arrays
    """
    
    num_x, num_y = X.shape[0], Y.shape[0]
    distances = np.zeros((num_x, num_y))

    #########################################################################################
    ### Your code starts here ###############################################################        
    for i in range(num_x):
        for j in range(num_y):
            distances[i, j] = np.linalg.norm(X[i] - Y[j])    
    ### Your code ends here #################################################################
    #########################################################################################            

    return distances




class MyTfidfVectorizer:
    
    ####################################################################
    # IMPORTANT: We assume that all the processing was done before
    ####################################################################
    
    def __init__(self):
        pass
    
    
    def fit(self, raw_documents):
        """
        Learn vocabulary and idf from training set.
        """
        self.D = len(raw_documents)
        self.idf = {}

        #########################################################################################
        ### Your code starts here ###############################################################
        for i, document in enumerate(raw_documents):
            terms = document
            terms = list(set(terms))
            for term in terms:
                self.idf[term] = self.idf.get(term, 0) + 1
        for k,v in self.idf.items():
            self.idf[k] = np.log10(self.D/self.idf[k])
        ### Your code ends here #################################################################
        #########################################################################################
    
        # features = sorted vocabulary (in line with sklearn.feature_extraction.text.TfidfVectorizer)
        self.feature_names = np.unique(list(self.idf.keys()))
        
        # Our document-term matrix will be indexed by integers, so we need to be able to map
        # between the features (i.e., the words/term/tokens) and their unique matrix index
        self.feature2idx, self.idx2feature = {}, {}
        for idx, feature in enumerate(self.feature_names):
            self.feature2idx[feature] = idx
            self.idx2feature[idx] = feature
        
        return self
        
    
    def transform(self, raw_documents):
        """
        Transform documents to document-term matrix.
        (NOTE: in the lecture, we use the term-document matrix)
        """
        
        tfidf_matrix = np.zeros((len(raw_documents), len(self.feature_names)))

        #########################################################################################
        ### Your code starts here ###############################################################
        for i, document in enumerate(raw_documents):
            terms = document
            for term in terms:
                if term not in self.feature2idx.keys():
                    continue
                termInd = self.feature2idx[term]
                tfidf_matrix[i][termInd] = tfidf_matrix[i][termInd] + 1
        
        for i, document in enumerate(raw_documents):
            terms = list(set(document))
            for term in terms:
                if term not in self.feature2idx.keys():
                    continue
                termInd = self.feature2idx[term]
                tfidf_matrix[i][termInd] = (1 + np.log10(tfidf_matrix[i][termInd])) * (self.idf[term])
        ### Your code ends here #################################################################
        #########################################################################################
        
        return tfidf_matrix        
    
    
    def fit_transform(self, raw_documents):
        """
        Learn vocabulary and idf, return document-term matrix.
        """
        # Call fit() to compute idf values and vocabulary
        self.fit(raw_documents)
        # Call transform() and return the result
        return self.transform(raw_documents)

    
    def get_feature_names_out(self):
        return self.feature_names
    
    
    

    
    
class MyKNeighborsClassifier():
    
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        
        
    def fit(self, X, y):
        """
        "Remember" the training data + create label mapping
        """
        self.X_train = X
        self.y_train = y
        self.create_label_maps(y)
        return self
        
        
    def create_label_maps(self, y):
        self.label2idx, self.idx2label, self.num_labels = {}, {}, -1
        # Get the set of unique labels
        labels = np.unique(y)
        # Assign each label an index from 0..(#classes-1)
        for idx, label in enumerate(labels):
            self.label2idx[label] = idx
            self.idx2label[idx] = label
        
        self.y_train = np.asarray([ self.label2idx[l] for l in y ])        
                
        
    def predict(self, X):
            
        # Calculate all pairwise distances between the unseen data X and the training data
        distances = euclidean_distances(X, self.X_train)
        
        num_test = distances.shape[0]
        
        # Initialize the array with the final predictions with 0
        y_pred = np.zeros(num_test)
        
        #########################################################################################
        ### Your code starts here ###############################################################        
        for i in range(num_test):
            # Sort distances for the i-th test example and get the indices of the sorted distances
            sortedIndices = np.argsort(distances[i])
            KNNIndices = sortedIndices[:self.n_neighbors]
            kNNLabels = self.y_train[KNNIndices]
            labelCounts = np.bincount(kNNLabels)
            y_pred[i] = np.argmax(labelCounts)        
        ### Your code ends here #################################################################
        #########################################################################################            
        
        # We convert the integer labels back to the original labels before returning the predictions
        return np.array([ self.idx2label[idx] for idx in y_pred ])
        
        
    
    

class MyAgglomerativeClustering():

    def __init__(self, linkage="average"):
        self.linkage = linkage
        self.clustering_hierarchy = None
        

    def merge_clusters(self, c1, c2):
        return np.concatenate((c1, c2), axis=0)
    
    
    def calculate_cluster_distance(self, c1, c2):

        cluster_distance = np.inf
        
        # Calculate all pairwise distances between the clusters
        distances = euclidean_distances(c1, c2)

        #########################################################################################
        ### Your code starts here ###############################################################    
        if self.linkage == "single":
            cluster_distance = np.min(distances)
        if self.linkage == "complete":
            cluster_distance =  np.max(distances)
        if self.linkage == "average":
            cluster_distance = np.mean(distances)       

        ### Your code ends here #################################################################
        #########################################################################################          

        return cluster_distance    
    
    
    def generate_distance_matrix(self, clustering):

        # We initialize the distance matrix with all elements set to INFINITY
        # This means by default that any 2 clusters are veeery far apart :)
        distance_matrix = np.full((len(clustering), len(clustering)), np.Inf)

        #########################################################################################
        ### Your code starts here ###############################################################  
        lenClustering = len(clustering)
        for i in range(lenClustering):
            for j in range(i+1, lenClustering):
                distance_matrix[i, j] = self.calculate_cluster_distance(clustering[i], clustering[j])
        ### Your code ends here #################################################################
        #########################################################################################              

        return distance_matrix    
    
    
    def fit(self, X):

        # List of all clusterings at each hierarchy level
        self.clustering_hierarchy = []

        # In the initial clustering, all data points form their own cluster
        clustering = np.split(X, X.shape[0])

        # Add initial cluster as first solution to the cluster hierarchy
        self.clustering_hierarchy.append(clustering)


        # Combine the nearest clusters as long as there a more than 1 cluster
        while len(self.clustering_hierarchy[-1]) > 1:

            # Represent the new clustering after merging the pair of closest clusters
            next_clustering = None

            #########################################################################################
            ### Your code starts here ###############################################################      
            clusterDistanceMatrix = self.generate_distance_matrix(clustering)
            minIndices = np.unravel_index(np.argmin(clusterDistanceMatrix), clusterDistanceMatrix.shape)
            rowInd = minIndices[0]
            colInd = minIndices[1]
            if rowInd > colInd:
                next_clustering = [clustering[i] for i in range(len(clustering)) if i != rowInd]
                next_clustering[colInd] = self.merge_clusters(clustering[rowInd], clustering[colInd])
            else:
                next_clustering = [clustering[i] for i in range(len(clustering)) if i != colInd]
                next_clustering[rowInd] = self.merge_clusters(clustering[rowInd], clustering[colInd])                
            clustering = next_clustering
            ### Your code ends here #################################################################
            #########################################################################################      

            # Add new cluster distribution to the cluster hierarchy
            self.clustering_hierarchy.append(next_clustering)

        return self
    
        
