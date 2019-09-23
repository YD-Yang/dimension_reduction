from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=200, n_iter=20, random_state=1234)
svd.fit(X)
X1 = svd.transform(X)
