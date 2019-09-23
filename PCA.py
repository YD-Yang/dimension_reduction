
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#fit PCA and plot 
pca_fit = PCA().fit(df_pca)
plt.plot(np.cumsum(pca_fit.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

#fit with a fixed number of components 
pca_fit_n = PCA(n_components = 200).fit(df_pca)

#components weights 
pca_fit_n.components_

#use it to select the most important features 
X_pc = pca_fit_n.transform(df_pca)

# number of components
n_pcs= pca_fit_n.components_.shape[0]

# get the index of the most important feature on EACH component i.e. largest absolute value
# using LIST COMPREHENSION HERE
most_important = [np.abs(pca_fit_n.components_[i]).argmax() for i in range(n_pcs)]

initial_feature_names = ['a','b','c', 'd']

# get the names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

# using LIST COMPREHENSION HERE AGAIN
dic = {'PC{}'.format(i+1): most_important_names[i] for i in range(n_pcs)}
