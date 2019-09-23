from sklearn.manifold import TSNE

med_tsne = TSNE(n_components=3, random_state=123).fit_transform(med_dummies)
