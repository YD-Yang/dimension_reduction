
from sklearn.feature_extraction.text import TfidfVectorizer  


X_value= X.code.values
# tfidfconverter=joblib.load(modelpath+'20190830T1705tfidfconverter.pkl')
tfidfconverter = TfidfVectorizer(max_features= 500, min_df=100, max_df=0.9, dtype =  np.float32)  
tfidfconverter = tfidfconverter.fit(X_value)
X_value = tfidfconverter_rx.transform(X_value).toarray().astype('float16')
X_value = pd.DataFrame(X_value)
X_value.columns = tfidfconverter.get_feature_names()
X_value = pd.concat([X_value, X[keycol].reset_index()], axis = 1)
X_value = X_value.drop('index', axis = 1)
