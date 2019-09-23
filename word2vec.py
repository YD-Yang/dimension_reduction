

import pandas as pd
from gensim.models import Word2Vec

#modify data if list in vertical format and not sentence 
df['value'] = df['value'].apply(lambda x: x if x in x_list else 'rare')
df_group = df.sort_values(by = ['id'])
df_group = df_group.groupby("id")["value"].apply(list)

df_group = df_group.to_frame()
df_group['id'] = df_group.index

#fit word2vector model
w2v_model = Word2Vec(
        df_group['value'],
        size=128,
        window=30,
        min_count=3,
        workers=10)

w2v_model.train(df_group['value'], total_examples=len(df_group['value']), epochs=10)
print(w2v_model)
# summarize vocabulary
words = list(w2v_model.wv.vocab)
print(len(words))

#create the dictionary from the model 
w2v_dict= dict(zip(w2v_model.wv.index2word, w2v_model.wv.syn0))

#scoring 
w2v_len = 128
df_score = df['value'].apply(lambda x: w2v_dict[x] if x in w2v_dict else[0]*w2v_len)
score_wide =pd.DataFrame(df_score.values.tolist(), index= df['id'])
columns = ['w2v_'+str(i) for i in  score_wide.columns.values]
score_wide.columns = columns
score_wide['id']= score_wide.index
score_wide.index.name = None
