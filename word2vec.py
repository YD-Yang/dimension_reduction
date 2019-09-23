

import pandas as pd
from gensim.models import Word2Vec

#modify data if list in vertical format and not sentence 
df['value'] = df['value'].apply(lambda x: x if x in x_list else 'rare')
df_group = df.sort_values(by = ['id'])
df_group = df_group.groupby("id")["value"].apply(list)

df_group = df_group.to_frame()
df_group['id'] = df_group.index


w2v_model = Word2Vec(
       med_group['value'],
        size=128,
        window=30,
        min_count=3,
        workers=10)
