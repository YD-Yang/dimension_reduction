from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(design_dta.transpose())

#find the features that could be reduced 
threshold = 0.5
comb_list = []
lens = len(sim)
drop_list = []
for i in range(lens):
    if cols_dict[i] not in drop_list:      
        comb = [cols_dict[i]]
        for j in range((i+1),lens):
            if cols_dict[j] not in drop_list:
                if sim[i, j] >= threshold:
                    comb.append(cols_dict[j])                    
        if len(comb) > 2:
            comb_list.append(comb)    
            for item in comb:
                drop_list.append(item)
                
                
                
#create a mapping of combined factors 
comb_dict = {}
for i in range(len(comb_list)):
    comb_dict['comb'+ str(i)] = comb_list[i]                
