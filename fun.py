# json_data = dumps(posts)
# with open("data.json", "w") as file:
#     file.write(json_data)
# with open('data.json') as json_file: 
#     data = json.load(json_file) 
# data_file = open('work_data.csv', 'w') 
# csv_writer = csv.writer(data_file) 
# count = 0
# for emp in data: 
# 	if count == 0: 
# 		header = emp.keys() 
# 		csv_writer.writerow(header) 
# 		count += 1

# 	csv_writer.writerow(emp.values()) 
  
# data_file.close() 
import pandas as pd 
from pandas import DataFrame
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("rec_system.csv")
#-------------user functions
def get_title_from_index(index):
    	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
#------------------
features = ['keywords','title']
for feature in features:
	df[feature] = df[feature].fillna('')
def combine_features(row):
	try:
		return row['keywords'] +" "+row['title']
	except:
		print("Error:", row)

df["combined_features"] = df.apply(combine_features,axis=1)
print(df.head())
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix) 
concept_user_likes = "HTML Advance"
movie_index = get_index_from_title(concept_user_likes)
similar_concepts =  list(enumerate(cosine_sim[movie_index]))
sorted_similar_concepts = sorted(similar_concepts,key=lambda x:x[1],reverse=True)
i=0
for element in sorted_similar_concepts:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>6:
        break
# file = open("work_data.csv","r+")
# file.truncate(0)
# file.close()
# file = open("data.json","r+")
# file.truncate(0)
# file.close()