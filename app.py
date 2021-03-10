from flask import Flask,render_template,url_for,request
from pymongo import MongoClient
import pandas as pd 
from pandas import DataFrame
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
# ----- Configuration ------------
app = Flask(__name__)
client = MongoClient("mongodb+srv://dbUser:DBUSER@mycluster.iaeyt.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db = client.get_database('abhishek')
records = db.users
posts = list(records.find())

# print(df.head())
# print(df.columns)
# ---------- machine learning ---------------
#-------------user functions
df = DataFrame(posts)
df['index'] = df['index'].astype(int)
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
#print(df.head())
#----------------end of Machine learning ------------

# ---------- Main implementation --------
#slink = 'https://res.cloudinary.com/teknet/image/upload/v1554137119/kea3pn2t4nasveqvenqs.jpg'

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/data',methods=["POST", "GET"])
def data():
	if request.method == "POST":
		content = request.form['content']	
		cv = CountVectorizer()
		count_matrix = cv.fit_transform(df["combined_features"])
		cosine_sim = cosine_similarity(count_matrix) 
		concept_user_likes = str(content)
		movie_index = get_index_from_title(concept_user_likes)
		similar_concepts =  list(enumerate(cosine_sim[movie_index]))
		sorted_similar_concepts = sorted(similar_concepts,key=lambda x:x[1],reverse=True)
		m = list(db.users.find({"title" : concept_user_likes }))
		l = []
		i=0
		for element in sorted_similar_concepts:
			l.append(records.find_one({'title':get_title_from_index(element[0])}))
			i=i+1
			if i>2:
				break
		return render_template('data.html',posts=l,msg='success',vidlink=m)	
if __name__ == "__main__":
    app.run(debug=True)