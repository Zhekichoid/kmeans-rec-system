from flask import Flask, request
from flask_restful import Resource, Api, reqparse

import csv
import plotly.express as px 
import pandas as pd
import pandas.io.sql as psql
import spotipy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import numpy as np

import os
import psycopg2
from dotenv import load_dotenv 

load_dotenv()

app = Flask("RecAPI")
api = Api(app)

url = os.environ.get('DATABASE_URL')
connection = psycopg2.connect(url)

parser = reqparse.RequestParser()
parser.add_argument('id', required = True)


sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ.get('CLIENT_ID'),
                                                        client_secret=os.environ.get('CLIENT_SECRET')))

INSERT_ENTITY = """INSERT INTO features VALUES 
        (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""


class KMeansModel():

    def __init__(self) -> None:    
        self.cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=3))])
        
        kmeans_dataframe = psql.read_sql('SELECT * FROM KMEANS', connection)
        if len(kmeans_dataframe) == 0:
            self.recalculate()
        else:
            self.data = kmeans_dataframe

        #fig = px.scatter(
        #self.data, x='x', y='y', color='cluster', hover_data=['x', 'y', 'entity_id'])
        #fig.show()



    def recalculate(self):
        entities_dataframe = psql.read_sql('SELECT * FROM FEATURES', connection)

        X = entities_dataframe.select_dtypes(np.number)
        self.cluster_pipeline.fit(X)
        entities_dataframe['cluster'] = self.cluster_pipeline.predict(X)

        tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
        entities_embedding = tsne_pipeline.fit_transform(X)
        
        projection = pd.DataFrame(columns=['x', 'y'], data=entities_embedding)
        projection['entity_id'] = entities_dataframe['entity_id']
        projection['cluster'] = entities_dataframe['cluster']

        self.data = projection
        self.data.to_sql('kmeans', con=connection)

    
    def get_recs(self, likes_list, n_songs=20):
        coords = []

        
        for like in likes_list:
            try:
                x = self.data[self.data["entity_id"] == like]['x']
                y = self.data[self.data["entity_id"] == like]['y']
                coords.append([[float(x.iloc[0]),float(y.iloc[0])]])
            except IndexError:
                continue
        
        entities_center = np.mean(coords, axis=0)
        distances = cdist(entities_center, self.data[['x', 'y']], 'cosine')
        index = list(np.argsort(distances)[:, :n_songs][0])
    
        rec_songs = [str(i) for i in self.data.iloc[index, 2]]
        rec_songs = list(set(rec_songs)-set(likes_list))
        return rec_songs

model = KMeansModel()
counter = 0


class Entity(Resource):

    def get(self, id):
        with connection:
            with connection.cursor() as cursor:
                cursor.execute('SELECT * FROM FEATURES WHERE entity_id = %s', (id,))
                result = cursor.fetchone()
        
        return result, 200
    
    def post(self, id):
        args = request.args
        type = args['type']

        to_embed = []
        features = []

        if type == "track":
            new_entity = sp.audio_features([id])
            features = list(new_entity[0].values())[:11]
            

        elif type == "album":            
            result = sp.album_tracks(id, limit=10, offset=1)

            for i in result['items']:
                to_embed.append(i['id'])


        elif type == "playlist":
            result = sp.playlist_items(id, fields='items(track(id,name))', limit=10, market='US', additional_types=['track'])

            for i in result['items']:
                to_embed.append(list(i['track'].values())[0])


        elif type == "artist":
            result = sp.artist_top_tracks(id, country='US')
            
            for i in result['tracks']:
                to_embed.append(i['id'])
            
        if type != "track":
            to_embed = sp.audio_features(to_embed)
            
            for i in to_embed: 
                features.append(list(i.values())[:11])
            
            features = np.array(features)
            features = list(np.mean(features,dtype=float, axis=0))

            for i in range(len(features)):
                features[i] = round(features[i], 3) 


        new_entity = [id] + features 
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(INSERT_ENTITY, tuple(new_entity))

        global counter
        counter += 1 
        if(counter == 9):
            model.recalculate()
            counter = 0

        return {"message": f"Entity {id} successfully added."}, 201
            
class EntityList(Resource):
    def get(self):        
        args = request.args
        likes = args['like_ids'].split(" ")

        #print(likes)
        return model.get_recs(likes)


api.add_resource(Entity, '/entities/<id>')
api.add_resource(EntityList, '/entities')

if __name__ == "__main__":
    app.run()


