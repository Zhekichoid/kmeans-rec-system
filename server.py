from flask import Flask, request
from flask_restful import Resource, Api, reqparse
import csv
import plotly.express as px 
import pandas as pd
import spotipy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import numpy as np
import os.path

app = Flask("RecAPI")
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('id', required = True)


sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="f1084ceeed8e41bb97a4ef2b2d762eac",
                                                            client_secret="8c2ba6fe43244da6b892775ae912dc0b"))

with open('song_data.csv', 'r') as f:
    reader = csv.reader(f)
    entities = {rows[0]:list(rows[1:]) for rows in reader}
    entities.pop('id')



def write_data(data: list):
    with open('song_data.csv', 'a', newline='') as f:    
            writer = csv.writer(f)
            writer.writerow(data)


class KMeansModel():

    def __init__(self) -> None:    
        self.cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=3))])
        
        if os.path.isfile('kmeans_data.csv'):
            self.data = pd.read_csv('kmeans_data.csv')

        else:
            self.recalculate()

        fig = px.scatter(
        self.data, x='x', y='y', color='cluster', hover_data=['x', 'y', 'id'])
        fig.show()



    def recalculate(self):
        entities_data = pd.read_csv('song_data.csv')

        X = entities_data.select_dtypes(np.number)
        self.cluster_pipeline.fit(X)
        entities_data['cluster'] = self.cluster_pipeline.predict(X)

        tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
        entities_embedding = tsne_pipeline.fit_transform(X)
        
        projection = pd.DataFrame(columns=['x', 'y'], data=entities_embedding)
        projection['id'] = entities_data['id']
        projection['cluster'] = entities_data['cluster']

        self.data = projection
        self.data.to_csv('kmeans_data.csv', index=False)

    
    def get_recs(self, likes_list, n_songs=20):
        coords = []
        
        for like in likes_list:

            x = self.data[self.data["id"] == like]['x']
            y = self.data[self.data["id"] == like]['y']
            coords.append([[float(x),float(y)]])
        
        entities_center = np.mean(coords, axis=0)
        distances = cdist(entities_center, self.data[['x', 'y']], 'cosine')
        index = list(np.argsort(distances)[:, :n_songs][0])
    
        rec_songs = [str(i) for i in self.data.iloc[index, 2]]
        rec_songs = list(set(rec_songs)-set(likes_list))
        return rec_songs
        # print(index)

model = KMeansModel()
counter = 0


class Entity(Resource):

    def get(self, id):
        return entities[id], 200
    
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
        entities[id] = features
        write_data(new_entity)

        global counter
        counter += 1 
        if(counter == 9):
            model.recalculate()
            counter = 0

        return entities[id], 201
            
class EntityList(Resource):
    def get(self):        
        args = request.args
        likes = args['like_ids'].split(" ")

        print(likes)
        return model.get_recs(likes)


api.add_resource(Entity, '/entities/<id>')
api.add_resource(EntityList, '/entities')

if __name__ == "__main__":
    app.run()    


    # model.get_recs(['6HZILIRieu8S0iqY8kIKhj', '2SJmZLSvxudk0YzGN3LmxS', '7GbTJsrNPXCoOWJ5vu2vSe'])

    # to_embed = sp.audio_features(to_embed)
            
    # for i in to_embed: 
    #     features.append(list(i.values())[:11])
    
    # features = np.array(features)
    # features = list(np.mean(features,dtype=float, axis=0))

    # for i in range(len(features)):
    #     features[i] = round(features[i], 3) 


    

'''
    

'''


'''

f = open('server/song_data.csv', 'a', newline='') 


for features in result:
    to_write = list(features.values())[:11]
    print(to_write)
    writer = csv.writer(f)
    writer.writerow(to_write)

f.close() 
'''
