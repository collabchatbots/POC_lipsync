import os
import re
import csv
import tqdm
import json
import configparser
import pandas as pd
import numpy as np
import elasticsearch
from elasticsearch_dsl import Search
from elasticsearch import Elasticsearch , helpers
from elasticsearch.helpers import streaming_bulk
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction 


def genActions(APIKEY,path,model_name,Function) :
     
    try :   
            
            EmbeddingFunction = Function(api_key=APIKEY,model_name=model_name)
       
            # path = os.path.join(os.getcwd(),'Data')

            path = 'C:\\Users\\Parv\\OneDrive\\Desktop\\ElasticSearch\\Data'

            embeddings_ques  = EmbeddingFunction(texts="0")
            
            for filename in os.listdir(path) :

                with open(os.path.join(path,filename),'r') as f :
                     documents  = f.read() 
                    #  questions  = " ".join(documents.split('\n')[:4])
                    #  questions1 = " ".join(documents.split('\n')[5:9]) 
                     
                    #  embeddings_ques1 = EmbeddingFunction(texts=questions1)
                     embeddings_doc   = EmbeddingFunction(texts=documents)
                    #  yield {'file_vector':embeddings_doc,'file_name':filename,'file_content':documents,'question_vector':[{"vector":embeddings_ques },{"vector":embeddings_ques1 }]}
                     
                     FILENAME = filename.lower().strip().replace(".pdf","").replace(".txt","")

                     FILENAME = re.sub(r'\s+|^-','-',FILENAME)

                     yield { '_index': 'test-index-new', '_id': FILENAME , '_source': { 'file_vector': embeddings_doc ,'file_name': filename, 'file_content' :documents , 'question_vector' :[]  } } # : [{"vector":embeddings_ques}]
    except Exception as e:
        
           print(f'Exception Occured :\n{e}')
             

def getConnection(ElasticSearch,cloud_id,basic_auth,count=[0]) :

    ES = None

    if count[0]<=0 :

       ES = ElasticSearch(cloud_id=cloud_id,basic_auth=basic_auth)
       
       count[0] += 1 

       getConnection.Elastic = ES

       return ES 

    return ES    


def createIndex( index , ES  ) :

    if not ES.indices.exists(index=index) :

       ES.indices.create(  
           
           index=index,

           body= {
               
                'settings':{ 'number_of_shards':10    } ,
                
#    "settings":{
#       "index":{
#          "number_of_shards":10,
#          "number_of_replicas":0,
#          "knn":{
#             "algo_param":{
#             #    "ef_search":40,
#                "ef_construction":40,
#                "m":"4"
#             }
#          }
#       },
#       "knn":"true"
#    },
                # 'mappings':{ 'properties':{   
                        
                #         'file_vector':{
                #                         'type':'dense_vector',
                #                         # 'type':'knn_vector',
                #                         'dims':384,
                #                         'index':'true',
                #                         'element_type':'float',
                #                         'similarity':'cosine',

                #                     },
                        
                #         'file_name':{ 'type':'text' },

                #         'file_content':{'type':'text'}

                # } }

                
  "mappings": {
    "properties": {
      "file_content": {
        "type": "text"
      },
      "file_name": {
        "type": "text"
      },
      "file_vector": {
        "type": "dense_vector",
        "dims": 384,
        "index": "true",
        "similarity": "cosine"
      },

      "question_vector":
      {
          "type":"nested",
          "properties" :
          {
              "vector":
              {
                "type":"dense_vector",
                # "type":"knn_vector",
                "dims":384,
                # "similarity":"cosine",
                # "index":"true",
              }

         
          }

      },

    #   "question":
    #   {
    #       "type":"text"
    #   }
    }
  
}
    
           }

         ) 

def vecQuery(QUERY,APIKEY,model_name,Function) :

    EmbeddingFunction = Function(api_key=APIKEY,model_name=model_name)

    return EmbeddingFunction(texts=QUERY) 


def new_main():


    config = configparser.ConfigParser()

    config.read( os.path.join(os.getcwd(),'config\\config.ini') ) 

    path       = ' '

    APIKEY     = config['chromaDB']['api_key']
    
    model_name = config['chromaDB']['model_name']
    
    Function   = HuggingFaceEmbeddingFunction

    ES = getConnection(ElasticSearch=Elasticsearch,cloud_id=config['ELASTIC']['cloud_id'],basic_auth=(config['ELASTIC']['user'],config['ELASTIC']['password']))
    
    createIndex(index='test-index-new',ES=ES)


    with tqdm.tqdm(total=35,desc="Bulk indexing progress") as progress :
         
         for success , info in streaming_bulk(client=ES,index='test-index-new',actions=genActions(APIKEY=APIKEY,path=path,model_name=model_name,Function=Function)) :

             if not success :

                print(f'reason : {info} \n')

             progress.update(1)    


#     QUERY     =  "Centre Street New York City Comptroller"

#     query_vec =  vecQuery(QUERY=QUERY,APIKEY=APIKEY,model_name=model_name,Function=Function)

#     # S         =  Search(using=ES,index='question-index-2')

#     # S.query("nested",path="question_vector",query={"match":{"question_vector.vector":query_vec} }) 

#     # query = {  
#     #             "field": "file_vector",
#     #             "query_vector": query_vec,
#     #             "k": 10,
#     #             "num_candidates": 50
#     # }
              
#     # response = ES.search(index="test-vec-index", knn=query)
#     # response = S.execute()

#     # for hit in response:
#     #     print(f'hit : {hit}') 

#     # for hit in response['hits']['hits'] :
#     #     print(f'file name {hit["_source"]["file_name"]} match score : {hit["_score"]} \n') 

# #     query = {
# #     "query": {
# #         "nested": {
# #             "path": "question_vectors",
# #             "query": {
# #                 "script_score": {
# #                     "query": {"match_all": {}},
# #                     "script": {
# #                         "source": "cosineSimilarity(params.queryVector, 'question_vectors.vector') + 1.0",
# #                         "params": {"queryVector": query_vec}
# #                     }
# #                 }
# #             }
# #         }
# #     }
# # }

#     # query = {
#     #     "query": {
#     #         "knn": {
#     #             "question_vectors.vector": {
#     #                 "vector": query_vec,
#     #                 "k": 10  # Number of nearest neighbors to retrieve
#     #             }
#     #         }
#     #     }
#     # }

#     query = {
#     "query": {
#         "nested": {
#             "path": "question_vector",  # Name of the nested field
#             "query": {
#                 "function_score": {
#                     "query": {
#                         "match_all": {}
#                     },
#                     "script_score": {
#                         "script": {
#                             "source": "cosineSimilarity(params.queryVector, 'question_vector.vector') + 1.0",
#                             "params": {
#                                 "queryVector": query_vec
#                             }
#                         }
#                     }
#                 }
#             }
#         }
#     }
#    }


#     response = ES.search(index='test-index', body=query)
#     for hit in response['hits']['hits'] :
#         print(f'file name {hit["_source"]["file_name"]} match score : {hit["_score"]} \n') 

if __name__ == '__main__' :

    new_main()


