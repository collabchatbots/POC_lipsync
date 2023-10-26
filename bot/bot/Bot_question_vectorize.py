import os
import tqdm
import configparser
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
# from .Bot_Questions_MW_ import Bot_Questions_MW

os.environ["OPENAI_API_KEY"] = "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"

def get_documents():
    loader = DirectoryLoader('bot/Bot_Questions_MW', glob="**/*.txt")
    data = loader.load()
    print("len",len(data))
    return data

def create_db_question():
    persist_directory = "Bot_Questions_db"

    if not os.path.exists(persist_directory):
        db_data = get_documents()

        text_splitter = TokenTextSplitter(chunk_size=1800, chunk_overlap=0)
        db_doc = text_splitter.split_documents(db_data)
        
        embeddings = OpenAIEmbeddings()
        vectordb = FAISS.from_documents(db_doc, embeddings)
        vectordb.save_local(persist_directory)
        print("Vector database created.")
    else:
        print("Vector database already exists. Skipping creation.")




# def genActions(APIKEY,path,model_name,Function) :
     
#     try :   
            
#             EmbeddingFunction = Function(api_key=APIKEY,model_name=model_name)
       
#             path = os.path.join(os.getcwd(),'Data')
            
#             for filename in os.listdir(path) :

#                 with open(os.path.join(path,filename),'r') as f :
#                      documents  = f.read() 
#                      embeddings = EmbeddingFunction(texts=documents)
#                      yield {'file_vector':embeddings,'file_name':filename,'file_content':documents}

#     except Exception as e:
        
#            print(f'Exception Occured :\n{e}')
             

# def getConnection(ElasticSearch,cloud_id,basic_auth,count=[0]) :

#     ES = None

#     if count[0]<=0 :

#        ES = ElasticSearch(cloud_id=cloud_id,basic_auth=basic_auth)
       
#        count[0] += 1 

#        getConnection.Elastic = ES

#        return ES 

#     return ES    





# def createIndex( index , ES  ) :

#     if not ES.indices.exists(index=index) :

#        ES.indices.create(  
           
#            index=index,

#            body= {
               
#                 'settings':{ 'number_of_shards':10    } ,
                
# #    "settings":{
# #       "index":{
# #          "number_of_shards":10,
# #          "number_of_replicas":0,
# #          "knn":{
# #             "algo_param":{
# #             #    "ef_search":40,
# #                "ef_construction":40,
# #                "m":"4"
# #             }
# #          }
# #       },
# #       "knn":"true"
# #    },
#                 'mappings':{ 'properties':{   
                        
#                         'file_vector':{
#                                         'type':'dense_vector',
#                                         # 'type':'knn_vector',
#                                         'dims':384,
#                                         'index':'true',
#                                         'element_type':'float',
#                                         'similarity':'cosine',

#                                     },
                        
#                         'file_name':{ 'type':'text' },

#                         'file_content':{'type':'text'}

#                 } }
#            }

#          ) 







# config = configparser.ConfigParser()

# config.read( os.path.join(os.getcwd(),'config\\config.ini') ) 

# path       = ' '

# APIKEY     = config['chromaDB']['api_key']

# model_name = config['chromaDB']['model_name']

# Function   = HuggingFaceEmbeddingFunction



# def create_db_question() :


#     ES = getConnection(ElasticSearch=Elasticsearch,cloud_id=config['ELASTIC']['cloud_id'],basic_auth=(config['ELASTIC']['user'],config['ELASTIC']['password']))
    

#     createIndex(index='test-vec-index',ES=ES)


#     with tqdm.tqdm(total=35,desc="Bulk indexing progress") as progress :
         
#          for success , info in streaming_bulk(client=ES,index='test-vec-index',actions=genActions(APIKEY=APIKEY,path=path,model_name=model_name,Function=Function)) :

#              if not success :

#                 print(f'reason : {info} \n')

#              progress.update(1)    

