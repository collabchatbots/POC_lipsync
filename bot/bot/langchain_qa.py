import os
import re
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import openai
from .Bot_question_vectorize import create_db_question
from .vectorize import create_db
from transformers import pipeline


from langchain.vectorstores import Chroma
# from .scrape_data import save_data
from .vectorize import create_db

from .test_new import *
import tempfile
# os.environ["OPENAI_API_KEY"] ="TestTesT"#"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"
os.environ["OPENAI_API_KEY"] = "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"

create_db()
create_db_question()

persist_directory = 'demo_db'
embeddings = OpenAIEmbeddings()
vector_db = FAISS.load_local(persist_directory, embeddings)

persist_directory = "Bot_Questions_db"
vector_db_question = FAISS.load_local(persist_directory, embeddings)

#llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
#llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")



#os.environ['SERPAPI_API_KEY'] = "db1d211f1b1e39c3d1afcc952aec6b05c3514779dcd25f0f690912a3dd6c2648"


config = configparser.ConfigParser()

config.read( os.path.join(os.getcwd(),'bot\\config\\config.ini') ) 

path       = ' '

APIKEY     = config['chromaDB']['api_key']

model_name = config['chromaDB']['model_name']

Function   = HuggingFaceEmbeddingFunction

ES = getConnection(ElasticSearch=Elasticsearch,cloud_id=config['ELASTIC']['cloud_id'],basic_auth=(config['ELASTIC']['user'],config['ELASTIC']['password']))

# createIndex(index='question-index-3',ES=ES)

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

       
def get_answer(query:str,feedback_flag=[0],temp_path=['tmp'],query_vec = ['question vector']):

        responses = [] 

        Feedbacks = [ "Were you looking for this file: {name}" , "I think you wanted this file {name} " ,
                     
                      "Hopefully {name} might be the file you're looking for ", "my appologies, were you looking for {name} by any chance ?"]

        if feedback_flag[0] == 0:
           print('created temp')
           temp = tempfile.NamedTemporaryFile(mode="w+",delete=False)
           temp_path[0] =temp.name



        if feedback_flag[0] > 0 and feedback_flag[0] <= 4 :
           
           emotions  = classifier(query.strip())[0]

           emotions  = sorted(emotions,key=lambda k:k['score'],reverse=True)
           
           emotions_ = [ x['label'] for x in emotions[:3] ]

# [x in emotions_ and x not in ["disapproval","annoyance","anger","disgust","neutral"] for x in ["admiration","approval","gratitude","neutral"]]
           if any([x in emotions_ and x not in ["disapproval","annoyance","anger","disgust","neutral"] for x in ["admiration","approval","gratitude","neutral"]]): 
              
            #   feedback_flag[0] = 0 
              
            #   query_vector = vecQuery(QUERY=query,APIKEY=APIKEY,model_name=model_name,Function=Function) 

              script = {
    
                "source": "ctx._source.question_vector.add(params['tag'])",
                "lang": "painless",
                "params": {
                "tag": {
            
                    "vector":query_vec[0]
                }
                }
            
                      }
              
              
                    
              with open(os.path.join(temp_path[0]),"r") as f :
                   data = json.load(f)
                   if feedback_flag[0] <=4 :
                      feedback_flag[0] = feedback_flag[0] - 1 
                  #  if feedback_flag[0]==1 :
                  #     feedback_flag[0] = 3 
                   FILENAME = sorted(data.items(), key=lambda item: item[1], reverse=True)[feedback_flag[0]][0].lower().strip().replace(".pdf","").replace(".txt","")

                   FILENAME = re.sub(r'\s+|^-','-',FILENAME)
                   ES.update(index='test-index-new',id=FILENAME,script=script)
                   print('Updated file :' ,FILENAME,feedback_flag)
                    # print(data)
            #        for i in data :
            #            print("FeedBack Flag working : ",i["filename"],'\n') 
              
            #   print(responses)
              responses.append("Happy to help, thanks for your time ") 
              os.remove(temp_path[0])
              feedback_flag[0] = 0 
           else:
               with open(os.path.join(temp_path[0]),"r") as f :
                   data = json.load(f)
                    # print(data)
                #    for i in data :
                #        print("FeedBack Flag working : ",i["filename"],'\n')
                   print(feedback_flag) 
                   if feedback_flag[0]<4 :               
                      responses.append( Feedbacks[ feedback_flag[0] ].format(name=sorted(data.items(), key=lambda item: item[1], reverse=True)[feedback_flag[0]][0]) )
               feedback_flag[0] += 1
        
           if feedback_flag[0] > 4:
              
        #    os.remove(temp.name)
              os.remove(temp_path[0])
              feedback_flag[0] = 0
              responses.append("appologies for inconvenience") #needs to be changed to appologies later

          #  return responses
           return responses     

        print(f"\n feed back flag : {feedback_flag}")

        # Feedbacks = [ "Were you looking for this file: {name}" , "I think you wanted this file {name} " ,
                     
        #               "Hopefully {name} might be the file you're looking for ", "my appologies, were you looking for {name} by any chance ?"]

        if any(x in query.lower() for x in ["tell me","brief","who", "what", "when", "where", "why","which" ,"how", "is", "are", "can", "may", "will", "could","give","did","to","is","was","does"]) :

            query_vec[0] =  vecQuery(QUERY=query,APIKEY=APIKEY,model_name=model_name,Function=Function) 



            query = {
                "query": {
                    "nested": {
                        "path": "question_vector",  # Name of the nested field
                        "score_mode":"max",
                        "query": {
                            "function_score": {
                                "query": {
                                    "match_all": {}
                                },
                                "script_score": {
                                    "script": {
                                        "source": "cosineSimilarity(params.queryVector, 'question_vector.vector') + 1.0",
                                        "params": {
                                            "queryVector": query_vec[0]
                                        }
                                    }
                                }
                                            }
                                        }
                                    }
                                }
                            }
           
            query1 = {  
                "field": "file_vector",
                "query_vector": query_vec[0],
                "k": 10,
                "num_candidates": 50
                }
              
            responses_knn = ES.search(index="test-index-new", knn=query1)


            response = ES.search(index='test-index-new', body=query)

            results = {}

            for hit in response['hits']['hits'] :
                
                results[ hit["_id"] ] = hit["_score"]

            # for hit in responses_knn['hits']['hits'] :
            #     if hit["_id"] in results and results[ hit["_id"] ] < hit["_score"] :
            #        results[ hit["_id"] ] = hit["_score"]

            for hit in responses_knn['hits']['hits'] :
                if hit["_id"] not in results or results[ hit["_id"] ] < hit["_score"] :
                   results[ hit["_id"] ] = hit["_score"]


            results_sorted = sorted( [(x,y) for x,y in results.items()] , key=lambda x : x[1] , reverse=True  )    

            

            if results:
               counter = 0
               dumpz = []
            #    temp = tempfile.NamedTemporaryFile(mode="w+",delete=False) 
              #  for hit in response['hits']['hits'] :
              #      if counter < 4: 
              #           dumpz.append({"filename":hit["_source"]["file_name"] ,"score" : hit["_score"] ,"file_content":hit["_source"]["file_content"] })
                        
                      
              #           # responses.append({'filename':hit["_source"]["file_name"] ,'score' : hit["_score"]})
              #           # print(f'file name {hit["_source"]["file_name"]} match score : {hit["_score"]} \n') 
              #           # yield responses[counter]
              #           # print(responses[counter])
              #           counter += 1

               print('new results',results_sorted)         
               json.dump(results,temp)         
            #    temp.flush()     
               temp.close()    
               counter = 0
               with open(os.path.join(temp.name),"r") as f :
                    data = json.load(f)
                    # print(data)
                    for i in data :
                        print(i,'\n') 
                    # responses.append(f"name : { data[0]['filename'] } score : { data[0]['score'] } ")
                    responses.append(Feedbacks[ feedback_flag[0] ].format( name=max(data, key=lambda k: data[k]) ) ) 
                    # responses.append(Feedbacks[ feedback_flag[0] ].format( name=data[0]['filename'] ) )    
            feedback_flag[0] += 1

            print("Hello this is a question")
        # if answer["content"] == 'AA':
        #     answer = choice(query)
        #     responses.extend([answer])  
        #     return responses
        # else:
        #     responses.append(answer["content"])
        # #print(answer["content"])
        #     Ques_to_user=Bot_Questions_MW(query,responses)
        #     responses.append(Ques_to_user)
        #     formatted_response = format_text_and_urls_as_links(responses)    
        else:
            responses = ["Hi","Hello"]
            # return responses
        # yield responses
        return responses  
   