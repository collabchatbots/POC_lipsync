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



# def chat_with_memory(full_prompt):

   
    

#     completion = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo-16k",
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": full_prompt}
#   ]
# )

#     return (completion.choices[0].message)

# def choice(full_prompt):

   
    

#     completion = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo-16k",
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant.If user provide name then greet in better way and encorage to ask questions and problems"},
#     {"role": "user", "content": full_prompt}
#   ]
# )
#     #print(completion.choices[0].message["content"])
#     return (completion.choices[0].message["content"])   
   
#     return jsonify({"Answer":reply})



# def format_text_and_urls_as_links(messages):
#     url_pattern = r'https?://\S+'
#     email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'

#     formatted_messages = []

#     for message in messages:
#         # Replace URLs with clickable links
#         urls = re.findall(url_pattern, message)
#         for url in urls:
#             # Remove unwanted characters from the end of URLs
#             cleaned_url = re.sub(r'[.,!?()\[\]]*$', '', url)
#             link = f'<a href="{cleaned_url}" target="_blank">{cleaned_url}</a>'
#             message = message.replace(url, link)

#         # Replace email IDs with clickable mailto links
#         emails = re.findall(email_pattern, message)
#         for email in emails:
#             mailto_link = f'<a href="mailto:{email}">{email}</a>'
#             message = message.replace(email, mailto_link)

#         formatted_messages.append(message)

#     return formatted_messages

'''def format_text_and_urls_as_links(messages):
    url_pattern = r'https?://\S+'
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'

    formatted_messages = []

    for message in messages:
        # Replace URLs with clickable links
        urls = re.findall(url_pattern, message)
        for url in urls:
            link = f'<a href="{url}" target="_blank">{url}</a>'
            message = message.replace(url, link)

        # Replace email IDs with clickable mailto links
        emails = re.findall(email_pattern, message)
        for email in emails:
            mailto_link = f'<a href="mailto:{email}">{email}</a>'
            message = message.replace(email, mailto_link)

        formatted_messages.append(message)
    url_pattern = r'https?://\S+'
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    
    formatted_messages = []
    
    for message in messages:
        urls = re.findall(url_pattern, message)
    
        for url in urls:
            message = message.replace(url, f'<a href="{url}" target="_blank">{url}</a>')
        emails = re.findall(email_pattern, message)
   
        for email in emails:
            message = message.replace(email, f'<a href="mailto:{email}">{email}</a>')
        
        formatted_messages.append(message)'''
    #print(formatted_messages)
    #return formatted_messages
      

# def Bot_Questions_MW(query,response):
#     mathcing_docs = vector_db_question.similarity_search(query)
#     final_str=f'''The given Context consists of questions related to the Query.Understand the Query and Please find one question from the context only that is most relevant to the Query and do not show anything except matching question.
#     If question is not match with the given context then pick the Wrap up question from given Context only and and do not show anything except question.
#     Context--{mathcing_docs}
#     Query--{query}'''
#     response=chat_with_memory(final_str)
#     return str(response["content"])
    


       
def get_answer(query:str,feedback_flag=[0],temp_path=['tmp']):
        # temp = tempfile.NamedTemporaryFile(mode="w+",delete=False) 
        # mathcing_docs = vector_db.similarity_search(query)
        # final_str=f'''Answer the query from the given context only please make sure provide all details along with matching urls and emails and please make sure that do not provide same urls and emails id muliple times in answer.
        # These are urls format examples https://magnawavepemf.com/how-it-works/,https://magnawavepemf.com/support-center/,https://magnawavepemf.com/find-a-practitioner-step-1/,https://magnawavepemf.com/find-a-practitioner-step-1/,https://magnawavepemf.com/return-refund-policies/ etc.
        # These are emails id format examples support@magnawavepemf.com,Support@magnawavepemf.com,education@magnawavepemf.com etc.
        # If question is not match with given context then please response 'AA' only.
        # Context--{mathcing_docs}
        # Query--{query}'''
        responses = [] 
        # answer=chat_with_memory(final_str)
        # #print("get answer",answer["content"],type(answer["content"]))
        # if query.lower()=="hi" :
        #    responses = ["Hope you are doing well, how may I help you ?"]  
        Feedbacks = [ "Were you looking for this file: {name}" , "I think you wanted this file {name} " ,
                     
                      "Hopefully {name} might be the file you're looking for ", "my appologies, were you looking for {name} by any chance ?"]
        # elif "what" in query :
           
        #    responses = ["searching..."]
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
              
              query_vector = vecQuery(QUERY=query,APIKEY=APIKEY,model_name=model_name,Function=Function) 

              script = {
    
                "source": "ctx._source.question_vector.add(params['tag'])",
                "lang": "painless",
                "params": {
                "tag": {
            
                    "vector":query_vector
                }
                }
            
                      }
              
              
                    
              with open(os.path.join(temp_path[0]),"r") as f :
                   data = json.load(f)
                   if feedback_flag[0] <=4 :
                      feedback_flag[0] = feedback_flag[0] - 1 
                  #  if feedback_flag[0]==1 :
                  #     feedback_flag[0] = 3 
                   FILENAME = data[ feedback_flag[0]  ]['filename'].lower().strip().replace(".pdf","").replace(".txt","")

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
                      responses.append( Feedbacks[ feedback_flag[0] ].format(name=data[ feedback_flag[0] ]['filename']) )
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

        if any(x in query.lower() for x in ["tell me","brief","who", "what", "when", "where", "why","which" ,"how", "is", "are", "can", "may", "will", "could","give"]) :

            query_vec =  vecQuery(QUERY=query,APIKEY=APIKEY,model_name=model_name,Function=Function) 

            query = {
                "query": {
                    "nested": {
                        "path": "question_vector",  # Name of the nested field
                        "query": {
                            "function_score": {
                                "query": {
                                    "match_all": {}
                                },
                                "script_score": {
                                    "script": {
                                        "source": "cosineSimilarity(params.queryVector, 'question_vector.vector') + 1.0",
                                        "params": {
                                            "queryVector": query_vec
                                        }
                                    }
                                }
                                            }
                                        }
                                    }
                                }
                            }
           
            query = {  
                "field": "file_vector",
                "query_vector": query_vec,
                "k": 10,
                "num_candidates": 50
                }
              
            responses = ES.search(index="test-vec-index", knn=query)


            response = ES.search(index='test-index-new', body=query)
            if response:
               counter = 0
               dumpz = []
            #    temp = tempfile.NamedTemporaryFile(mode="w+",delete=False) 
               for hit in response['hits']['hits'] :
                   if counter < 4: 
                        dumpz.append({"filename":hit["_source"]["file_name"] ,"score" : hit["_score"] ,"file_content":hit["_source"]["file_content"] })
                        

                        # responses.append({'filename':hit["_source"]["file_name"] ,'score' : hit["_score"]})
                        # print(f'file name {hit["_source"]["file_name"]} match score : {hit["_score"]} \n') 
                        # yield responses[counter]
                        # print(responses[counter])
                        counter += 1
               json.dump(dumpz,temp)         
            #    temp.flush()     
               temp.close()    
               counter = 0
               with open(os.path.join(temp.name),"r") as f :
                    data = json.load(f)
                    # print(data)
                    for i in data :
                        print(i["filename"],'\n') 
                    # responses.append(f"name : { data[0]['filename'] } score : { data[0]['score'] } ")
                    responses.append(Feedbacks[ feedback_flag[0] ].format( name=data[0]['filename'] ) )    
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
   