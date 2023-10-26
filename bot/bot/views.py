from django.http import HttpResponse
from django.shortcuts import render
# Code for video 7
from .vectorize import create_db
from .Bot_question_vectorize import create_db_question

from .langchain_qa import get_answer
#from .langchain_qa import first_bot_message

def index(request):
    if 'clear_history' in request.GET:
        request.session.pop('conversation_history', None)
        return render(request, 'index.html', {'conversation_history': []})
    
    #conversation_history = request.session.get('conversation_history', [])
    conversation_history = []  
    if not conversation_history:
        #initial_message = get_initial_message()
        initial_message="Hello, How can I assist you ?"
        conversation_history.append(('bot', initial_message))
        request.session['conversation_history'] = conversation_history
    return render(request, 'index.html', {'conversation_history': conversation_history})




def analyze(request):
    if request.method == 'GET':
        user_input = request.GET.get('text', '')

        #first_message=first_bot_message(user_input)

        bot_responseall = get_answer(user_input)
        #print("view", bot_responseall)

        conversation_history = request.session.get('conversation_history', [])
        conversation_history.append(('user', user_input))

        for bot_response in bot_responseall:
            #print("view", bot_response)
            conversation_history.append(('bot', bot_response))
        
        request.session['conversation_history'] = conversation_history

        return render(request, 'index.html', {'conversation_history': conversation_history})

