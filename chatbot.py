'''This is a virtual assistant which reads from a text file and gives the user information according to his input.
   The text file includes information about tourist destinations in Dubai.
Members - Aman Mohandas    2016A7PS0011U
          Gracia Tabitha   2016A7PS0027U
          Kiran Kushal     2016A7PS0246U
'''          

import nltk
import warnings
warnings.filterwarnings("ignore")

#nltk.download() #for downloading packages (first-time use only)

import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


f=open('tourist.txt','r')
raw=f.read()
raw=raw.lower()
#nltk.download('punkt')                 #first-time use only
#nltk.download('wordnet')               #first-time use only
sent_tokens = nltk.sent_tokenize(raw)   #converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)   #converts to list of words


lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello"]


# Checking for greetings
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Generating response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

#main
flag=True
print("BOT: Hi, I am your personal assistant.\nFeel free to ask me question about the tourist destinations in Dubai.\nIf you want to exit, type Bye!")

while(flag):
    print("Human: ",end='')
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            print("BOT: You are welcome.")
        else:
            if(greeting(user_response)!=None):
                print("BOT: "+greeting(user_response))
            else:
                print("BOT: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
        print ("")
    else:
        flag=False
        print("BOT: Bye!")    
        
        
