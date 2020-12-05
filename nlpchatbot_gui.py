# -*- coding: utf-8 -*-
"""
Project : Chat Bot 
Team Members : 
Vishal Panuganti 
Shivani 
Shouray 
Vinay Bollapu 

Description: GUI Code
"""

# Importing all the required modules
# nltk is used to perform text vectorization and prepare the data before building the model
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import requests

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents_chatbot.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


#Checks for any internal Display
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
    
  
import matplotlib.pyplot as plt

#Tokenizes the words 
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

#This function gets the zipcodes from the defined api 
#This api is created out of Hospitals data set 
#If zipcode matches from our API, it returns the address of the Hospital 
#This function returns the list of all zipcodes present in the data set or API 
def getzipcodes():
    response = requests.get('https://sheetdb.io/api/v1/q4kez33jk9piv')
    jsonData = response.json()
    zipcodelist = []
    for item in jsonData:
        if item["zipcode"] not in zipcodelist:
            zipcodelist.append(item["zipcode"])
    return zipcodelist

# This method takes in zipcode and returns the address of the hospital 
def getHospitalsData(zipcode):
    response = requests.get('https://sheetdb.io/api/v1/q4kez33jk9piv')
    jsonData = response.json()
    hospitalslist = []
    for item in jsonData:
        if item["zipcode"] == zipcode:
            info = item["Hospital"] + ", " + item["Hospital_Address"]
            hospitalslist.append(info)
        else:
            hospitalslist.append("No hospital found, Please enter nearby zipcode.")
    return hospitalslist

#return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

#This function gets the response from intents file 
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res
    
#Creating GUI with tkinter
import tkinter as tk
from tkinter import *

#Calling the zipcode function and zipcodes holds all the zipcode present in our hospital api 
zipcodes = getzipcodes()

#This function send the data back to user
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    #These conditions are matched against the user input 
    #If user enters zipcode then, bot returns the hospital data
    if msg in zipcodes:
            ChatLog.config(state=NORMAL)
            ChatLog.insert(END, "You: " + msg + '\n\n')
            ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
    
            res = getHospitalsData(msg)     #chatbot_response(msg)
            res = "\n".join(res)
            ChatLog.insert(END, "Bot: " + res + '\n\n')
            ChatLog.config(state=DISABLED)
            ChatLog.yview(END)
        
    #If not zipcode, then it checks for the intent response 
    elif msg != '':
            ChatLog.config(state=NORMAL)
            ChatLog.insert(END, "You: " + msg + '\n\n')
            ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
    
            res = chatbot_response(msg)
            ChatLog.insert(END, "Bot: " + res + '\n\n')
            ChatLog.config(state=DISABLED)
            ChatLog.yview(END)
 
#Creating the UI for printing the messages 
base = tk.Tk()
base.title("HEALTH BOT")
base.geometry("400x500")
base.resizable(width=False, height=False)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)

# #Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                     bd=0, bg="slate gray", activebackground="#3c9d9b",fg='gray14',
                     command= send )

# #Create the box to enter message
EntryBox = Text(base,bd=0, bg="gray88",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)

# #Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=6, y=401, height=90, width=265)
SendButton.place(x=274, y=401, height=90)

base.mainloop()



