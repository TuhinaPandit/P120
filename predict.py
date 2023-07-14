#Text Data Preprocessing Lib
import nltk
import random

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
import json
import pickle
import numpy as np

ignore_words = ['?', '!',',','.', "'s", "'m"]
import tensorflow
from data_preprocessing import get_stem_words

model = tensorflow.keras.models.load_model("chatbot_model.h5")

#loading intents.json file
intents = json.loads(open('intents.json').read())
#pickle converts data to binary form
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))



def bot_response(user_input):
    #passing user input to tokenizer -> breaks down into smaller words
    input1 = nltk.word_tokenize(user_input)
    #converting words to its root form
    input2 = get_stem_words(input1,ignore_words)
    #removing repeated words using set, converting to list, arranging in alphabetical order using sorted
    input3 = sorted(list(set(input2)))
    bag = []
    bag_of_words = []
    for word in words:
        if word in input3:
            #append = add smth to the list 
            #if words match with stem_words(pattern =user input)
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    bag.append(bag_of_words)
    #converting bag_of_words to array
    bag = np.array(bag)
    #passings bag_of_words to model for prediction
    prediction = model.predict(bag)
    #output of model in form of label (num assigned to class)
    predicted_label = np.argmax(prediction[0])
    #choosing correct class using predicted label (greeting/goodbye)
    predicted_class = classes[predicted_label]
    #writing for loop through the intents.json file
    #pattern user input, bot is bot response
    for intent in intents["intents"]:
        #checking if the tag matches with the predicted class or not
        if intent["tag"]==predicted_class:
            #choosing random response from responses
            bot_response = random.choice(intent["responses"])
            #printing response in output
            return bot_response


#first statement user will see
print("Hi I am Stella, how can I help you?")
while True:
    #second statement
    user_input = input("Type your message here: ")
    #takes user input and passes to bot_response function
    response = bot_response(user_input)
    #prints response
    print(response)


