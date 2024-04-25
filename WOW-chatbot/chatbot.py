"""
Created on Thu Feb 22 18:25:53 2024

@author: joannamukasa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
import random
import numpy as np

import nltk
from nltk.sem import Expression
from nltk.stem import WordNetLemmatizer
from nltk.inference import ResolutionProver
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog

import langid
from deep_translator import GoogleTranslator

from simpful import *


#  Initialising  AIML agent
import aiml
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="aimlPatterns.xml")


# Opening Question and answers csv
csvData = []
with open('QaPairs.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        User_Input, UserResponse = row
        csvData.append((User_Input.strip(), UserResponse.strip()))


# Reading from Knowledge base and appending to list
kb = []
def read_expr(expr):
    return Expression.fromstring(expr)


with open('knowledgeBase.csv', 'r', encoding='utf-8') as file2:
    reader = csv.reader(file2)
    for row in reader:
        kb.append(read_expr(row[0]))
    
# Checks Knowledge base integrity

def kb_integrity(kb):
    q = None
    if ResolutionProver().prove(q, kb):
        print("There is a Contradiction in the Knowledge base")
        sys.exit()
kb_integrity(kb)

# Append new statements to KB and write to CSV
def update_kb(expr):
    if expr not in kb:
        kb.append(expr)
        with open('knowledgeBase.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([str(expr)])

# checks Expression integrity and checks for contradiction in the kb

def check_expr_integrity(new_expr, kb):
    # Check if the new expression contradicts any existing expression in the KB
    for existing_expr in kb:
        if ResolutionProver().prove(read_expr('not ' + str(new_expr)), [existing_expr]):
            return False
    return True

#Perform inference
def perform_inference(query, kb):
    return ResolutionProver().prove(query, kb, verbose=False)

# Lemmatize function
def lemmatize(text):
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)


# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()


# Prepare CSV data for TF-IDF
csv_questions = [lemmatize(pair[0]) for pair in csvData]

# Fit TF-IDF vectorizer
tfidf_matrix = vectorizer.fit_transform(csv_questions)


# Select Image from Files
def select_image():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select Image File")
    root.destroy()
    return file_path

#Use Model to Predict class Name
def model_prediction(img_path):
    model = tf.keras.models.load_model('myModel.h5')
    # Load the image from file
    img = image.load_img(img_path, target_size=(256, 256))  # Resize image to (256, 256)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make predictions using the loaded model
    predictions = model.predict(img_array)

    # Get the predicted class
    predicted_class = np.argmax(predictions[0])
    
    # Assuming you have a list of class names
    class_names = ['chichen_itza', 'christ_the_redeemer', 'great_wall_of_china', 'machu_pichu', 'roman_colosseum', 'taj_mahal']
    
    # Check if the predicted class index is within the range of class_names
    if predicted_class < len(class_names):
        predicted_class_name = class_names[predicted_class]
        # Plot the image
        plt.imshow(img)
        plt.title('Predicted class: {}'.format(predicted_class_name))
        plt.axis('off')
        plt.show()
        print("This is an image of:",predicted_class_name)
    else:
        print("Error: Predicted class index out of range")




#Detect language of user input and translate to english
def detect_language(userInput):
    detected_language, confidence = langid.classify(userInput)
    
    # translate to English if detected language is not English
    if detected_language != 'en':
        userInput = GoogleTranslator(source=detected_language, target='en').translate(userInput)
    return userInput

#Translate output back to the same language of input
def translate_output(userInput, response):
    detected_language, confidence = langid.classify(userInput)
    
    # translate to English if detected language is not English
    if detected_language != 'en':
        response = GoogleTranslator(source='en', target=detected_language).translate(response)
    return response


#Fuzzy Logic Game
def fuzzyGame():
    #Creating object
    FS = FuzzySystem()

    #Define fuzzy sets and linguistic variables
    F_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="False")
    F_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=10), term="True")
    FS.add_linguistic_variable("Answer", LinguisticVariable([F_1, F_2], concept="True or False", universe_of_discourse=[0,10]))

    T_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="ancient")
    T_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=20), term="natural")
    T_3 = FuzzySet(function=Trapezoidal_MF(a=10, b=20, c=25, d=25), term="current")
    FS.add_linguistic_variable("Wonder", LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0,25]))

    #Define rules
    R1 = "IF (Wonder IS current) THEN (Answer IS True)"
    R2 = "IF (Wonder IS ancient) THEN (Answer IS False"
    R3 = "IF (Wonder IS natural) THEN (Answer IS False)"

    FS.add_rules([R1, R2, R3])

    #Set Values
    FS.set_crisp_output_value("False", 0)
    FS.set_crisp_output_value("True", 10)
    
    #Lists of Wonders
    New = ['Petra', 'Colosseum', 'Chichen Itza', 'Machu Pichu']
    Ancient = ['Light House of Alexandria', 'Colossus of Rhodes', 'Statue of Zues at Olympia', 'Hanging Gardens of Babylon', 'Great Pyramid of Giza']
    Natural = ['Victoria Falls', 'Mount Evarest', 'Grand Canyon', 'Northern Lights', 'Great Barrier Reef']

    #Start of game
    print("")
    print("Welcome to The Fuzzy Game")
    print("You will be presented with three options. You have to choose which of the three options is a Current Wonder of the World")

    score = 0
    j = 0
    while j < 5:
        print("\n")
        print("Question ",j+1," : Which of the folowing Monuments is a Current or New Wonder of the world")
        print("*Please type in the answer in full")
        print("\n")
        
        #Creates list and shuffles entries from lists above
        options = []

        options.append(random.choice(New))
        options.append(random.choice(Ancient))
        options.append(random.choice(Natural))

        random.shuffle(options)
        
        i = 1
        for item in options:
            print(i,".", item, end=' ')
            i += 1
            
        print("")
        ans = input("\nAnswer: ")
        
        if ans in New:
            FS.set_variable("Wonder", 25)
            inf = FS.Sugeno_inference(["Answer"])
            outputValue = inf["Answer"]
            score += outputValue
        elif ans in Ancient:
            FS.set_variable("Wonder", 0)
            inf = FS.Sugeno_inference(["Answer"])
            outputValue = inf["Answer"]
            score += outputValue
        elif ans in Natural:
            FS.set_variable("Wonder", 15)
            inf = FS.Sugeno_inference(["Answer"])
            outputValue = inf["Answer"]
            score += outputValue
        else:
            print("Invalid Input")
        
        
        
        j += 1
        
    if score > 30:
        print("\n Congratulations!!")
        print("you Scored:", score ,"0ut of 50")
    else:
        print("\n Better Luck Next Time")
        print("you Scored:", score ,"0ut of 50")
        
    print("\n Thank you for playing my game. You will now be returned to the regular chat")

    



# Start of ChatBot
print("Welcome! Please feel free to ask me any questions about the Wonders of the World!")

# Main loop

while True:
    #get user input
    try:
        userInput = input("> ")
        #detects language of user input
        userInput = detect_language(userInput)
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    
    #determine response agent
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
        answer = translate_output(userInput, answer)
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        elif cmd == 31: # if input pattern is "I know that * is *"
            object,subject=params[1].split(' is ')
            expr=read_expr(subject + '(' + object + ')')
            if check_expr_integrity(expr, kb):
                #Adds new expression to knowledge base
                update_kb(expr)
                print('OK, I will remember that',object,'is', subject)
            else:
                print('This statement contraditcs my existing knowledge')
            
        elif cmd == 32: # if the input pattern is "check that * is *"
            object,subject=params[1].split(' is ')
            expr=read_expr(subject + '(' + object + ')')
            if perform_inference(expr, kb):
               print('Correct.')
            else:
               print('Sorry that is incorrect.') 
        elif cmd == 50:
            #Select image from files
            file_path = select_image()
            if file_path:
                print("Selected image file:", file_path)
                #make prediction
                model_prediction(file_path)
            else:
                print("No file selected")
        elif cmd == 40: # Fuzzy Logic Game
            answer = fuzzyGame();
        elif cmd == 99:
            csv_response = None
            preprocessed_input = lemmatize(userInput)
            
            #TF-IDF vector for user input
            user_tfidf = vectorizer.transform([preprocessed_input])

            #cosine similarities
            similarities = cosine_similarity(user_tfidf, tfidf_matrix)

            # Finding the most similar question
            most_similar_index = np.argmax(similarities)
            max_similarity = similarities[0, most_similar_index]
            
            if max_similarity > 0.5:
                answer = csvData[most_similar_index][1]
                answer = translate_output(userInput, answer)
                print(answer)
            else:
                print("I did not get that, please try again.")
            
            
    else:
        answer = translate_output(userInput, answer)
        print(answer)
        
        