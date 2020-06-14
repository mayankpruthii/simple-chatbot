# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy 
import re
import tensorflow as tf
import time


# making a list of movies lines and conversations
# lines are the raw conversations in alphabets
# conversations are a list of line numbers given by the users-id and movie-id
lines = open('movie_lines.txt',  encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt',  encoding = 'utf-8', errors = 'ignore').read().split('\n')

# we'll make a dictionary that will map all the lines with their conversations
id2Line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2Line[_line[0]] = _line[4]
        
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" " , "")
    conversations_ids.append(_conversation.split(','))


# Getting the questions and answers separately    
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2Line[conversation[i]])
        answers.append(id2Line[conversation[i + 1]])
        
# Cleaning the texts
def clean_texts(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)    
    return text

clean_questions =  []
for question in questions:
    clean_questions.append(clean_texts(question))
    
clean_answers =  []
for answer in answers:
    clean_answers.append(clean_texts(answer))
    
# Creating a dictionary that maps each word to the number of occurences
# implementation of bag_of_words
word2count = {}
for question in clean_questions:
    for word in question.split():
        if (word not in word2count):
            word2count[word] = 1
        else:
            word2count[word] += 1
    
for answer in clean_answers:
    for word in answer.split():
        if (word not in word2count):
            word2count[word] = 1
        else:
            word2count[word] += 1
            
threshold = 20

questionswords2int = {}
word_number = 0
for Word, count in word2count.items():
    if count >= threshold:
        questionswords2int[Word] = word_number
        word_number += 1

answerswords2int = {}
word_number = 0
for Word, count in word2count.items():
    if count >= threshold:
        answerswords2int[Word] = word_number
        word_number += 1

# Adding the last tokens to the above two dictionaries
# used by the seq2seq model for eos and other various checkpoints
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

# Appending the tokens to the bag of words
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
    
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1

# Inverse the dictionary
answersints2word = {w_i: w for w, w_i in answerswords2int.items()}    

#Adding EOS to every answers in list
for i in range(len(clean_answers)):
    clean_answers[i] += " <EOS>"
    
# Translate all the words in the clean_questions and clean_answers into integers
# Replacing all the words that were filtered out by <OUT>
questions2int = []
for question in clean_questions:
    new_question_int = []
    for word in question.split():
        if word not in questionswords2int:
            new_question_int.append(questionswords2int['<OUT>'])
        else:
            new_question_int.append(questionswords2int[word])
    questions2int.append(new_question_int)
    
answers2int = []
for answer in clean_answers:
    new_answer_int = []
    for word in answer.split():
        if word not in answerswords2int:
            new_answer_int.append(answerswords2int['<OUT>'])
        else:
            new_answer_int.append(answerswords2int[word])
    answers2int.append(new_answer_int)
    
# Sorting Questions and Answers by the *length of questions*
# so that the padding can be done easily
    
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):
    for i in enumerate(questions2int):
        # enumrate returns index and the value
        # here it returns i[0] = index and i[1] = question in int form
        if len(i[1]) == length:
            sorted_clean_questions.append(questions2int[i[0]])
            sorted_clean_answers.append(answers2int[i[0]])