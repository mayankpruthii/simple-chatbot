import numpy 
import re
import tensorflow as tf
import time
import numpy as np

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

# Adding EOS to every answers in list
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
            
############################################
############################################
# Building the seq2seq model
    
# Creating placeholder for inputs
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    
    return inputs, targets, learning_rate, keep_prob


# Preprocessing targets
# By this method we put <SOS> token at the start of batch and
# remove the <EOS> token at the end
def preprocess_targets(targets, word2int, batch_size):
    
    # tf.fill fills the value in [a,b] dimension tensor
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    
    # strided_slice extracts a subset of tensor
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1])
    
    # axis = 1 means horizontally
    preprocessed_targets = tf.concat([left_side, right_side], axis = 1)

    return preprocessed_targets


# Creating the Encoder RNN Layer
# rnn_inputs -> the model inputs
# rnn_size -> number of input tensors
# num_layers -> number of layers
# keep_prob -> for dropout
# sequence_length -> length of questions in each batch
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_with_dropout = tf.contrib.rnn.DropoutWrapper(lstm,
                                                      input_keep_prob = keep_prob)
    # [lstm_with_dropout] * num_layers returns cells*layers 
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_with_dropout] * num_layers)
    # the length of backward and forward cell should be same
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32)
    
    return encoder_state


# Decoding Training set
# embedding used to convert the word into a vector of real numbers
# decoding scope -> advanced data structure like tensor
def decode_train_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length,
                     decoding_scope, output_function, keep_prob, batch_size):
    
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                                                                    attention_option = 'bahdanau',
                                                                                                                                    num_units = decoder_cell.output_size)
    
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = 'attn_dec_train')
    
    decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                  training_decoder_function,
                                                                  decoder_embedded_input,
                                                                  sequence_length,
                                                                  scope = decoding_scope)
    
    decoder_output_with_dropout = tf.nn.dropout(decoder_output, keep_prob)
    
    return output_function(decoder_output_with_dropout)



# Decoding test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words,
                    sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                                                                    attention_option = 'bahdanau',
                                                                                                                                    num_units = decoder_cell.output_size)
    
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix, 
                                                                              sos_id, 
                                                                              eos_id, 
                                                                              maximum_length, 
                                                                              num_words,
                                                                              name = 'attn_dec_inf')
 
    test_predictions, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                    test_decoder_function,
                                                                    scope = decoding_scope)
    
    
    return test_predictions


# Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state,
                num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    
    with tf.variable_scope('decoding') as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_with_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_with_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                       num_words,
                                                                       None,
                                                                       scope = decoding_scope,
                                                                       weights_initializer = weights,
                                                                       biases_intializer = biases)
        
        training_predictions = decode_train_set(encoder_state, 
                                                decoder_cell, 
                                                decoder_embedded_input,
                                                sequence_length,
                                                decoding_scope,
                                                output_function,
                                                keep_prob,
                                                batch_size)
        
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
        
    return training_predictions, test_predictions


# Building the model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, 
                  answers_num_words, questions_num_words, encoder_embedding_size,
                  decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))

    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers,
                               keep_prob, sequence_length)
    
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
   
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    
    return training_predictions, test_predictions
    

######################################################
######################################################
# Training the model

# Setting the hyperparameters
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

# Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()

# Loading model inputs
inputs, targets, lr, keep_prob = model_inputs()

# Setting sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')

# Getting shape of the inputs tensor
input_shape = tf.shape(inputs)

# Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int)

# Stting up loss error, optimizer and gradient clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_var) for grad_tensor, grad_var in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
    
# Padding the sequences with <PAD> token to make the length of question = answer
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]

# Splitting data into batches of question and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, (len(questions) // batch_size)):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch
        
# Splitting the questions and answers into training and validation set
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split :]
training_answers = sorted_clean_answers[training_validation_split :]
validation_questions = sorted_clean_questions[: training_validation_split]
validation_answers = sorted_clean_answers[: training_validation_split]

# Training 
batch_index_check_training_loss = 100
batch_index_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "checkpoint_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                        targets: padded_answers_in_batch,
                                                                                        learning_rate: learning_rate,
                                                                                        sequence_length: padded_answers_in_batch.shape[0],
                                                                                        keep_prob: keep_probability})
        
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        
        if batch_index % batch_index_check_training_loss == 0:
            print("Epoch: {:>3/{}, Batch: {:4>{}, Training Loss Error: {:>6.3f}, Training time on 100 batches: {:d} seconds".format(epoch,
                                                                                                                                    epochs,
                                                                                                                                    len(training_questions) // batch_size,
                                                                                                                                    total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                    int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        
        if batch_index % batch_index_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                
                _, batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                                targets: padded_answers_in_batch,
                                                                                learning_rate: learning_rate,
                                                                                sequence_length: padded_answers_in_batch.shape[0]})
        
                total_validation_loss_error += batch_validation_loss_error
          
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print("Validation Loss Error : {:>6.3f}, Batch validation time: {:d}".format(average_validation_loss_error,
                                                                                         int(batch_time)))
            
            learning_rate *= min_learning_rate
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print("I speak better now!")
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I dont speak better, need more practice!")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apology lmao me bad boi")
        break
    
print("Game Over!")