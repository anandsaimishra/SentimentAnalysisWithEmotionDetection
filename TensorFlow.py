#  Property of Godslayer™
#  Code wirtten by Anand Sai Mishra
#  On : 7/5/20, 3:52 PM

import matplotlib.pyplot as plt
# to import the imdb dataset
import numpy as np
# We are using IMDB review dataset
from tensorflow.keras.datasets import imdb
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
# We will only use the most used 10,000 words and the rest will be ignored
print("This is the first entry on the training set :", x_train[0])
'''
Here each word is actually represented by a number value instead of the word.
As neural networks dont understand text they only understand numbers.
'''
print("These are the training labels : ", y_train[0])
'''
0 represents a negative review and 1 represents a positive review.
'''
class_names = ['Negative', 'Positive']
'''There is a helper function that can help us get the token for the corresponding word.'''
word_index = imdb.get_word_index()
'''To find the token for the word 'hello' we can use'''
print(word_index['hello'])

# Decoding the Reviews
# The decoded part is for our referance... for the decodng we use a dictionary where
# the key values paris.
VOCAB_SIZE = 1000
reverse_word_index = dict((value, key) for key, value in word_index.items())


def decode(review):
    text = ''
    for i in review:  # i means the token value of the review
        text += str(reverse_word_index[i])  # gives us the word and then
        text += ' '
    return text  # this will give back the text

print(decode(x_train[0]))
# this is for our comfort to see the word instead of the tokens... though these words are not in the correct order.
'''
Before we can push this review into the neural network we have on problem ....

All these reviews are of different lengths 
'''


def show_len(apple):
    return 'Length of the first training example :', len(apple)


print(show_len(x_train[0]))

'''
Now to solve this we can use a technique called padding.
We can use meaningless words to pad our cases.
'''
print("The token value of the word 'the' is: ", word_index['the'])
from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(x_train, value=word_index['the'], padding='post', maxlen=256)
x_test = pad_sequences(x_test, value=word_index['the'], padding='post', maxlen=256)

print(show_len(x_train[0]))
'''
We can consider each single word in the entire dataset as a seperate feature.1velala
The problem with that is each word will process individually and will be generalized. 
This methodology is called 'ONE HOT ENCODING' It doesnt understand the feature correlation at all.
"This Tuna Sandwich is quite tasty" -> this will not translate the learning to "This Chicken _____ is quite tasty."
'''

'''
Word Embedding: These are featre representations of various words. This is essentially that each word will have some 
values corresponding to each feature.
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D

model = Sequential([
    # Embeding(layers, dimentions) layer can only be used as the first layer.
    Embedding(10000, 16),
    GlobalAveragePooling1D(),  # This will convert our 10000,16 Layer to a 16 dimentional vector layer for each batch.
    Dense(16, activation='relu'),  # Rectified Linear Unit. This layer has only 16 nodes
    Dense(1, activation='sigmoid')  # This will give you a binary classfication output.
])

model.compile(
    loss='binary_crossentropy',  # THis is commonly used in reclassification problems.
    optimizer='adam',  # THis is a varient of the stocastic gradient dissent.
    metrics=['accuracy']
)
# print("Model Summery", model.summary())
model.summary()
from tensorflow.python.keras.callbacks import LambdaCallback

simple_log = LambdaCallback(
    on_epoch_end=lambda e, l: print(e, end='.'))  # THis is to print a '.' at the end of each epoc.

E = 20  # This is the number of Epoc's that we are going to use.
h = model.fit(
    x_train, y_train,
    validation_split=0.2,
    batch_size=512,
    epochs=E,
    callbacks=[simple_log],
    verbose=1
)



plt.title('Accuracy Graph')
plt.plot(range(E), h.history['accuracy'], label='Training')
plt.plot(range(E), h.history['val_accuracy'], label='Validation')
plt.legend()
plt.show()
plt.title('Loss Graph')
plt.plot(range(E), h.history['loss'], label='Training')
plt.plot(range(E), h.history['val_loss'], label='Validation')
plt.legend()
plt.show()


loss, accuracy = model.evaluate(x_test, y_test)
print('Test set accuracy is : ', accuracy * 100)

print('Test set loss is : ', loss * 100)
'''print("x_test =", x_test[0])
print("npExpand = ",np.expand_dims(x_test[0], axis=0))
# Testing some predictions
#p = model.predict(np.expand_dims(x_test[0], axis=0))  # If we are using the entire set then we dont need this
# But since we are only passing only one example at a time hence we need to expand the singular set.

#print("The current review that we are checking is ", class_names[np.argmax(p[0])])
sample_text = 'The movie was not good. The animation and the graphics were terrible. I would not recommend this movie.'
'''

def RunCode():
    sample_text = input("Please Enter the text to be taken for analysis \n")
    tokenized = tf.keras.preprocessing.text.one_hot(sample_text, round(len(sample_text) * 1.3))
    text1 = pad_sequences([tokenized], value=word_index['the'], padding='post', maxlen=256)
    predictions = model.predict(text1)
    if predictions > 0.5:
        result = 'Positive'
    else:
        result = 'Negative'

    # print("The Program has determined that the review you have provided is ",class_names[np.argmax(predictions)])
    print("\n The Program has determined that the sentiment you have provided is ", result)

    '''
    The movie was cool. The animation and the graphics were out of this world. I would recommend this movie.

    The movie was not good. The animation and the graphics were terrible. I would not recommend this movie.
    '''
def Loop():
    r = input("Would you like to try another one? (Input Format: Y 'or' N) \n")
    if r == "Y":
        RunCode()
        Loop()
    elif (r == "N"):
        print("Program terminating. Thanks for using\n")
    else:
        print("Error Input Terminating.....The correct input format is (Y 'or' N) \n")
        Loop()

def main():
    RunCode()
    Loop()


if __name__ == '__main__':
    main()

    '''
    Negative:
    I thought this film would be a lot better then it was. It sounded like a spoof off of the spy gener, and the start of it reminded me of Pleasantvil, but this film came up short.<br /><br />The plot is just to ridiculous. The KGB and Soviet Union in Russia have started up a spy school to teach their spies' how to act like Americans, but the town they set up in it for training is a bit dated, so they grab two yanks from the US to spice things up. I don't know, but this seems just to out there. It gets really odd when next to no one in this all Russian town speaks in a Russian accent. Someone screwed up in the casting job.<br /><br />Also, for a comedy this is painfully dry. There is one, two funny spots tops, and they are nothing to sing and dance about. The film in the end will likely put you to sleep.<br /><br />And, as a twisted punch in the face, this film is so pro the US it makes me sick. The movie keeps on saying again and again, the US is God and Russia is the devil. This is the kind of smear campaign that was done against the Japanese in World War 2. It's films like these that makes everyone think that the US is full of itself.<br /><br />This gets a 4 out of 10, and I'm being kind. It should really get a one, but the dance scene was funny, but then again it dragged far to long to be really funny.
    
    
    Positive:
this movie is well done on so many [UNK] that i am in [UNK] that the score is as low as it is [UNK] [UNK] as of this writing this movie has [UNK] special effects a true [UNK] storyline [UNK] great character [UNK] and [UNK] [UNK] they have to be seen to be [UNK] the only [UNK] i have is the [UNK] on the [UNK] dvd version i got some lines were not [UNK] br br i just dont understand when i read hear from various [UNK] it has a [UNK] plot i couldnt follow the story or characters came from [UNK] from the very [UNK] time i watched this movie i [UNK] it [UNK] it knew why characters were there and i absolutely loved it ive watched it about [UNK] times already and each time it is [UNK] [UNK] oh and this is not just my opinion because ive shown this movie to many [UNK] [UNK] people who have never seen an [UNK] film before who feel the same way not one of them [UNK] to follow the storyline and each person [UNK] their love for this movie oh man why cant we have stuff like this coming out of hollywood at least [UNK] of the [UNK] had a nice [UNK] of special effects character development and [UNK] br this is not coming from a [UNK] film [UNK] [UNK] either i own an [UNK] [UNK] of [UNK] films and i must say that this movie is one of my greatest [UNK] when you watch it you will be [UNK] away by the amazing special effects and [UNK] feel of this movie you will be [UNK] into this fantasy world and you wont want to leave ive seen both the [UNK] version and the [UNK] both done by [UNK] [UNK] and the [UNK] is far better in [UNK] [UNK] br [UNK] the [UNK] i have one [UNK] [UNK] about this movie i didnt want it to end im [UNK] you mr [UNK] can we please have a sequel  
    '''