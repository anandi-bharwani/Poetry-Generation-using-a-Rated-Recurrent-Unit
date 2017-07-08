# Poetry-Generation-using-a-simple-Recurrent-Unit

In this project I used RNN and Word embeddings to create a language model. I've used Robert Frost poems for training where each line would serve as an input sequence to the model. The model can be divided into a word embedding layer and the recurrent layer.

The input to the model is a one-hot-encoded vector of the size of the vocabulary(V). The Word embedding layer will convert every word to a D-sized vector. This will then be the input to the recurrent layer which would predict the next word in the sequence.[poetry_generation.py]
After the training is complete, the model is saved into an npz file which can later be loaded to generate poems.

A couple of changes were made to the model later. It now used a Rated recurrecnt unit instead of a simple one. The other change is related the the end of a sequence/line. The end of line was over represented in the dataset, which is why short lines were predicted by the first model. The second model now was trained with endline only 50% of the time, thus generating longer sequences.[rrnn_generation.py]


