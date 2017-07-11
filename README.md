# Poetry-Generation-using-Recurrent-Units

In this project I used RNN and Word embeddings to create a language model. I've used Robert Frost poems for training where each line would serve as an input sequence to the model. The model can be divided into an embedding layer and the recurrent layer.

The input to the model is a one-hot-encoded vector of the size of the vocabulary(V). The embedding layer will convert every word to a D-sized vector. This will then be the input to the recurrent layer which would predict the next word in the sequence.[poetry_generation.py]

After the training is complete, the model is saved into an npz file which can later be loaded to generate poems.

One limitation of with this model was that the end of line was over represented in the dataset, which is why short lines were predicted by the first model. The second model now is trained with endline only 50% of the time, thus generating longer sequences. Another modification is that I replaced the simple recurrent unit wit a rated recurrent unit.[rrnn_generation.py]


