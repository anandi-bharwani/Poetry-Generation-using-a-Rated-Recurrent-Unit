# Poetry-Generation-using-a-Rated-Recurrent-Unit
The aim of this project is to generate poetry using an RNN. We perform unsupervised learning for 1500 lines of Robert Frost's poems. Following the training, we attempt to generate 4 lines of poetry which would mimick his style.

## Dataset

The **robert_frost.txt** contains poetry lines from Robert Frost's Poems. The two models have been trained on this data to remember words and their structure. We create a vocabulary vector for each line in the poetry. Each line is represented by vocabulary vector of size V where V is the vocabulary space. The vocabulary vector V contains the word indexes from the vocabulary dictionary in the order in which they appear in that line. In addition it contains a "0" as the first element signifying the start of a line and a "1" at the end signifying the end of line.

Hence, "I love you" would generate a vector V = [0,4,2,3,1] if the vocabulary dictionary is defined as D = {"love":2, "you":3, "I":4, "caught":5,  "sugar":6, "but":7}.

## Model I

The file **poetry_generation.py** contains the code for training on the Robert Frost dataset. The code has been written in Python using theano. It performs a stochastic gradient descent on a recurrent neural network propgating the error backwards at each layer. The model can be divided into two layers:-

- **Word Embedding Layer**:  Here the one-hot-encoded vector of size V is converted to a D-sized vector using the **Word Embedding Matrix (We)** of size V X D (where D<<<V). This embedding matrix is trained over time.

- **Recurrent Layer**:  A simple recurrent unit with one layer and 500 hidden units is used. The input and output are as follows:-

    * **Input is a sequence of D-sized vectors**: Every D-sized vector represents a word, signifying the features of a particular word by D weights. It is the output we get from the Word Embedding Layer.

    * **Output is a word which follows the input sequence** : The output vector contains the prediction of next word at every step. The last value in the vector is the final prediction. If the last value is "1", it signifies "End of Line".
    
   
## Model II

The file **rrnn_generation.py** contains the code for training the Robert Frost data same as **poetry_generation.py** with some modifications:

- The simple recurrent unit is now replaced by a rated recurrent unit.

- One limitation of the previous model was that the 'end of line' was over represented in the dataset, which is why short lines were predicted. To solve this, I trained the second model with 'end of line' only 50% of the time, thus generating longer sequences.

