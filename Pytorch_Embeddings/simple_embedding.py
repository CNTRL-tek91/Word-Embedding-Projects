#This is a simple embedding layer program to see the steps that are involved in building an embedding layer for a transformer

#Library Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import re



#Sample Corpus of text
corpus = [
    "A cat sat on the mat.",
    "The dog played in the park.",
    "Cats and dogs are great pets."
]
print(f"Original Corpus: {corpus}\n")



# Step 1: Text Preprocessing and Integer Encoding
# 1.1. Normalization and Tokenization

# Function to lowercase and remove punctuation from text
def normalize_and_tokenize(text):
    text = text.lower() #lowercasing text
    text = re.sub(r'[^\w\s]', '', text) #removing punctuation
    return text.split() #tokenizes the text into individual words

'''
For every element in corpus, the element is passed into the normalize_and_tokenize() function, 
and the returned data is appended to the tokenize_corpus list
'''
tokenized_corpus = [] #Empty list to store tokens
for doc in corpus:
    tokenized_corpus.append(normalize_and_tokenize(doc)) 
print(f"Tokenized Corpus: {tokenized_corpus}\n") #tokenized_corupus is a list of lists


# 1.2. Building the Vocabulary

vocab = set() #Emtpy set to ensure no duplicate words
for doc in tokenized_corpus: #for every list in the tokenized_corpus list
    for word in doc: #for every word in the specified list
        vocab.add(word) #add the word the the vocab set
vocab = sorted(vocab) #sorts all the words in alphabetical order


word_to_index = {} #create an empty dictionary that will store the words and the integer mappings
word_to_index['<PAD>'] = 0 #The 0th index will be mapped to the padding value
word_to_index['<UNK>'] = 1 #The 1st index will be mapped to the unknown value
for i,word in enumerate(vocab):
    word_to_index[word] = i+2 #maps every word in the vocab set to its corresponding integer index

index_to_word = {i:word for word,i in word_to_index.items()}#an dictionary that maps the index to the word for all the items in the word_to_index dictionary

vocab_size = len(word_to_index) #total number of words in the vocabulary

print(f"Vocabulary Size: {vocab_size}")
print(f"Vocabulary Mapping (word -> index): {word_to_index}")


# 1.3. Integer Encoding for the corpus

'''
Each word in every sentence in our original corpus is converted to the integer index that is
associated to the word based on the word_to_integer dictionary mappings
'''
encoded_corpus = [[word_to_index.get(word, word_to_index['<UNK>']) for word in doc] for doc in tokenized_corpus]

print(f"Integer Encoded Corpus: {encoded_corpus}\n")


# 1.4. Padding and Truncating

#Find the length of the longest sequence
max_sequence_length = max(len(doc) for doc in encoded_corpus)

#Pad all sequences to the max_sequence_length
padded_corpus = [doc + [word_to_index['<PAD>']] * (max_sequence_length - len(doc)) for doc in encoded_corpus]
print(f"Padded Corpus: {padded_corpus}")

#Convert to a Pytorch tensor
input_tensor = torch.tensor(padded_corpus, dtype = torch.long)

print(f"Padded Input Tensor of shape (batch size, sequence length): ")
print(input_tensor)
print(f"Shape: {input_tensor.shape}\n")



# Step 2: Initializing Embedding Layer and Lookup

# 2.1. Defining the Embedding Layer
embedding_dimension = 10 #Each word will be represented by a 10-dimensional vector
embedding_layer = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dimension, padding_idx = 0)

# 2.2. Initializing the matrix with random values
print("Embedding Matrix initialized with random values(not trained): ")
print(embedding_layer.weight.data)
print(f"Shape of Embedding Matrix: {embedding_layer.weight.shape}")


# Step 3: Performing Lookup
output_tensor = embedding_layer(input_tensor)

print(f"Output Tensor of shape (batch size, sequence length, embedding dimension): ")
print(output_tensor)
print(f"Shape of Output Tensor: {output_tensor.shape}")


#------------------------------------------------------------------------------------------------------------------------------

#Training and Updating the embeddings

#This is a simple model for a dummy classification task. The main idea here is to show that the embedding weights change during training

class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SimpleClassifier, self).__init__()
        
        #Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        #Linear layer for classification
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x shape: (batch size, sequence length)
        embedded = self.embedding(x) #Shape: (batch_size, sequence length, embedding dim)

        pooled = embedded.mean(dim = 1)

        logits = self.fc(pooled)
        return logits
    

dummy_model = SimpleClassifier(vocab_size, embedding_dimension)

dummy_labels = torch.tensor([[1.0], [0.0], [1.0]], dtype = torch.float)

loss_function = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(dummy_model.parameters(), lr = 0.1)


cat_index = torch.tensor([word_to_index['cat']], dtype = torch.long)
original_cat_embedding = dummy_model.embedding(cat_index).detach().clone()
print(f"Original embedding for cat: {original_cat_embedding.numpy()}")


print("Running a few training steps: \n")
dummy_model.train()

for epoch in range(50):
    optimizer.zero_grad()
    predictions = dummy_model(input_tensor)
    loss = loss_function(predictions, dummy_labels)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 4.4 Check the embedding for 'cat' AGAIN after training
updated_cat_embedding = dummy_model.embedding(cat_index).detach().clone()
print(f"\nUpdated embedding for 'cat' (after training):\n{updated_cat_embedding.numpy()}\n")



        



    

