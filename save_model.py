from pickle import dump
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from numpy.random import shuffle


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# split a loaded document into sentences
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    return pairs


# clean a list of lines
def clean_pairs(lines):
    cleaned = list()

    # prepare translation table for removing punctuation
    # table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # split on white space
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]

            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)


# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)


# load dataset
filename = 'kan-eng.tsv'
doc = load_doc(filename)
# split into english-kannada pairs
pairs = to_pairs(doc)
# clean sentences
clean_pairs = clean_pairs(pairs)
# save clean pairs to file
save_clean_data(clean_pairs, 'english-kannada36.pkl')
for i in range(100):
    print('[%s] => [%s]' % (clean_pairs[i, 0], clean_pairs[i, 1]))


# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))


# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)


# load dataset
raw_dataset = load_clean_sentences('english-kannada36.pkl')

# reduce dataset size
n_sentences = 36000
dataset = raw_dataset[:n_sentences, :]
# random shuffle
shuffle(dataset)
# split into train/test
train, test = dataset[:34000], dataset[34000:]
# save
save_clean_data(dataset, 'english-kannada-both36.pkl')
save_clean_data(train, 'english-kannada-train36.pkl')
save_clean_data(test, 'english-kannada-test36.pkl')


# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))


# load datasets
dataset = load_clean_sentences('english-kannada-both36.pkl')
train = load_clean_sentences('english-kannada-train36.pkl')
test = load_clean_sentences('english-kannada-test36.pkl')


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)


# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))
# prepare kannada tokenizer
kan_tokenizer = create_tokenizer(dataset[:, 1])
kan_vocab_size = len(kan_tokenizer.word_index) + 1
kan_length = max_length(dataset[:, 1])
print('Kannada Vocabulary Size: %d' % kan_vocab_size)
print('Kannada Max Length: %d' % (kan_length))


# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X


# one hot encode target sequence
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


# prepare training data
trainX = encode_sequences(kan_tokenizer, kan_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)
# prepare validation data
testX = encode_sequences(kan_tokenizer, kan_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = encode_output(testY, eng_vocab_size)


# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model


# define model
model = define_model(kan_vocab_size, eng_vocab_size, kan_length, eng_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
print(model.summary())
# plot_model(model, to_file='model.png', show_shapes=True)


# fit model
filename = 'model32k.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=1000, batch_size=128, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)
