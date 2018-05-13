import codecs
import csv
import re
import random
import string
from nltk.corpus import wordnet
def load_glove_model(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r',encoding="utf8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean_str_2(line):
    line = re.sub(r'[.,"!]+', '', line, flags=re.MULTILINE)  # removes the characters specified
    line = re.sub(r'^RT[\s]+', '', line, flags=re.MULTILINE)  # removes RT
    line = re.sub(r'https?:\/\/.*[\r\n]*', '', line, flags=re.MULTILINE)  # remove link
    line = re.sub(r'[:]+', '', line, flags=re.MULTILINE)
    printable = set(string.printable)
    filter(lambda x: x in printable, line)
    line = ''.join(list(line))
    line = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", line)
    return line.strip().lower()
def check_sentence(string, gate):
    words = string.split() 
    count = 0
    for word in words:
        if wordnet.synsets(word):
            count = count + 1
    if count >= gate:
        return True
    else:
        return False 
def load_new_tweet_data(path, change):
    data_inc = []
    data_dec = []
    data_cos = []
    filenames_list = []
    for (dirpath, dirnames, filenames) in walk(path):
        filenames_list.extend(filenames)
        break
    for filename in filenames_list:
        date = filename[7:11] + "-" + filename[11:13] + "-" + filename[13:15]
        if date not in change:
            continue
        with open(path + "\\" + filename, encoding="ISO-8859-1") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                example = {}
                text = clean_str_2(row[0])
                if not check_sentence(text, 2):
                    continue
                example['text'] = text
                if change[date] > 3:
                    data_inc.append(example)
                elif change[date] < -3:
                    data_dec.append(example)
                else:
                    data_cos.append(example)
def tokenize(string):
    return string.split()

def build_dictionary(training_datasets):
    """
    Extract vocabulary and build dictionary.
    """  
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            word_counter.update(tokenize(example['text']))
        
    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING, UNKNOWN] + vocabulary
        
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))

    return word_indices, len(vocabulary)

def sentences_to_padded_index_sequences(word_indices, datasets):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """
    for i, dataset in enumerate(datasets):
        for example in dataset:
            example['text_index_sequence'] = torch.zeros(max_seq_length)

            token_sequence = tokenize(example['text'])
            padding = max_seq_length - len(token_sequence)

            for i in range(max_seq_length):
                if i >= len(token_sequence):
                    index = word_indices[PADDING]
                    pass
                else:
                    if token_sequence[i] in word_indices:
                        index = word_indices[token_sequence[i]]
                    else:
                        index = word_indices[UNKNOWN]
                example['text_index_sequence'][i] = index

            example['text_index_sequence'] = example['text_index_sequence'].long().view(1,-1)
            #example['label'] = torch.LongTensor([example['label']])

def epoch_generate(inc_set, dec_set, cos_set, data_size):
    total = []
    curr = []
    random.shuffle(inc_set)
    random.shuffle(dec_set)
    random.shuffle(cos_set)
    for i in range(len(inc_set)):
        if i != 0 and i % data_size == 0:
            curr_dict = {}
            curr_dict["label"] = 2
            curr_dict["data"] = curr
            total.append(curr_dict)
            curr = []
        curr.append(inc_set[i])
    curr = []
    for i in range(len(dec_set)):
        if i != 0 and i % data_size == 0:
            curr_dict = {}
            curr_dict["label"] = 0
            curr_dict["data"] = curr
            total.append(curr_dict)
            curr = []
        curr.append(dec_set[i])
    for i in range(len(cos_set)):
        if i != 0 and i % data_size == 0:
            curr_dict = {}
            curr_dict["label"] = 1
            curr_dict["data"] = curr
            total.append(curr_dict)
            curr = []
        curr.append(cos_set[i])
    # random.seed(19)
    random.shuffle(total)
    
    return total
def data_iter(source, batch_size):
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
            random.shuffle(order)   
        batch_indices = order[start:start + batch_size]
        yield [source[index] for index in batch_indices]

# This is the iterator we use when we're evaluating our model. 
# It gives a list of batches that you can then iterate through.
def eval_iter(source, batch_size):
    batches = []
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while start < dataset_size - batch_size:
        start += batch_size
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        batches.append(batch)
     
    return batches

# The following function gives batches of vectors and labels, 
# these are the inputs to your model and loss function
def get_batch(batch):
    vectors = []
    labels = []
    for dict in batch:
        vectors.append(dict["text_index_sequence"])
        labels.append(dict["label"])
    return vectors, labels

 