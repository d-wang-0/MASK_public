import random
import string
import copy
import csv

random.seed(42)
tags2index = {'O':0, 'ID':1, 'PHI':2, 'NAME':3, 'CONTACT':4, 'DATE':5, 'AGE':6, 'PROFESSION':7, 'LOCATION':8}

def _random_string(length=10):
    """
    Returns a random string of fixed length.

    :param length: length of the string to be returned
    """
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

def random_string_generator(length=10):
    """
    Generates a random string of fixed length.

    :param length: length of the string to be generated
    """
    while True:
        yield _random_string(length)

def random_name_generator(parts_min = 1, parts_max = 1, lower = True):
    """
    Generates a random name.

    :param parts_min: minimum number of parts in the name
    :param parts_max: maximum number of parts in the name
    :param lower: whether to use lowercase letters only
    """
    names = []  # not a set since we can't set a random seed
    names_set = set()  # for quicker duplicate checking
    first_name_file = open(
        "Dictionaries/dictionary_first_names.txt", 'r', encoding='utf-8')
    dictionary_first_name = first_name_file.readlines()

    for line in dictionary_first_name:
        name = line.strip()
        if lower:
            name = name.lower()
        if name not in names_set and len(name) > 1:
            names.append(name)
            names_set.add(name)

    surname_file = open(
        "Dictionaries/dictionary_surnames.txt", 'r', encoding='utf-8')
    dictionary_surname = surname_file.readlines()

    for line in dictionary_surname:
        name = line.strip()
        if lower:
            name = name.lower()
        if name not in names_set and len(name) > 1:
            names.append(name)
            names_set.add(name)
    while True:
        parts = random.randint(parts_min, parts_max)
        name = ' '.join(random.choice(names) for _ in range(parts))
        yield name

def random_profession_generator():
    """
    Generates a random profession.
    """
    jobs = []
    jobs_set = set()
    with open('Dictionaries/job_title_dictionary.csv', 'r', encoding='utf 8') as job_file:
        csv_file = csv.reader(job_file, delimiter=',')
        for row in csv_file:
            if row[2] == 'assignedrole':
                words = row[0].lower().split()
                min_2char_words = [word for word in words if len(word) > 2]
                job = ' '.join(min_2char_words)
                if job not in jobs_set:
                    jobs.append(job)
                    jobs_set.add(job)
    while True:
        yield random.choice(jobs)

def random_generator_mix(gen1, gen2, gen1_ratio=0.5):
    """
    Generates a random mix of two generators.

    :param gen1: first generator
    :param gen2: second generator
    :param gen1_ratio: ratio of samples from gen1 to gen2
    """
    while True:
        if random.random() < gen1_ratio:
            yield next(gen1)
        else:
            yield next(gen2)


def augment(X, Y, randomizers={3: random_generator_mix(random_name_generator(), random_string_generator()), 7: random_string_generator(),
            8: random_string_generator()}):
    """
    Returns a copy of samples from X with randomizers applied and their associated labels in Y.
    Samples without any randomizers applied are not included.

    :param X: list of sequences of tokens
    :param Y: list of sequences of token labels
    :param randomizers: a dictionary of iterators where the key is the tag index of the text that should be replaced with strings from the iterator.
    """
    assert len(X) == len(Y), "X and Y must have the same length"
    X_copy = []
    Y_copy = []
    for i in range(len(X)):
        assert len(X[i]) == len(Y[i]), "samples in X and Y must have the same length"
        X_i_copy = [text for text in X[i]] # deep copy
        for j in range(len(X[i])):
            tag = Y[i][j]
            if tag in randomizers:
                X_i_copy[j] = next(randomizers[tag], X[i][j])
        if X_i_copy != X[i]: # only add if any randomizers were applied
            X_copy.append(X_i_copy)
            Y_copy.append(Y[i])
    return X_copy, Y_copy

def merge_aug(X, X_aug, Y, Y_aug):
    """
    Add non-duplicate samples from X_aug to X, does the same with labels in Y_aug.
    Modifies X and Y in place.

    :param X: list of sequences of tokens
    :param X_aug: augmented list of sequences of tokens
    :param Y: list of sequences of token labels
    :param Y_aug: augmented list of sequences of token labels
    """
    assert len(X) == len(Y), "X and Y must have the same length"
    assert len(X_aug) == len(Y_aug), "X_aug and Y_aug must have the same length"
    X_set = set(tuple(row) for row in X)
    for i, row in enumerate(X_aug):
        if tuple(row) not in X_set:
            X.append(row)
            Y.append(Y_aug[i])
    return X, Y

def augment_documents(documents, randomizers={'PROFESSION': random_profession_generator(), 'NAME': random_name_generator(parts_min = 1, parts_max = 3,
                        lower =  False)}):
    """
    Returns a list of copies of documents with randomizers applied.
    Unchanged documents are not included.

    :param documents: list of documents
    :param randomizers: a dictionary of iterators where the key is the tag of the text that should be replaced with strings from the iterator.
    """
    documents_aug = []
    for doc in documents:
        doc = copy.deepcopy(doc)
        augmented = False
        for tag in doc['tags']:
            tag_label = tag['tag']
            tag_start, tag_end = int(tag['start']), int(tag['end'])
            tag_text = tag['text']
            if tag_label in randomizers:
                augmented = True
                new_text = next(randomizers[tag_label], tag_text)
                doc['text'] = doc['text'][:tag_start] + new_text + doc['text'][tag_end:]
                tag['text'] = new_text
        if augmented:
            documents_aug.append(doc)
    return documents_aug


# Probably deprecated
def randomize_tagged_text(X, Y):
    """
    Returns a copy of X with tagged tokens replaced by random strings.

    :param X: list of sequences of tokens
    :param Y: list of sequences of token labels
    """
    assert len(X) == len(Y)
    X_copy = [row[:] for row in X]
    for i in range(len(X)):
        for j in range(len(X[i])):
            include_classes = set([tags2index[tag] for tag in ['NAME', 'PROFESSION', 'LOCATION']]) # tags that can be replaced by random strings
            if Y[i][j] in include_classes:
                X_copy[i][j] = _random_string()
    return X_copy


def randomize_names(X, Y):
    """
    Returns a copy of X with names replaced by random names.

    :param X: list of sequences of tokens
    :param Y: list of sequences of token labels
    """
    names = []  # not a set since we can't set a random seed
    names_set = set()  # for quicker duplicate checking
    first_name_file = open(
        "Dictionaries/dictionary_first_names.txt", 'r', encoding='utf-8')
    dictionary_first_name = first_name_file.readlines()

    for line in dictionary_first_name:
        name = line.strip().lower()
        if name not in names_set and len(name) > 1:
            names.append(name)
            names_set.add(name)

    surname_file = open(
        "Dictionaries/dictionary_surnames.txt", 'r', encoding='utf-8')
    dictionary_surname = surname_file.readlines()

    for line in dictionary_surname:
        name = line.strip().lower()
        if name not in names_set and len(name) > 1:
            names.append(name)
            names_set.add(name)

    assert len(X) == len(Y)
    X_copy = [row[:] for row in X]
    for i in range(len(X)):
        for j in range(len(X[i])):
            if Y[i][j] == tags2index['NAME']:
                X_copy[i][j] = random.choice(names)
    return X_copy
