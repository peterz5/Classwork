START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram = defaultdict(int)
    bigram = defaultdict(int)
    trigram = defaultdict(int)


    for sent in training_corpus:
        uni = sent.strip().split() + [STOP_SYMBOL]
        bi = [START_SYMBOL] + sent.strip().split() + [STOP_SYMBOL]
        tri = [START_SYMBOL] + [START_SYMBOL] + sent.strip().split() + [STOP_SYMBOL]

        for a in uni:
            unigram[a] += 1

        for b in nltk.bigrams(bi):
            bigram[b] += 1

        for c in nltk.trigrams(tri):
            trigram[c] += 1

    size = sum(unigram.values())
    unigram_p = {key: math.log(float(val) / size, 2) for key, val in nltk.FreqDist(unigram).items()}

    unigram[START_SYMBOL] = len(training_corpus)
    bigram_p = {key: math.log(float(val) / unigram[key[0]], 2) for key, val in nltk.FreqDist(bigram).items()}

    bigram[(START_SYMBOL, START_SYMBOL)] = len(training_corpus)
    trigram_p = {key: math.log(float(val) / bigram[key[:2]], 2) for key, val in nltk.FreqDist(trigram).items()}
    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []

    for sent in corpus:
        lst = [START_SYMBOL, START_SYMBOL] 
        lst += sent.split() + [STOP_SYMBOL] 
        value = 0

        for i in range(2, len(lst)): 
            l = [lst[j] for j in range(i-n+1, i+1)]

            if len(l) > 1:
                x = tuple(l)
            else:
                x = l[0]
        
            if x in ngram_p:
                value += ngram_p[x]
            else:
                value = MINUS_INFINITY_SENTENCE_LOG_PROB

        scores.append(value)

    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    lamda = 1.0/3.0

    for sent in corpus:
        lst = [START_SYMBOL, START_SYMBOL] + sent.split() + [STOP_SYMBOL]
        uni = 0
        bi = 0
        tri = 0

        value = 0
        for i in range(2, len(lst)):
            uni = lst[i]
            bi = tuple(lst[i-1: i+1])
            tri = tuple(lst[i-2: i+1])

            if(not (tri in trigrams) and not(bi in bigrams)and not(uni in unigrams)):
                value = MINUS_INFINITY_SENTENCE_LOG_PROB
            
            uni = 2.0 ** unigrams.get(uni, MINUS_INFINITY_SENTENCE_LOG_PROB)
            bi = 2.0 ** bigrams.get(bi,MINUS_INFINITY_SENTENCE_LOG_PROB)
            tri = 2.0 ** trigrams.get(tri,MINUS_INFINITY_SENTENCE_LOG_PROB)
            
            x = lamda * (uni + bi + tri)

            x = math.log(x, 2)
            value += x

        scores.append(value)
    return scores

DATA_PATH = '/home/classes/cs477/data/' # absolute path to use the shared data
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'