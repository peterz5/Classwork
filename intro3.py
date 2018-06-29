import sys
import nltk
import math
import time
import solutionsB as B
import itertools

from collections import defaultdict
from collections import deque
from collections import OrderedDict

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    for sent in brown_train:
        
        words = [START_SYMBOL, START_SYMBOL]
        tags = [START_SYMBOL, START_SYMBOL]

        for i in sent.strip().split(' '):
            x = i.find('/')
            while (i[x+1:].upper() != i[x+1:]) or not i[x+1].isupper() and not i[x+1] == '.':
                x += 1 + i[x+1:].find('/')
 
            words.append(i[:x])
            tags.append(i[x+1:])

        words.append(STOP_SYMBOL)
        tags.append(STOP_SYMBOL)

        brown_words.append(words)
        brown_tags.append(tags)

    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}

    bigrams = defaultdict(int)
    trigrams = defaultdict(int)

    for sent in brown_tags:
        for i in nltk.bigrams(sent):
            bigrams[i] += 1

        for i in nltk.trigrams(sent):
            trigrams[i] += 1

    q_values = {key: math.log(float(val)/ bigrams[key[0:2]], 2) for key, val in nltk.FreqDist(trigrams).items()}
    
    return q_values

#TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates the tag trigrams in reverse.  In other words, instead of looking at the probabilities that the third tag follows the first two, look at the probabilities of the first tag given the next two.
# Hint: This code should only differ slightly from calc_trigrams(brown_tags)
def calc_trigrams_reverse(brown_tags):
    q_values = {}

    bigrams = defaultdict(int)
    trigrams = defaultdict(int)

    for sent in brown_tags:
        
        lst = [START_SYMBOL, START_SYMBOL] + list(reversed(sent))[1:len(sent)-2]+ [STOP_SYMBOL]
        for i in nltk.bigrams(lst):
            bigrams[i] += 1
        
        for i in nltk.trigrams(lst): 
            trigrams[i] += 1

    for key, val in nltk.FreqDist(trigrams).items():
        q_values[key] = math.log(float(val)/ bigrams[key[0:2]], 2) 

    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])

    count = defaultdict(int)
    for sent in brown_words:
        for i in sent:
            count[i] += 1

    for i,j in nltk.FreqDist(count).items():
        if j > RARE_WORD_MAX_FREQ:
            known_words.add(i)

    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    for sent in brown_words:
        for i in range(len(sent)):
            if sent[i] not in known_words:
                sent[i] = RARE_SYMBOL
        brown_words_rare.append(sent)

    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    taglist = set([])

    tups = defaultdict(int)
    tags = defaultdict(int)

    for sent in brown_tags:
        for tag in sent:
            tags[tag] += 1

    for sent1, sent2 in zip(brown_words_rare, brown_tags):
        for word, tag in zip(sent1, sent2):
            tups[(word,tag)] +=1

    for (word,tag), count in nltk.FreqDist(tups).items():
        e_values[word, tag] = math.log(float(count)/ tags[tag], 2)
    
    taglist = set(tags)
    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):

    tagged = []
    pi = defaultdict(float)
    bp = {}

    pi[(-1,START_SYMBOL,START_SYMBOL)] = 0.0

    bp[(-1,START_SYMBOL,START_SYMBOL)] = START_SYMBOL
    
    for sent in brown_dev_words:
        
        words = [s if s in known_words else RARE_SYMBOL for s in sent]      

        for t in taglist:
            if(words[0], t) in e_values: 
                pi[(0, START_SYMBOL, t)] = pi[(-1,START_SYMBOL,START_SYMBOL)] + q_values.get((START_SYMBOL, START_SYMBOL, t), LOG_PROB_OF_ZERO) + e_values.get((words[0], t), LOG_PROB_OF_ZERO)
                bp[(0, START_SYMBOL, t)] = START_SYMBOL

        for (w, u) in itertools.product(taglist, taglist):
            if (words[0], w) in e_values and (words[1], u) in e_values:
                key = (START_SYMBOL, w, u)
                pi[(1, w, u)] = pi.get((0, START_SYMBOL, w), LOG_PROB_OF_ZERO) + q_values.get(key, LOG_PROB_OF_ZERO) + e_values.get((words[1], u), LOG_PROB_OF_ZERO)
                bp[(1, w, u)] = START_SYMBOL

        for i in range (2, len(words)):
            for (u, v) in itertools.product(taglist, taglist):
                maxp = -float('Inf')
                maxt = ''

                for w in taglist:
                    if (words[i-2], w) in e_values and (words[i-1], u) in e_values and (words[i], v) in e_values:
                        prob = pi.get((i-1, w, u), LOG_PROB_OF_ZERO) + q_values.get((w,u,v), LOG_PROB_OF_ZERO) + e_values.get((words[i], v), LOG_PROB_OF_ZERO)
                        if(prob > maxp):
                            maxp = prob
                            maxt = w
                pi[(i,u,v)] = maxp
                bp[(i,u,v)] = maxt

        maxp = float('-Inf')
        
        for (u,v) in itertools.product(taglist,taglist):
            prob = pi.get((len(sent)-1, u, v),LOG_PROB_OF_ZERO) + q_values.get((u,v, STOP_SYMBOL),LOG_PROB_OF_ZERO) 
            
            if prob > maxp:
                maxp = prob
                maxu = u
                maxv = v
        
        tags = []
        tags.append(maxv)
        tags.append(maxu)
        count = 0
        for k in range(len(sent) - 3, -1, -1):
            tags.append(bp[(k + 2, tags[count+1], tags[count])])
            count +=1
        tagged_sentence = ''
      
        tags.reverse()
        
        for k in range(0, len(sent)):
            tagged_sentence = tagged_sentence + sent[k] + "/" + str(tags[k]) + " "
        tagged_sentence += "\n"
        tagged.append(tagged_sentence)
    return tagged 

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    
    defaultTag = nltk.DefaultTagger('NOUN')
    bigramTag = nltk.BigramTagger(training, backoff=defaultTag)
    trigramTag = nltk.TrigramTagger(training, backoff = bigramTag)

    for sent in brown_dev_words:
        tagged_sentence = trigramTag.tag(sent)

        sentence = ''
        for tag in tagged_sentence:
            ''.join((sentence,tag[0], '/', tag[1], ' '))
        sentence = sentence[:-1] + '\n' 

        tagged.append(sentence)

    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = '/home/classes/cs477/data/'
OUTPUT_PATH = 'output/'


#This is the same as the code in Part B, but with a Spanish corpus being evaluated 

def main():
    time.clock()

    infile = open(DATA_PATH + "wikicorpus_tagged_train.txt", "r")

    train = infile.readlines()
    infile.close()

    words, tags = B.split_wordtags(train)

    if len(sys.argv) > 1 and sys.argv[1] == "-reverse":
        q_values = B.calc_trigrams_reverse(tags)
    else:
        q_values = B.calc_trigrams(tags)

    B.q2_output(q_values, OUTPUT_PATH + 'C2.txt')

    known_words = B.calc_known(words)

    words_rare = B.replace_rare(words, known_words)

    B.q3_output(words_rare, OUTPUT_PATH + "C3.txt")

    e_values, taglist = B.calc_emission(words_rare, tags)

    B.q4_output(e_values, OUTPUT_PATH + "C4.txt")

    del train
    del words_rare

    infile = open(DATA_PATH + "wikicorpus_dev.txt", "r")

    dev = infile.readlines()
    infile.close()

    dev_words = []
    for sentence in dev:
        dev_words.append(sentence.split(" ")[:-1])

    viterbi_tagged = B.viterbi(dev_words, taglist, known_words, q_values, e_values)

    B.q5_output(viterbi_tagged, OUTPUT_PATH + 'C5.txt')

    nltk_tagged = B.nltk_tagger(words, tags, dev_words)

    B.q6_output(nltk_tagged, OUTPUT_PATH + 'C6.txt')

    print "Part C time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
