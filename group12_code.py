"""
Final DS2500 Project
Song Generator - Takes in a variety of songs from an artist and generates a song based on their lyrics

Andrea Keiper, Jeremiah Payeur, Samantha Sobhian, Kaydence Lin

Credit to John Rachlin for Dot, mag, vecorize, and cosine_similarity functions
Credit to James Wenzel For Phyme library,
Credit to Sebleier For list of stop Words
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns
from textblob import TextBlob
import random
from Phyme import Phyme  # rhyming library from github user jameswenzel

ph = Phyme()
STOP_WORDS = 'stop_words.txt'  # sebleier/NLTK's list of english stopwords https://gist.github.com/sebleier/554280
ex_factor = 'ex_factor.txt'
doo_wop = 'doo_wop.txt'
ready_or_not = 'ready_or_not.txt'
fugeela = 'fugeela.txt'
zealots = 'zealots.txt'
killing_me_softly = 'killing_me_softly.txt'
august = 'august.txt'
betty = 'betty.txt'
cardigan = 'cardigan.txt'
epiphany = 'epiphany.txt'
exile = 'exile.txt'
hoax = 'hoax.txt'
illicit_affairs = 'illicit_affairs.txt'
invisible_string = 'invisible_string.txt'
mad_woman = 'mad_woman.txt'
mirrorball = 'mirrorball.txt'
my_tears_ricochet = 'my_tears_ricochet.txt'
peace = 'peace.txt'
seven = 'seven.txt'
the_1 = 'the_1.txt'
the_last = 'the_last_great_american_dynasty.txt'
this_is_me_trying = 'this_is_me_trying.txt'
bad_habit = 'bad_habit.txt'
infrunami = 'infrunami.txt'
fourreal = '4real.txt'
dark_red = 'dark_red.txt'
ryd = 'ryd.txt'
cocky_girl = 'cocky_girl.txt'
sunshine = 'sunshine.txt'
formation = 'formation.txt'
america_has = 'america_has_a_problem.txt'
irreplaceable = 'irreplaceable.txt'
cuff_it = 'cuff_it.txt'
single_ladies = 'single_ladies.txt'
brown_skin = 'brown_skin_girl.txt'
crazy_in_love = 'crazy_in_love.txt'
love_on_top = 'love_on_top.txt'
seven_eleven = '7_11.txt'
run_the_world = 'run_the_world.txt'


def read_txt(file):
    """
    name: read_txt
    parameter: txt file
    returns: long list of lines in the file
    """
    txt = []
    with open(file, 'r', encoding='UTF8') as infile:
        for line in infile:
            line = line.replace('"', '')
            line = line.replace(',', '')
            line = line.replace('(', '')
            line = line.replace(')', '')
            txt.append(line.replace("'", ""))

    return txt


def split_txt(dct):
    """
    name: split_txt
    parameter: dictionary with values as strings
    returns: a dictionary with split strings
    does: takes in a dictionary and reads it, and changes each value so that they are a list of lines
    """
    new_dct = {}
    for key, value in dct.items():
        value = [line for line in value if line != '']
        lst = []
        for line in value:
            lst.append(line)
        new_dct[key] = lst
    return new_dct


def artist_sentiment(dct, sent='polarity'):
    """
    name: speech_analysis
    parameter: dictionary with values as strings of text
    returns: a lst that correlates to average sentiment over time
    does: takes in a dictionary, finds the average length of the values, and finds average sentiment score of position
    """
    lst = []
    length = int(sum([len(value) for value in dct.values()]) / len([len(value) for value in dct.values()]))
    for i in range(length):
        count = 0
        sent_count = 0
        for value in dct.values():
            if len(value) > i:
                count += 1
                if sent == 'subjectivity':
                    sent_count += TextBlob(value[i]).sentiment.subjectivity
                if sent == 'polarity':
                    sent_count += TextBlob(value[i]).sentiment.polarity

        lst.append(sent_count / count)
    return lst


def n_grams(dct):
    """
    name: n_grams
    parameter: dictionary with values that are a list of lines
    returns: a list of tri-grams within that dictionary
    does: reads in a dictionary and iterates each values looking for all valid tri-grams
    """
    new = []
    for value in dct.values():
        for line in value:
            line = TextBlob(line)
            new.append(line.ngrams())
    return new


def start_words(dct):
    """
    name: start_words
    parameter: dictionary with values as a list of lines or sentences
    returns: a dictionary where keys are start words and values are their associated probability of occuring
    does: iterates through all lines of each value and finds how often each start word occurs and calculates its
    probability of occuring
    """
    start_dct = {}
    for value in dct.values():
        for line in value:
            line = line.split(' ')
            if line[0] in start_dct:
                start_dct[line[0]] += 1
            else:
                start_dct[line[0]] = 1
    return {key: value / sum([v for v in start_dct.values()]) for key, value in start_dct.items()}


def end_grams(dct):
    """
    name: end_grams
    parameter: dictionary that has values that are lists of lines or sentences
    returns: a list of tuples that are valid ways to end a sentence
    does: reads every line in every value and appends to the list a tuple of the last two words
    """
    end_context = []
    for value in dct.values():
        for line in value:
            line = line.strip()
            line = line.split(' ')

            if len(line) > 1:
                if (line[len(line) - 2], line[len(line) - 1]) not in end_context:
                    end_context.append((line[len(line) - 2], line[len(line) - 1]))

    return end_context


def count_ngrams(lst):
    """
    name: count_ngrams
    parameter: lst of n_grams
    returns: a dictionary where the keys are the n-grams and the value is how many times they occured
    """
    counter = {}
    for row in lst:
        for ngram in row:
            ngram = tuple(ngram)
            if ngram in counter:
                counter[ngram] += 1
            else:
                counter[ngram] = 1
    return counter


def counting_prob(context, ngrams_count):
    """
    name: counting_prob
    parameters: context is the previous two words ngram count is a dictionary where keys are n-grams and value is
    how many times they occur
    returns: a filtered dictionary, only including keys and values where the keys are the same as the context
    does: looks at all the ngrams and determines whether the context is 1 or multiple words then, for all the
    contexts that hold the same starting values as the key, find the next occuring word and count it
    """
    word_count = {}

    for key, value in ngrams_count.items():

        if context == key[0]:
            if key[1] in word_count:
                word_count[key[1]] += 1
            else:
                word_count[key[1]] = 1

        elif context == key[:len(key) - 1]:
            if key[len(key) - 1] in word_count:
                word_count[key[len(key) - 1]] += 1
            else:
                word_count[key[len(key) - 1]] = 1
    return word_count


def prob(context, ngrams_count):
    """
    name: prob
    parameter: context the last 1 or 2 words of a line, ngrams_count, a dictionary with ngrams as keys and their counts
    as values
    returns: probability of a word coming next based on previous words
    does: finds all valid next words and counts them then finds their relative probability of occuring
    """
    word_count = counting_prob(context, ngrams_count)
    total = sum([v for v in word_count.values()])
    return {key: value / total for key, value in word_count.items()}


def generate_line(start_word, end_gram, ngrams_count, min_length=3, max_length=20):
    """
    name: generate_line
    parameters: start_word, dictionary of start words and associated probabilitys, end_gram, a list of tuples that are
    valid ways to end a line, ngrams_count, a dictionary with ngrams as keys and count as values, min_length an int
    is the minimum length of a line and max_length an int is the maximum length of a line
    returns: a string that represents a line in a song
    does: creates a song line using context and probability of a specified length
    """
    line = ''
    word = random.choices([k for k in start_word.keys()], [v for v in start_word.values()])[0]
    line += word
    count = 0
    context = [word]
    while (count < min_length or tuple(context[len(context) - 2:len(context)]) not in end_gram) and count < max_length:

        if count == 0:
            probability = prob(word, ngrams_count)
        else:
            probability = prob(tuple(context[len(context) - 2:len(context)]), ngrams_count)
        if probability == {}:
            break
        new_word = random.choices([k for k in probability.keys()], [v for v in probability.values()])[0]

        context.append(new_word)
        line += ' ' + new_word
        count += 1

    return line


def rhyme_order(rhyme_scheme):
    """
    name: rhyme_order
    parameter: rhyme_schem, a string of letters that represents a rhyme scheme
    return: a tuple, the length of the rhyme scheme in the first entry and a dictionary of the letters and what line
    number they correspond to
    does: takes in a rhyme scheme and finds how long it is and the order of the rhyme
    """
    order = {}
    for i in range(len(rhyme_scheme)):
        if rhyme_scheme[i] in order:
            order[rhyme_scheme[i]].append(i)
        else:
            order[rhyme_scheme[i]] = [i]

    return len(rhyme_scheme), order


def find_rhyme_letter(length, order, num):
    """
    name: find_rhyme_letter
    parameters: length, an int of how long the rhyme scheme is, an order which is a dictionary that represents
    what numbers go to which letter, and a num which is an int of what overall line of generation the program is on
    return: a letter that is the rhyme
    """
    for j in range(length):
        if num % length == j:
            for key, value in order.items():
                if j in value:
                    return key


def doTheyRhyme(word1, word2):
    """
    note: uses jameswenzels rhyming library from github, Phyme, to find a family of rhymes
    name: doTheyRhyme
    parameters: two words both strings
    returns: True or false
    does: tests to see if two words rhyme and returns a bool
    """
    if word1 == word2:
        return False
    try:
        words = ph.get_family_rhymes(word1)
    except KeyError:
        return False
    for list in words.values():
        if word2 in list:
            return True
    else:
        return False


def create_song_basic(sent, ngrams_count, start_word, end_gram, sent_tolerance=.05,
                      rhyme='ABAB', chorus_num=2):
    """
    name_create_song_basic
    parameters: sentiment, list of sentiment over time, ngrams_count, dictionary of ngrams and their counts, start_word,
    dictionary of start words and their probability, end_gram, list of ending tuples, sent-tolerance, a float that
    outlines how close to the actual artist sentiment you want to be, rhyme, a string of letters representing a rhyme
    scheme, chorus_num, how many total choruss you want
    returns: a song that is a string of text
    does: creates a song line by line and only accepts the line if it both fits the rhyme scheme and fits within the
    sentiment tolerance at the given point in the song. The rhyme can be overwritten if there have been over 1000 lines
    tried that don't fit both parameters
    """
    song = ''
    chorus = ''
    length, order = rhyme_order(rhyme)
    rhyming_words = {}

    for i in range(2 * (chorus_num + 1) * length):
        if i % (2 * length) == 0 and i / (2 * length) != 1:
            song += chorus
        sent_index = int(i * len(sent) / (2 * (chorus_num + 1) * length))
        sent_num, count, end_line_rhyme, rhyme_count = 2, 0, False, i // length
        rhyme_letter = find_rhyme_letter(length, order, i)

        while (
                sent_num > sent[sent_index] + sent_tolerance or sent_num < sent[sent_index] - sent_tolerance) or (not
        end_line_rhyme):
            line = generate_line(start_word, end_gram, ngrams_count)
            sent_num = TextBlob(line).sentiment.polarity
            words = line.split()
            end_line_rhyme = False
            if i % length == order[rhyme_letter][0]:
                end_line_rhyme = True
                rhyming_words[rhyme_letter] = words[len(words) - 1]

            elif doTheyRhyme(words[len(words) - 1], rhyming_words[rhyme_letter]):
                end_line_rhyme = True

            count += 1
            if count > 1000:
                end_line_rhyme = True
        if (rhyme_count + 1) / 3 == 1:
            chorus += line + '\n'
        song += line + '\n'

    return song


def filter_word_from_dct(dct, words):
    """
    name: filter_word_from_dct
    parameters: dct dictionary, words a list of words to take out of dictionary values
    returns: a dictionary that has a value of filtered words
    """
    filtered_dct = {}
    words = [word.strip() for word in words]
    for key, value in dct.items():
        value =value.replace('\n', ' ')
        value = value.split(' ')
        filtered_dct[key] = [word for word in value if word not in words]
    return filtered_dct


def vectorize(words, unique):
    """
    name: vectorize
    parameter: words: list of a users words, unique list of all unique words
    returns: a vector with counter values that give a value to how many of the certain unique word was in the user list
    does: convert into a vectors from user words and unique words
    """
    return [Counter(words)[word] for word in unique]


def mag(v):
    """
    name: mag
    parameters: vector like list
    returns: magnitude of a vector
     """
    return (sum([i ** 2 for i in v])) ** .5


def dot(u, v):
    """
    name: dot
    parameters: u, v both vector like lists of same size
    returns: dot product of two vectors
    """
    return sum([i * j for i, j in zip(u, v)])


def cosine_similarity(u, v):
    """
    name: cosine_similarity
    parameters: u, v both vector like lists of same size
    returns cosine similarity between two vectors
    """
    if mag(u) != 0 and mag(v) != 0:
        return dot(u, v) / (mag(u) * mag(v))
    else:
        return


def cosine_similarity_array(dct, unique):
    """
    name: cosine_similarity_array
    parameters: dct: dictionary, unique, a set of unique words
    returns nothing, plots a heatmap of which keys are most similar to each other
    """
    lst = list(dct.items())
    arr = np.ones((len(lst), len(lst)), dtype=float)
    x_labels = []

    for i in range(len(lst)):
        vi = vectorize(lst[i][1], unique)
        x_labels.append(lst[i][0])
        for j in range(i + 1, len(lst)):
            vj = vectorize(lst[j][1], unique)

            arr[i, j] = cosine_similarity(vi, vj)
            arr[j, i] = arr[i, j]

    sns.heatmap(arr, xticklabels=x_labels, yticklabels=x_labels)
    plt.show()
    return


def unique_words_in_dct(dct, most_common=None):
    """
    name: unique_words_in_dct
    parameter: dictionary a dictionary with values as list of words, most_common optional is in an
    return: returns a unique set of words
    """
    words = []
    if most_common is not None:
        for value in dct.values():
            top_n = Counter(value).most_common(most_common)
            top_n = [item[0] for item in top_n]
            value = [word for word in value if word in top_n]
            words.extend(value)

    else:
        for value in dct.values():
            words.extend(value)

    return set(words)


def read_song(file):
    """
    name: read_txt
    parameter: txt file
    returns: long string of text
    """
    txt = ''
    with open(file, 'r', encoding='UTF8') as infile:
        for line in infile:
            txt += line
    return txt


def song_from_dct(dct, sent='polarity', chorus=2, tolerance=.05, rhyme='ABAB'):
    """
    name: song_from_dct
    parameters: dct, a dictionary of songs with keys as titles and values as songs, sent, a string either 'polarity' or
    'subjectivity' of what sentiment we want to measure, chorus, an int and number of choruss, tolerance, a float that
    is how close we want to match the artists sentiment, rhyme, a string of letters and the rhyme scheme we want to
    return: song
    does: takes a dictionary and calls the create_song_basic function fulfilling all parameters, and then prints the
    song
    """
    sent = artist_sentiment(split_txt(dct), sent=sent)
    counted_ngrams = count_ngrams(n_grams(dct))
    song = create_song_basic(sent, counted_ngrams, start_words(dct), end_grams(dct),
                             sent_tolerance=tolerance,
                             rhyme=rhyme, chorus_num=chorus)
    print(song)
    return song


def cos_similarity_from_dct(dct, name):
    """
    name: cos_similarity_from_dct
    parameters: dct: a dictionary of songs with keys as titles and values as lyrics, name, a string of what is to be
    the title of the graph
    returns nothing, plots a graph
    does: calls necessary functions to plot a cosine similarity graph from a dictionary
    """
    dct = filter_word_from_dct(dct, read_txt(STOP_WORDS))
    unique_words = unique_words_in_dct(dct)
    title = str(name) + ' Cosine Similarity Map'
    plt.title(title)
    cosine_similarity_array(dct, unique_words)
    return


def sentiment_plot_from_dct(dct, name):
    """
    name: cos_similarity_from_dct
    parameters: dct: a dictionary of songs with keys as titles and values as lyrics, name, a string of what is to be
    the title of the graph
    returns nothing, plots a graph of sentiment over time
    does: calls necessary functions to plot a sentiment of artist
    """
    title = 'Sentiment over Time of ' + name
    plt.title(title)
    plt.xlabel('Time throughout song')
    plt.ylabel('Polarity (Higher Numbers are more Positive)')
    sent = artist_sentiment(dct)
    plt.axhline(y=0, color='r')
    plt.plot(sent)
    plt.show()
    return


def txt_to_line(txt):
    return txt.split('\n')

def main():

    # Plot all artist cosine similarity
    all_artists = {'August (Taylor)': read_song(august), 'Doo Wop (Lauryn Hill)': read_song(doo_wop),
                   'Bad Habit (Steve Lacey)': read_song(bad_habit), 'Formation (Beyonce)': read_song(formation)}
    cos_similarity_from_dct(all_artists, 'All Artists')

    Taylor = {'august': read_txt(august), 'betty': read_txt(betty), 'cardigan': read_txt(cardigan), 'epiphany':
        read_txt(epiphany), 'exile': read_txt(exile), 'hoax': read_txt(hoax), 'illicit_affairs':
                  read_txt(illicit_affairs), 'invisible_string': read_txt(invisible_string),
              'mad_woman': read_txt(mad_woman),
              'mirrorball': read_txt(mirrorball), 'my_tears_ricochet': read_txt(my_tears_ricochet),
              'peace': read_txt(peace), 'seven': read_txt(seven), 'the_1': read_txt(the_1),
              'the_last': read_txt(the_last),
              'this_is_me_trying': read_txt(this_is_me_trying)}

    # generate song and plot sentiment over time
    song = song_from_dct(Taylor, sent='polarity', tolerance=.05, rhyme='AABCBC', chorus=2)
    sentiment_plot_from_dct(Taylor, 'Taylor Swift')
    sentiment_plot_from_dct({'Generated Song': txt_to_line(song)}, 'Taylor Swift Generated Song')

    Taylor = {'august': read_song(august), 'betty': read_song(betty), 'cardigan': read_song(cardigan),
              'epiphany': read_song(epiphany), 'exile': read_song(exile), 'hoax': read_song(hoax),
              'illicit_affairs': read_song(illicit_affairs), 'invisible_string': read_song(invisible_string),
              'mad_woman': read_song(mad_woman), 'mirrorball': read_song(mirrorball),
              'my_tears_ricochet': read_song(my_tears_ricochet), 'peace': read_song(peace), 'seven': read_song(seven),
              'the_1': read_song(the_1), 'the_last': read_song(the_last),
              'this_is_me_trying': read_song(this_is_me_trying), 'Generated Song': song}

    # generate cosine similarity graph
    cos_similarity_from_dct(Taylor, 'Taylor Swift')

    Lauryn = {'Ex-Factor': read_txt(ex_factor), 'Doo Wop': read_txt(doo_wop), 'Ready or Not': read_txt(ready_or_not),
              'Fu Gee La': read_txt(fugeela), 'Zealots': read_txt(zealots),
              'Killing Me Softly': read_txt(killing_me_softly)}

    # generate song and plot sentiment over time
    song = song_from_dct(Lauryn)
    sentiment_plot_from_dct(Lauryn, 'Lauryn Hill')
    sentiment_plot_from_dct({'Generated Song': txt_to_line(song)}, 'Lauryn Hill Generated Song')

    Lauryn = {'Ex-Factor': read_song(ex_factor), 'Doo Wop': read_song(doo_wop), 'Ready or Not': read_song(ready_or_not),
              'Fu Gee La': read_song(fugeela), 'Zealots': read_song(zealots), 'Generated Song': song,
              'Killing Me Softly': read_song(killing_me_softly)}

    # generate cosine similarity graph
    cos_similarity_from_dct(Lauryn, 'Lauryn Hill')

    Steve = {'Bad Habit': read_txt(bad_habit), 'Infrunami': read_txt(infrunami), '4Real': read_txt(fourreal),
             'Dark Red':
                 read_txt(dark_red), 'Ryd': read_txt(ryd), 'Cocky Girl': read_txt(cocky_girl),
             'Sunshine': read_txt(sunshine)}

    # generate song and plot sentiment
    song = song_from_dct(Steve)
    sentiment_plot_from_dct(Steve, 'Steve Lacey')
    sentiment_plot_from_dct({'Generated Song': txt_to_line(song)}, 'Steve Lacey Generated Song')

    Steve = {'Bad Habit': read_song(bad_habit), 'Infrunami': read_song(infrunami), '4Real': read_song(fourreal),
             'Dark Red':
                 read_song(dark_red), 'Ryd': read_song(ryd), 'Cocky Girl': read_song(cocky_girl),
             'Sunshine': read_song(sunshine), 'Generated Song': song}

    # generate cosine similarity graph
    cos_similarity_from_dct(Steve, 'Steve Lacy')

    Beyonce = {'formation': read_txt(formation), 'america_has': read_txt(america_has), 'irreplaceable':
        read_txt(irreplaceable), 'cuff_it': read_txt(cuff_it), 'single_ladies': read_txt(single_ladies),
               'brown_skin': read_txt(brown_skin), 'crazy_in_love': read_txt(crazy_in_love), 'love_on_top':
                   read_txt(love_on_top), 'seven_eleven': read_txt(seven_eleven),
               'run_the_world': read_txt(run_the_world)}

    # generate song and plot sentiment
    song = song_from_dct(Beyonce)
    sentiment_plot_from_dct(Beyonce, 'Beyonce')
    sentiment_plot_from_dct({'Generated Song': txt_to_line(song)}, 'Beyonce Generated Song')

    Beyonce = {'formation': read_song(formation), 'america_has': read_song(america_has), 'irreplaceable':
        read_song(irreplaceable), 'cuff_it': read_song(cuff_it), 'single_ladies': read_song(single_ladies),
               'brown_skin': read_song(brown_skin), 'crazy_in_love': read_song(crazy_in_love), 'love_on_top':
                   read_song(love_on_top), 'seven_eleven': read_song(seven_eleven),
               'run_the_world': read_song(run_the_world), 'Generated Song': song}

    # generate cosine similarity graph
    cos_similarity_from_dct(Beyonce, 'Beyonce')

if __name__ == '__main__':
    main()
