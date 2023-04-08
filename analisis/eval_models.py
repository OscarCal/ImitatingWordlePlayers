from random import choice, sample, randrange, random, shuffle
from string import ascii_lowercase
import numpy as np
from itertools import permutations
import multiprocessing as mult
import eval_models as evalmod

valid_inputs = []
with open("valid-wordle-words.txt") as f:
    valid_inputs = f.read().splitlines()
comparer = set(valid_inputs)
valid_mutations = {}
alphabet = list(map(chr, range(97, 123)))
for word in valid_inputs:
    valid_mutations[word] = set()
    for i in range(len(word)):
        for letter in alphabet:
            if (letter != word[i]) and ((word[:i] + letter + word[i+1:]) in comparer):
                valid_mutations[word].add(word[:i] + letter + word[i+1:])

valid_inputs = []
with open("valid-wordle-words.txt") as f:
    valid_inputs = f.read().splitlines()
comparer = set(valid_inputs)
valid_permutations = {}
changes = [[i,j] for i in range(5) for j in range(i+1,5)]
for word in valid_inputs:
    valid_permutations[word] = set()
    for i,j in changes:
        swap = list(word)
        swap[i], swap[j] = swap[j], swap[i]
        swap = ''.join(swap)
        if swap != word and swap in comparer:
            valid_permutations[word].add(swap)               

valid_inputs = []
with open("valid-wordle-words.txt") as f:
    valid_inputs = f.read().splitlines()
comparer = set(valid_inputs)
valid_inversions = {}
inversions = [[i,j]for i in range(4) for j in range(i+1,5)]
for word in valid_inputs:
    valid_inversions[word] = set()
    for i,j in inversions:
        inv = list(word)
        inv[i:j+1] = reversed(word[i:j+1])
        inv = ''.join(inv)
        if inv != word and inv in comparer:
            valid_inversions[word].add(inv)
alphabet = np.asarray(list(map(chr, range(97, 123))))
def get_keys(alphabet, length):
    if len(alphabet) == 1:
        return np.asarray([alphabet[0]*length])
    elif length == 1:
        return alphabet
    else:
        res = np.asarray([])
        for i in range(len(alphabet)):
            aux = np.asarray(get_keys(alphabet[i:], length - 1))
            res = np.concatenate((res,np.char.add(np.asarray([alphabet[i]]*len(aux)), 
                                         aux)), axis = 0)
        return res
def get_words(key):
    res = set()
    p = permutations(key)
    for word in list(p):
        aux = ''.join(word)
        if aux in comparer:
            res.add(aux)
    return res


keys = get_keys(alphabet, 5)
words_from_letters = {key : get_words(key) for key in keys}

frecs_new = np.load('adjusted_word_frequency_per_cluster.npy', allow_pickle = True)
frec0_new = frecs_new[0]
frec1_new = frecs_new[1]
frec2_new = frecs_new[2]

frecs = np.load('new_word_frequency_per_cluster.npy', allow_pickle = True)
frec0 = frecs[0]
g0_sorted = sorted(frec0.items(), key=lambda x:x[1], reverse = True)
g0_top = g0_sorted[:30]
g0_top = np.asarray(g0_top)
top_vals = g0_top[:,1].astype(float)
top_vals = top_vals/top_vals.sum()
top_labels = g0_top[:,0]

#dictionary of games and target words
game_dict = {}
with open("wordle-words-by-day2.txt") as f:
    for line in f:
        line = line.strip()
        game_dict.update({int(line[0:3]) : line[4::]})
daily_words = list(game_dict.values())

from random import choice, sample, randrange, random, shuffle
from string import ascii_lowercase
import numpy as np
from itertools import permutations
import multiprocessing as mult

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def check_words(input_word ,target):
    input_word = input_word.lower()
    target = target.lower()
    sequence = ['⬜']*5
    selected = [False]*5
    for i in range(0,5):
        if input_word[i] == target[i]:
            sequence[i] = '🟩'
            selected[i] = True
    for i in range(0,5):
        if input_word[i] != target[i]:
            indexes = find(target, input_word[i])
            for index in indexes:
                if not selected[index]:
                    selected[index] = True
                    sequence[i] = '🟨'
                    break
    aux = ""
    return aux.join(sequence)

def fitness(word, game_state):
    sum = 0
    for guess, result in game_state:
        match = result.count('🟩')
        misplaced =  result.count('🟨')
        result = check_words(guess, word)
        pos_match = result.count('🟩')
        pos_misplaced = result.count('🟨')
        sum += abs(pos_match - match) + abs(pos_misplaced - misplaced) + abs(match + misplaced - pos_match - pos_misplaced)
    return sum


def first_word(labels, vals):
    return labels[np.random.choice(len(labels), p = vals)]

def update_dict(dict_frec):
    frecs = np.asarray(list(dict_frec.values()))
    frecs = frecs/frecs.sum()
    return dict(zip(dict_frec.keys(), frecs))

def mutate2(word, dict_frec, game_state, prob=0.03):
    if random() < prob and len(valid_mutations[word]) > 0:
        fit = 0
        res = word
        for mut in valid_mutations[word]:
            if mut in dict_frec.keys() and fitness(mut,game_state) > fit:
                fit = fitness(mut,game_state)
                res = mut
        return res
    return word

def permute2(word, dict_frec, game_state, prob=0.03):
    if random() < prob and len(valid_permutations[word]) > 0:
        fit = 0
        res = word
        for mut in valid_permutations[word]:
            if mut in dict_frec and fitness(mut,game_state) > fit:
                fit =fitness(mut,game_state)
                res = mut
        return res
    return word

def invert2(word, dict_frec, game_state, prob=0.02):
    if random() < prob and len(valid_inversions[word]) > 0:
        fit = 0
        res = word
        for inv in valid_inversions[word]:
            if inv in dict_frec and fitness(inv,game_state) > fit:
                fit = fitness(inv,game_state)
                res = inv
        return res
    return word


def get_keys(alphabet, length):
    if len(alphabet) == 1:
        return np.asarray([alphabet[0]*length])
    elif length == 1:
        return alphabet
    else:
        res = np.asarray([])
        for i in range(len(alphabet)):
            aux = np.asarray(get_keys(alphabet[i:], length - 1))
            res = np.concatenate((res,np.char.add(np.asarray([alphabet[i]]*len(aux)), 
                                         aux)), axis = 0)
        return res

def get_words(key):
    res = set()
    p = permutations(key)
    for word in list(p):
        aux = ''.join(word)
        if aux in comparer:
            res.add(aux)
    return res

def crossover2(parent1, parent2, dict_prob, game_state, prob=0.5):
    child1, child2 = parent1, parent2
    if random() < prob:
        split = randrange(1, len(parent1)-1)
        child1 = parent1[:split] + parent2[split:]
        child2 = parent2[:split] + parent1[split:]
        if child1 not in dict_prob:
            key1 = ''.join(sorted(child1))
            if len(words_from_letters[key1]) > 0:
                fit = 0
                child1 = np.random.choice(a = np.asarray(list(dict_prob.keys())))
                for cand1 in words_from_letters[key1]:
                    if cand1 in dict_prob and fitness(cand1, game_state) > fit:
                        fit = fitness(cand1, game_state)
                        child1 = cand1
        if child2 not in dict_prob:
            key1 = ''.join(sorted(child2))
            if len(words_from_letters[key1]) > 0:
                fit = 0
                child2 = np.random.choice(a = np.asarray(list(dict_prob.keys())))
                for cand1 in words_from_letters[key1]:
                    if cand1 in dict_prob and fitness(cand1, game_state) > fit:
                        fit = fitness(cand1, game_state)
                        child2 = cand1
    return [child1, child2]


def selection2(fitnesses, population, k=20, keep_size=True):
    if all(fit == 0 for fit in fitnesses):
        return np.random.choice(a=population, replace = True, size = len(fitnesses))
    props = np.asarray(fitnesses)/np.asarray(fitnesses).sum()
    return np.random.choice(a=population, p = props, replace = True, size = len(fitnesses))

def remove_impossible_words(game_state, list_frec):
    guess, result = game_state[-1]
    correct_letters = [(guess[i], i) for i in range(len(guess)) if result[i] == '🟩']
    impossible_words = [word for word in list_frec.keys() if (any(word[i] != guess for guess, i in correct_letters))]
    new_list = {key: value for key,value in list_frec.items() if key not in impossible_words}
    new_list.pop(result,None)
    incorrect_letters = [(guess[i], i) for i in range(len(guess)) if result[i] == '⬜']
    impossible_words = [word for word in new_list.keys() if (any(word[i] == guess for guess, i in incorrect_letters))]
    for word in impossible_words:
        new_list.pop(word)
    misplaced_letters = [(guess[i], i) for i in range(len(guess)) if result[i] == '🟨']
    misplaced_words = [word for word in new_list.keys() if (any(word[i] == guess for guess, i in misplaced_letters))]
    for word in misplaced_words:
        new_list.pop(word)
    invalid_words = [word for word in new_list.keys() if (any(guess not in word for guess, _ in misplaced_letters))]
    for word in invalid_words:
        new_list.pop(word)
    return new_list


def find_best_word(eligible_words, game_state, dict_frec):
    fit = float('inf')
    frec = -1
    best = ''
    for word in eligible_words:
        aux = fitness(word, game_state)
        aux2 = dict_frec[word]
        if aux < fit or (aux == fit and aux2 > frec):
            best = word
            fit = aux
            frec = aux2
    return best


def wordle_genetic(game_state, guess_count, pop_size, dict_frec, max_gen, tour_size, crossover_prob,
                   mutate_prob, permute_prob, invert_prob, top_vals, top_labels):
    
    if guess_count == 0:
        guess = first_word(top_labels, top_vals)
        dict_frec.pop(guess)
        return guess, dict_frec
    
    dict_frec = remove_impossible_words(game_state, dict_frec)
    possible_guesses = np.asarray(list(dict_frec.keys()))
    frecs = np.asarray(list(dict_frec.values()))

    # create population
    sample_size = min(pop_size, len(possible_guesses))
    population = np.random.choice(a = possible_guesses, size = sample_size , replace = False)
    eligible_words = set()
    generation = 0

    # do genetic iterations
    while generation < max_gen:
        # selection
        fitnesses = [fitness(p, game_state) for p in population]
        selected = selection2(fitnesses, population, k=tour_size)
        # new generation
        if sample_size > 1:
            new_pop = []
            for p1, p2 in zip(selected[0::2], selected[1::2]):
                for c in crossover2(p1, p2, dict_frec, game_state , prob=crossover_prob):
                    if (c in eligible_words) or (c not in possible_guesses):
                        new_pop.append(np.random.choice(a = possible_guesses))
                    else:
                        c = mutate2(c, dict_frec, game_state, prob=mutate_prob)
                        c = permute2(c, dict_frec, game_state, prob=permute_prob)
                        c = invert2(c, dict_frec, game_state, prob=invert_prob)
                        new_pop.append(c)

            population = new_pop
        eligible_words.update(population)

        generation += 1

    # choose word in eligible_words with maximum
    best_word = find_best_word(eligible_words, game_state, dict_frec)
    dict_frec.pop(best_word)
    return best_word, dict_frec


def tryout(target, num, frecs, frecs_new, crossover_prob = 0.5,  mutate_prob = 0.03, permute_prob = 0.03, invert_prob = 0.02, pop_size = 150):
    #Prep sets
    frec0 = frecs[num]
    dict_frec = frec0
    g0_sorted = sorted(frec0.items(), key=lambda x:x[1], reverse = True)
    g0_top = g0_sorted[:10]
    g0_top = np.asarray(g0_top)
    top_vals = g0_top[:,1].astype(float)
    top_vals = top_vals/top_vals.sum()
    top_labels = g0_top[:,0]
    game_state = []
    dict_frec = frecs_new[num]
    frecs = np.asarray(list(dict_frec.values()))
    frecs = (frecs + abs(min(frecs)))
    frecs = frecs/frecs.sum()
    dict_frec = dict(zip(dict_frec.keys(), frecs))
    word, dict_frec = wordle_genetic(game_state = game_state, guess_count = 0, pop_size = pop_size, 
                                         dict_frec = dict_frec, max_gen = 100, tour_size = 40,
                                        crossover_prob = crossover_prob,  mutate_prob = mutate_prob, 
                                        permute_prob = permute_prob, invert_prob = invert_prob,
                                         top_vals = top_vals, top_labels = top_labels)
    check_words(word, target)
    game_state.append((word, check_words(word, target)))
    #print(str(game_state[-1][1]) + " " +game_state[-1][0])
    i = 1
    while game_state[-1][1] != "🟩🟩🟩🟩🟩" and i <= 5:
        dict_frec = update_dict(dict_frec)
        word, dict_frec = wordle_genetic(game_state = game_state, guess_count = i, pop_size = 150, 
                                         dict_frec = dict_frec, max_gen = 100, tour_size = 40,
                                        crossover_prob = 0.5,  mutate_prob = 0.03, permute_prob = 0.03, invert_prob = 0.02,
                                         top_vals = top_vals, top_labels = top_labels)
        check_words(word, target)
        game_state.append((word, check_words(word, target)))
        #print(str(game_state[-1][1]) + " " + game_state[-1][0])
        i += 1
    #if game_state[-1][1] == "🟩🟩🟩🟩🟩":
    #    print("Ended with a win in " + str(i) + " attempts")
    #else:
    #    print("Ended with a loss")
    return game_state



def batch0_0():
    for crossover in np.arange(0.5,1.1,0.1):
        for mutate in np.arange(0.01,0.06,0.01):
            for permutation in np.arange(0.01,0.11,0.01):
                for invert in np.arange(0.01,0.11,0.01):
                    for pop_size in range(50,250,50):
                        results = []
                        try_words = np.random.choice(a = daily_words, size = 100 , replace = False)
                        for word in try_words:
                            results.append(tryout(word, 0, frecs, frecs_new, crossover_prob = crossover,  
                                                    mutate_prob = mutate, permute_prob = permutation, 
                                                    invert_prob = invert, pop_size = pop_size))
                        aux = np.asarray(results)
                        try_words = np.asarray(try_words)
                        np.save(f"../hyperparam/m0c{int(crossover*10)}m{int(mutate*100)}p{int(permutation*100)}i{int(invert*100)}p{pop_size}.npy", aux)
                        np.save(f"../hyperparam/m0c{int(crossover*10)}m{int(mutate*100)}p{int(permutation*100)}i{int(invert*100)}p{pop_size}words.npy", try_words)

def batch0_1():
    for crossover in np.arange(0.5,1.1,0.1):
        for mutate in np.arange(0.06,0.11,0.01):
            for permutation in np.arange(0.01,0.11,0.01):
                for invert in np.arange(0.01,0.11,0.01):
                    for pop_size in range(50,250,50):
                        results = []
                        try_words = np.random.choice(a = daily_words, size = 100 , replace = False)
                        for word in try_words:
                            results.append(tryout(word, 0, frecs, frecs_new, crossover_prob = crossover,  
                                                    mutate_prob = mutate, permute_prob = permutation, 
                                                    invert_prob = invert, pop_size = pop_size))
                        aux = np.asarray(results)
                        try_words = np.asarray(try_words)
                        np.save(f"../hyperparam/m0c{int(crossover*10)}m{int(mutate*100)}p{int(permutation*100)}i{int(invert*100)}p{pop_size}.npy", aux)
                        np.save(f"../hyperparam/m0c{int(crossover*10)}m{int(mutate*100)}p{int(permutation*100)}i{int(invert*100)}p{pop_size}words.npy", try_words)

def batch1_0():
    for crossover in np.arange(0.5,1.1,0.1):
        for mutate in np.arange(0.01,0.06,0.01):
            for permutation in np.arange(0.01,0.11,0.01):
                for invert in np.arange(0.01,0.11,0.01):
                    for pop_size in range(50,250,50):
                        results = []
                        try_words = np.random.choice(a = daily_words, size = 100 , replace = False)
                        for word in try_words:
                            results.append(tryout(word, 1, frecs, frecs_new, crossover_prob = crossover,  
                                                    mutate_prob = mutate, permute_prob = permutation, 
                                                    invert_prob = invert, pop_size = pop_size))
                        aux = np.asarray(results)
                        try_words = np.asarray(try_words)
                        np.save(f"../hyperparam/m1c{int(crossover*10)}m{int(mutate*100)}p{int(permutation*100)}i{int(invert*100)}p{pop_size}.npy", aux)
                        np.save(f"../hyperparam/m1c{int(crossover*10)}m{int(mutate*100)}p{int(permutation*100)}i{int(invert*100)}p{pop_size}words.npy", try_words)

def batch2_0():
    for crossover in np.arange(0.5,1.1,0.1):
        for mutate in np.arange(0.01,0.06,0.01):
            for permutation in np.arange(0.01,0.11,0.01):
                for invert in np.arange(0.01,0.11,0.01):
                    for pop_size in range(50,250,50):
                        results = []
                        try_words = np.random.choice(a = daily_words, size = 100 , replace = False)
                        for word in try_words:
                            results.append(tryout(word, 2, frecs, frecs_new, crossover_prob = crossover,  
                                                    mutate_prob = mutate, permute_prob = permutation, 
                                                    invert_prob = invert, pop_size = pop_size))
                        aux = np.asarray(results)
                        try_words = np.asarray(try_words)
                        np.save(f"../hyperparam/m2c{int(crossover*10)}m{int(mutate*100)}p{int(permutation*100)}i{int(invert*100)}p{pop_size}.npy", aux)
                        np.save(f"../hyperparam/m2c{int(crossover*10)}m{int(mutate*100)}p{int(permutation*100)}i{int(invert*100)}p{pop_size}words.npy", try_words)


def batch1_1():
    for crossover in np.arange(0.5,1.1,0.1):
        for mutate in np.arange(0.06,0.11,0.01):
            for permutation in np.arange(0.01,0.11,0.01):
                for invert in np.arange(0.01,0.11,0.01):
                    for pop_size in range(50,250,50):
                        results = []
                        try_words = np.random.choice(a = daily_words, size = 100 , replace = False)
                        for word in try_words:
                            results.append(tryout(word, 1, frecs, frecs_new, crossover_prob = crossover,  
                                                    mutate_prob = mutate, permute_prob = permutation, 
                                                    invert_prob = invert, pop_size = pop_size))
                        aux = np.asarray(results)
                        try_words = np.asarray(try_words)
                        np.save(f"../hyperparam/m1c{int(crossover*10)}m{int(mutate*100)}p{int(permutation*100)}i{int(invert*100)}p{pop_size}.npy", aux)
                        np.save(f"../hyperparam/m1c{int(crossover*10)}m{int(mutate*100)}p{int(permutation*100)}i{int(invert*100)}p{pop_size}words.npy", try_words)

def batch2_1():
    for crossover in np.arange(0.5,1.1,0.1):
        for mutate in np.arange(0.06,0.11,0.01):
            for permutation in np.arange(0.01,0.11,0.01):
                for invert in np.arange(0.01,0.11,0.01):
                    for pop_size in range(50,250,50):
                        results = []
                        try_words = np.random.choice(a = daily_words, size = 100 , replace = False)
                        for word in try_words:
                            results.append(tryout(word, 2, frecs, frecs_new, crossover_prob = crossover,  
                                                    mutate_prob = mutate, permute_prob = permutation, 
                                                    invert_prob = invert, pop_size = pop_size))
                        aux = np.asarray(results)
                        try_words = np.asarray(try_words)
                        np.save(f"../hyperparam/m2c{int(crossover*10)}m{int(mutate*100)}p{int(permutation*100)}i{int(invert*100)}p{pop_size}.npy", aux)
                        np.save(f"../hyperparam/m2c{int(crossover*10)}m{int(mutate*100)}p{int(permutation*100)}i{int(invert*100)}p{pop_size}words.npy", try_words)


p0 = mult.Process(target = batch0_0)
p1 = mult.Process(target = batch0_1)
p2 = mult.Process(target = batch1_0)
p3 = mult.Process(target = batch1_1)
p4 = mult.Process(target = batch2_0)
p5 = mult.Process(target = batch2_1)


if __name__ == '__main__':
    p0.start()
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p0.join()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()