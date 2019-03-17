
# coding: utf-8

# In[62]:


class text_game:
    from subprocess import Popen, PIPE, STDOUT
    from random import randint
    import binascii
    import os
    from queue import Queue, Empty
    from threading  import Thread
    from time import sleep
    import random
    import nltk
    nltk.download('averaged_perceptron_tagger')
    from itertools import permutations, combinations
    import spacy
    from spacy.matcher import Matcher
    from spacy.attrs import POS
    nlp = spacy.load('en_core_web_sm')
    import pandas as pd
    nltk.download('punkt')
    import numpy as np
    import gensim
    from gensim.models import KeyedVectors
    from gensim.models import Word2Vec
    from nltk.tokenize import sent_tokenize, word_tokenize
    import timeit
    from keras.models import Model
    from keras.preprocessing.text import one_hot, Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    import re
    from agent import DQNAgent as DQNAgent
    import importlib
    from collections import deque
    from numpy.random import choice
    import math
    import pickle
    from progressbar import ProgressBar
    from agent import DQNAgent
    from game_commands import commands
 
    def __init__(self):
        
        import agent
        from agent import DQNAgent
        self.emulator_file = 'dfrotz.exe'
        
        self.agent = DQNAgent()
        
        self.p = None
        self.q = None
        self.t = None
        
        self.word_2_vec = None
        self.tutorials_text = 'tutorials_2.txt'
        
        self.tokenizer = None
        self.vocab_size = 800
        
        self.sleep_time = 0.1
        self.random_action_weight = 4
        self.random_action_basic_prob = 0.5
        self.random_action_low_prob = 0.2
        self.game_score_weight = 5
        self.negative_per_turn_reward = 1
        self.inventory_reward_value = 20
        self.new_area_reward_value = 10
        
        import game_commands
        from game_commands import commands
        
        cmds = commands()
        self.basic_actions = cmds.basic_actions
        self.directions = cmds.directions
        self.command1_actions = cmds.command1_actions
        self.command2_actions = cmds.command2_actions
        self.action_space = cmds.action_space
        self.filtered_tokens = cmds.filtered_tokens
        self.invalid_nouns = [] 
        
        self.story = pd.DataFrame(columns=['Surroundings', 'Inventory', 'Action', 'Response', 'Score', 'Moves'])
        
    def __create_agent(self):
        dqna = DQNAgent()
        return dqna
    
    def enqueue_output(self, out, queue):
        for line in iter(out.readline, b''):
            queue.put(line)
        out.close()

    def start_game(self, game_file):
        load_invalid_nouns()
        init_word2vec()
        init_tokenizer()
        
        score = 0
        moves = 0
        p = Popen([emulator_file,game_file], stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
        q = Queue()
        t = Thread(target=enqueue_output, args=(p.stdout, q))
        t.daemon = True # thread dies with the program
        t.start()
        sleep(sleep_time)
        return(p, q, t, score, moves)
    
    def end_game(self):
        save_invalid_nouns()
        kill_game()
        
    def kill_game(self, p):
        p.terminate()
        p.kill()

    # read line without blocking
    def readLine(self, q):
        cont = True
        narrative = ""
        while cont:
            try:  line = q.get_nowait() # or q.get(timeout=.1)
            except Empty:
                cont = False
            else: 
                narrative = narrative + line.decode("utf-8").replace('\n', " ").replace('\r', " ")
        if ('840726' in narrative): ## Opening narrative
            narrative = narrative[narrative.index('840726') + len('840726'):]
        try:
            score, moves = grab_score_moves(narrative)
            narrative = narrative[narrative.index('Moves: ')+len('Moves:')+5:-1].strip()
        except:  ## not valid move
            pass
        sleep(sleep_time)
        return(narrative, score, moves)

    def grab_score_moves(self, narrative):
        try:
            score = int(narrative[narrative.index('Score: ') + len('Score: '):][0:3].strip())
            moves = int(narrative[narrative.index('Moves: ') + len('Moves: '):][0:3].strip())
        except:  ## not valid move
            score = 0
            moves = 0
        return(score, moves)

    def look_surroundings(self, p):
        perform_action('look', p)

    def check_inventory(self, p):
        perform_action('inventory', p)

    def get_nouns(self, narrative):
        matcher = Matcher(nlp.vocab)
        matcher.add('Noun phrase', None, [{POS: 'NOUN'}])
        doc = nlp(narrative)
        matches = matcher(doc)
        noun_list = [doc[start:end].text for id, start, end in matches]
        for direction in directions:
            if direction in noun_list:
                noun_list.remove(direction)
        for invalid in invalid_nouns:
            if invalid in noun_list:
                noun_list.remove(invalid)
        return(noun_list)
    
    def generate_action_tuples(self, nouns):
        possible_actions = []
        similarities = []
        for i in basic_actions:
            possible_actions.append(i)
            similarities.append(random_action_basic_prob)
        for i in nouns:
            for action1 in command1_actions:   ## first loop replaces 'x' in each action in command1_actions
                action_to_add = action1.replace('OBJ', i)
                possible_actions.append(action_to_add)
                try:
                    similarities.append(model.similarity(word_tokenize(action_to_add)[0], i))
                except:
                    similarities.append(random_action_low_prob)
            noun_permutations = list(permutations(nouns, 2))    ## second loop replaces 'x' and 'y' in each action in command2_actions
            for action2 in command2_actions:
                for perm in noun_permutations:
                    if (perm[0] == perm[1]):  ## ignore same noun acting on itself
                        pass
                    else:
                        action_to_add = action2.replace('OBJ', perm[0])
                        action_to_add = action_to_add.replace('DCT', perm[1])
                        possible_actions.append(action_to_add)
                        try:
                            similarities.append(model.similarity(word_tokenize(action_to_add)[0], i))
                        except:
                            similarities.append(random_action_low_prob)

        return possible_actions
    
    def selectOne(self, action_space, similarities):
        return action_space[choice(len(action_space), p=similarities)]
    
    def add_to_action_space(self, action_space, actions):
        ## 
        
        similarities = []

        for action in actions:
            action_space.add(action)
        for action in action_space:
            words = word_tokenize(action)
            verb = words[0]
            if verb in basic_actions:    ## basic commands i.e. go north, go south
                similarities.append(random_action_basic_prob)
            elif len(words)<3:           ## commands with one noun i.e. open mailbox, read letter
                noun = word_tokenize(action)[1]
                try:
                    sim_score = model.similarity(verb, noun)**random_action_weight
                    if sim_score < 0:
                        sim_score = random_action_basic_prob**random_action_weight
                    similarities.append(sim_score)
                except:
                    similarities.append(random_action_low_prob**random_action_weight)

            else:                       ## commands with two nouns i.e. unlock chest with key
                try:
                    noun1 = word_tokenize(action)[1]
                    prep = word_tokenize(action)[2]
                    noun2 = word_tokenize(action)[3]
                    sim_score1 = model.similarity(verb, noun1)
                    sim_score2 = model.similarity(prep, noun2)
                    sim_score = ((sim_score1 + sim_score2)/2)**random_action_weight
                    if sim_score < 0:
                        sim_score = 0.05
                    similarities.append(sim_score**random_action_weight)
                except:
                    similarities.append(random_action_low_prob**random_action_weight)


        return action_space, similarities
        
    def perform_action(self, command, p):
        p.stdin.write(bytes(command+ "\n", 'ascii'))
        p.stdin.flush()
        sleep(sleep_time)## wait for action to register
        
    def preprocess(self, text):
        # fix bad newlines (replace with spaces), unify quotes
        text = text.strip()
        text = text.replace('\\n', ' ').replace('‘', '\'').replace('’', '\'').replace('”', '"').replace('“', '"')
        # convert to lowercase
        text = text.lower()
        # remove all characters except alphanum, spaces and - ' "
        text = re.sub('[^ \-\sA-Za-z0-9"\']+', ' ', text)
        # split numbers into digits to avoid infinite vocabulary size if random numbers are present:
        text = re.sub('[0-9]', ' \g<0> ', text)
        # expand unambiguous 'm, 't, 're, ... expressions
        text = text.                 replace('\'m ', ' am ').                 replace('\'re ', ' are ').                 replace('won\'t', 'will not').                 replace('n\'t', ' not').                 replace('\'ll ', ' will ').                 replace('\'ve ', ' have ').                 replace('\'s', ' \'s')
        return text
    
    def vectorize_text(text, tokenizer):
        text = preprocess(text)
        words = word_tokenize(text)
        tokenizer.fit_on_texts(words)
        seq = tokenizer.texts_to_sequences(words)
        sent = []
        for i in seq:
            sent.append(i[0])
        padded = pad_sequences([sent], maxlen=50, padding='post')
        return (padded)
    
    def calculate_reward(self, story, moves_count, new_narrative):
        reward = 0

        ## add reward from score in game
        if(moves_count != 0):
            new_score = int(story['Score'][moves_count]) - int(story['Score'][moves_count-1])
            reward = reward + new_score*game_score_weight

        ## add small negative reward for each move
        reward = reward - negative_per_turn_reward

        ## add reward for picking up / using items
        if(moves_count != 0):
            pre_inventory = story['Inventory'][moves_count-1]
            inventory = story['Inventory'][moves_count]
            if pre_inventory != inventory:  ## inventory changed
                reward = reward + inventory_reward_value
                print('inventory changed')


        ## add reward for discovering new areas
        if new_narrative not in unique_narratives:  ## new location
            reward = reward + new_area_reward_value
            print('discovered new area')
        print(reward)
        return reward

    def detect_invalid_nouns(self, action_response):
        ## detect and remove invalid nouns from future turns
        action_response = preprocess(str(action_response))
        if('don\'t know the word' in response):
            startIndex = action_response.find('\"')
            endIndex = action_response.find('\"', startIndex + 1)
            word = action_response[startIndex+1:endIndex]
            print('Didn\'t know the word: ' + word)
            invalid_nouns.append(word)
    
    def save_invalid_nouns(self):
        ## save invalid nouns to pickled list
        try:
            with open('invalid_nouns.txt', 'wb') as fp:
                pickle.dump(invalid_nouns, fp)
        except:
            pass
    
    def load_invalid_nouns(self):
        ## load previously found invalid nouns from pickled list
        try:
            with open ('invalid_nouns.txt', 'rb') as fp:
                n = pickle.load(fp)
                invalid_nouns.extend(n)
        except:
            pass
    
    def init_word2vec(self):
        #model = Word2Vec.load('tutorial.model')
        f = open(tutorials_text, 'r')
        tutorials = f.read()
        sentences = word_tokenize(tutorials)
        model = Word2Vec([sentences])
        return model
    
    def init_tokenizer(self):
        tokenizer = Tokenizer(num_words=vocab_size)
        
    def run_game(self, agent, num_rounds):
        pbar = ProgressBar(maxval=num_rounds)
        pbar.start()
        try:
            for i in range(0,num_rounds):
                ## Check surroundings, check inventory, choose action, check action response
                narrative,score,moves = readLine(q)
                check_inventory(p)
                inventory,s,m = readLine(q)
                
                ## grab nouns in current env
                nouns = get_nouns(narrative)

                # build action space
                current_action_space = generate_action_tuples(nouns)
                action_space = set()
                action_space, probs = add_to_action_space(action_space, current_action_space)
                actions = []
                for a in action_space:
                    actions.append(a)

                narrativeVector = vectorizeText(narrative,tokenizer)

                ## decide which type of action to perform
                if (agent.act_random() or i < 10): ## choose random action
                    print('random choice')
                    probs = np.array(probs)
                    probs /= probs.sum()
                    action = selectOne(actions, probs)
                    df = pd.DataFrame(columns=['Action', 'Prob'])
                    df['Action'] = actions
                    df['Prob'] = probs

                else: ## choose predicted max Q value action
                    print('predicted choice')
                    actionsVectors = []
                    for a in actions:
                        actionsVectors.append(vectorizeText(a,tokenizer))
                    best_action, max_q = agent.predict_actions(narrativeVector, actionsVectors)
                    action = actions[best_action]

                ## perform selected action
                perform_action(action, p)

                ## grab response from action
                response,score,moves = readLine(q)
                story.loc[i] = [narrative, inventory, action, response, str(s), str(i+1)]
                unique_narratives.add(preprocess(narrative))

                actionVector = vectorizeText(action,tokenizer)

                look_surroundings(p)
                new_narrative,s,m = readLine(q)
                new_narrativeVector = vectorizeText(new_narrative, tokenizer)

                
                # get reward
                reward = calculate_reward(story, i, preprocess(new_narrative))

                agent.remember(narrativeVector, actionVector, reward, new_narrativeVector, False)

                look_surroundings(p)

                if moves_count%batch_size == 0:  ## batch of 5 experiences in memory
                    agent.replay(batch_size)

                pbar.update(i) ## update progress bar
            pbar.finish()
            kill_game(p)
        except Exception as e: 
            kill_game(p)
            print(e.with_traceback())

