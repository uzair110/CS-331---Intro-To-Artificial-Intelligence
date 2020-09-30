
# Set up library imports.
import random
from collections import Counter
from itertools import chain
from bitstring import *

########################################################
'''
    - This module assumes you first read the Jupyter notebook. 
    - You are free to add other members functions in class GeneticAlgorithm
      as long as you do not modify the code already written. If you have justified
      reasons for making modifications in code, come talk to us. 
    - Our implementation uses recursive solutions and some flavor of 
      functional programming (maps/lambdas); You're not required to do so.
      Just Write clean code. 
'''
########################################################

class GeneticAlgorithm(object):

    def __init__(self, POPULATION_SIZE, CHROMOSOME_LENGTH, verbose):
        self.wall_bit_string_raw = "01010101011001101101010111011001100101010101100101010101"
        self.wall_bit_string = ConstBitStream(bin = self.wall_bit_string_raw)
        self.population_size = POPULATION_SIZE
        self.chromosome_length = CHROMOSOME_LENGTH # this is the length of self.wall_bit_string
        self.terminate = False
        self.verbose = verbose # In verbose mode, fitness of each individual is shown. 

    def run_genetic_alg(self):
        '''  
        The pseudo you saw in slides of Genetic Algorithm is implemented here. 
        Here, You'll get a flavor of functional 
        programming in Python- Those who attempted ungraded optional tasks in tutorial
        have seen something similar there as well. 
        Those with experience in functional programming (Haskell etc)
        should have no trouble understanding the code below. Otherwise, take our word that
        this is more or less similar to the generic pseudocode in Jupyter Notebook.

        '''
        "You may not make any changes to this function."

        # Creation of Population
        solutions = self.generate_candidate_sols(self.population_size) # arg passed for recursive implementation.

        # Evaluation of individuals
        parents = self.evaluate_candidates(solutions)

        while(not self.terminate):
            # Make pairs
            pairs_of_parents = self.select_parents(parents)

            # Recombination of pairs.
            recombinded_parents = list(chain(*map(lambda pair: \
                self.recombine_pairs_of_parents(pair[0], pair[1]), \
                    pairs_of_parents))) 

            # Mutation of each individual
            mutated_offspring = list(map(lambda offspring: \
                self.mutate_offspring(offspring), recombinded_parents))

            # Evaluation of individuals
            parents = self.evaluate_candidates(mutated_offspring) # new parents (offspring)
            if self.verbose and not self.terminate:
                self.print_fitness_of_each_indiviudal(parents)

######################################################################
###### These two functions print fitness of each individual ##########

# *** "Warning" ***: In this function, if an individual with 100% fitness is discovered, algorithm stops. 
# You should implement a stopping condition elsewhere. This codition, for example,
# won't stop your algorithm if mode is not verbose.
    def print_fitness_of_one_individual(self, _candidate_sol):
        _WallBitString = self.wall_bit_string
        _WallBitString.pos = 0
        _candidate_sol.pos = 0
        
        matching_bit_pairs = 0
        try:
            if not self.terminate:
                while (_WallBitString.read(2).bin == _candidate_sol.read(2).bin):
                    matching_bit_pairs = matching_bit_pairs + 1
                # return round((matching_bit_pairs)/28*100, 2), matching_bit_pairs
                print('Individual Fitness: ', round((matching_bit_pairs)/28*100, 2), '%')
        except: # When all bits matched. 
            pass
            return 

    def print_fitness_of_each_indiviudal(self, parents):
        if parents:
            for _parent in parents:
                self.print_fitness_of_one_individual(_parent)

###### These two functions print fitness of each individual ##########
######################################################################

    def select_parents(self, parents):
        '''
        args: parents (list) => list of bitstrings (ConstbitStream)
        returns: pairs of parents (tuple) => consecutive pairs.
        '''

        # **** Start of Your Code **** #
        pass
        # **** End of Your Code **** #


    # A helper function that you may find useful for `generate_candidate_sols()`
    def random_num(self):
        random.seed()
        return random.randrange(2**14) ## for fitting in 14 bits.

    def generate_candidate_sols(self, n): 
        '''
        args: n (int) => Number of cadidates solutions to generate. 
        retruns: (list of n random 56 bit ConstBitStreams) 
                 In other words, a list of individuals: Population.

        Each cadidates solution is a 56 bit string (ConstBitStreams object). 

        One clean way is to first get four 14 bit random strings then concatenate
        them to get the desired 56 bit candidate. Repeat this for n candidates.
        '''

        pops=[]
        randomstring=''
        for j in range(n):
            randstring1 = str(format(self.random_num(), "014b"))
            randstring2 = str(format(self.random_num(), "014b"))
            randstring3 = str(format(self.random_num(), "014b"))
            randstring4 = str(format(self.random_num(), "014b"))
            # print(randstring1)
            # for i in range (0, len(randstring1)):
            #     if randstring1[i]==" ":
            #         randstring1[i]=0
            # for i in range (0, len(randstring2)):
            #     if randstring2[i]==" ":
            #         randstring2[i]=0

            # for i in range (0, len(randstring3)):
            #     if randstring3[i]==" ":
            #         randstring3[i]=0

            # for i in range (0, len(randstring4)):
            #     if randstring3[i]==" ":
            #         randstring3[i]=0

            randomstring = randstring1 + randstring2 + randstring3 + randstring4
            randomstring = ConstBitStream(bin = randomstring)
            pops.append(randomstring)

        return pops

                


    def recombine_pairs_of_parents(self, p1, p2):
        """
        args: p1, and p2  (ConstBitStream)
        returns: p1, and p2 (ConstBitStream)

        split at .6-.9 of 56 bits (CHROMOSOME_LENGTH). i.e. between 31-50 bits
        """
        cutoff = random.randrange(31, 52)
        print(cutoff)

        head_p1 = p1[cutoff:]
        tail_p1 = p1[:cutoff-len(p1)]

        head_p2 = p2[cutoff:]
        tail_p2 = p2[:cutoff-len(p2)]

        p1 = head_p1+tail_p2
        p2 = head_p2+tail_p1

        return p1, p2
        
        

    def mutate_offspring(self, p):
        ''' 
            args: individual (ConstBitStream)
            returns: individual (ConstBitStream)
        '''

        # **** Start of Your Code **** #
        pass
        # **** End of Your Code **** #

    def fitness_of_one_individual(self, _candidate_sol):
            _WallBitString = self.wall_bit_string
            _WallBitString.pos = 0
            _candidate_sol.pos = 0
            
            matching_bit_pairs = 0
            try:
                if not self.terminate:
                    while (_WallBitString.read(2).bin == _candidate_sol.read(2).bin):
                        matching_bit_pairs = matching_bit_pairs + 1
                    return round((matching_bit_pairs)/28*100, 2), matching_bit_pairs
            except: # When all bits matched. 
                pass
                return 

    def evaluate_candidates(self, candidates): 
        '''
        args: candidate solutions (list) => each element is a bitstring (ConstBitStream)
        
        returns: parents (list of ConstBitStream) => each element is a bitstring (ConstBitStream) 
                    but elements are not unique. Fittest candidates will have multiple copies.
                    Size of 'parents' must be equal to population size.  
        '''
        sum_of_pairs = 0
                
        fitness_individual = []
        for i in range(len(candidates)):
            fitness_ind, matching_pairs = self.fitness_of_one_individual(candidates[i])
            sum_of_pairs = sum_of_pairs + matching_pairs
            fitness_individual.append((candidates[i],fitness_ind))

        f_avg = (sum_of_pairs / len(candidates)*28)/len(candidates)
        # print("f_avg:", f_avg)
        to_send_ind = []
        if f_avg > 0:
            relative_fitness = []
            for i,j in fitness_individual:
                r_fitness = j/f_avg
                relative_fitness.append((i,round(r_fitness)))

            prob = []
            for _,fitness_ind in relative_fitness:
                p = fitness_ind/(sum_of_pairs/28*len(candidates))
                prob.append(p)
 
            while len(to_send_ind) < len(candidates):
                for i in range(len(candidates)):
                    num = random.randrange(2)
                    if num > prob[i]:
                        to_send_ind.append(candidates[i])
        else:
            #randomly adding values 
            while len(to_send_ind) < len(candidates):
                rand = random.randrange(0,1)
                if i in range(len(candidates)):
                    if rand == 1:
                        to_send_ind.append(candidates[i])

        return to_send_ind

                
# def main():
    
#     xyz = GeneticAlgorithm(64, 56, False)
#     cands = xyz.generate_candidate_sols(64)
#     # print(cands)
#     fit = xyz.evaluate_candidates(cands)
#     print(fit)

# main()










