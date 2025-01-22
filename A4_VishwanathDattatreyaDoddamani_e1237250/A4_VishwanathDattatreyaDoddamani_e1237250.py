import numpy as np
import requests, json

import spacy
nlp = spacy.load("en_core_web_sm", disable=['parse', 'ner'])


class MyRAKE:
    
    def __init__(self):
        pass
    
    
    def extract_candidates(self, doc):
        candidates, vocabulary = [], set([])

        #########################################################################################
        ### Your code starts here ###############################################################   
        candidateText = []
        for token in doc:
            word = token.text.lower()
            if  not token.is_stop and not token.is_punct and token.text not in ['(', ')']:
                candidateText.append(word)
                vocabulary.add(word)
            else:
                if candidateText != []:
                    candidates.append(candidateText)
                    candidateText = []

        ### Your code ends here #################################################################
        #########################################################################################        

        return candidates, vocabulary
    
    
    
    def calc_term_scores(self, candidates, vocabulary):
        
        term_scores = {}
        
        #########################################################################################
        ### Your code starts here ############################################################### 
        for word in vocabulary:
            freq = 0
            deg = 0
            for candidate in candidates:
                freq += candidate.count(word)
                if word in candidate:
                    deg += len(candidate)
            term_scores[word] = deg / freq
        
        ### Your code ends here #################################################################
        #########################################################################################                            

        return term_scores
                
                

    def get_top_keywords(self, candidates, term_scores, top=5):
        
        top_keywords_scored = []
    
        #########################################################################################
        ### Your code starts here ############################################################### 
        
        candidateScores = []
        candidatesText = []
        for candidate in candidates:
            candidateScore = 0
            for word in candidate:
                candidateScore += term_scores[word]
            candidateScores.append(candidateScore)
            candidatesText.append(' '.join(candidate))   
        candidateScoresZip = list(zip(candidatesText, candidateScores))
        candidateScoresSorted = sorted(candidateScoresZip, key = lambda x : x[1], reverse = True)
        top_keywords_scored = candidateScoresSorted[:top]
        
        ### Your code ends here #################################################################
        #########################################################################################           
        
        return top_keywords_scored
    
    
    
    
class MyTextRank:
    
    
    def __init__(self, alpha=0.85, eps=0.0001, max_iter=1000):
        self.alpha = alpha
        self.eps = eps
        self.max_iter = max_iter
        self.num_iter_ = None
    
    
    def convert_text_to_graph(self, sentences, preprocess_only=False):
        
        N = len(sentences)
        
        preprocessed = []
        
        #########################################################################################
        ### Your code starts here ###############################################################         
        for sentence in sentences:
            doc = nlp(sentence)
            preprocessed.append(' '.join(token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct))

        
        ### Your code ends here #################################################################
        #########################################################################################          
        
        if preprocess_only:
            return preprocessed
    
        # Initialize adajcency matrix A; this is what we return to represent the graph
        A = np.zeros((N, N))
    
        #########################################################################################
        ### Your code starts here ###############################################################         
        for i in range(N):
            si = set(preprocessed[i].split())
            for j in range(N):
                if i == j:
                    A[i, j] = 0.0
                    break
                sj = set(preprocessed[j].split())
                A[i , j] = len(si.intersection(sj)) / (np.log(len(si)) + np.log(len(sj)))
                A[j, i] = A[i, j]
        ### Your code ends here #################################################################
        #########################################################################################  
    
        return A
    
    
    
    def create_transition_matrix(self, A):
    
        M = None

        #########################################################################################
        ### Your code starts here ###############################################################    
        rowSums = np.sum(A, axis = 1)
        normalizeArray = A / rowSums[ :, np.newaxis]
        M = np.transpose(normalizeArray)
        ### Your code ends here #################################################################
        #########################################################################################

        return np.asarray(M)

    
    
    def power_method(self, M):

        E, c = None, None

        #########################################################################################
        ### Your code starts here ############################################################### 

        ## Initialize E and v
        n = M.shape[0]
        v = np.ones(n) / n
        E = np.transpose(v)

        ### Your code ends here #################################################################
        ######################################################################################### 

        # Run the power method: iterate until differences between steps converges
        self.num_iter_ = 0
        
        while True:

            self.num_iter_ += 1

            #########################################################################################
            ### Your code starts here ###############################################################  
            vNew = self.alpha * M.dot(v) + (1 - self.alpha) * E
            delta = np.linalg.norm(vNew - v, ord=1)
            v = vNew
            if delta < self.eps or self.num_iter_ >= self.max_iter:
                break

            ### Your code ends here #################################################################
            #########################################################################################            

            pass

        ## Return the results as a dictiory with the nodes as keys and the PageRank score as values
        return { k:score for k, score in enumerate(v) }    
    
    
    
    def run(self, sentences):
        # Create graph as adjacency matrix
        A = self.convert_text_to_graph(sentences)
        # Create transition matrix
        M = self.create_transition_matrix(A)
        # Compute and return PageRank scores
        return self.power_method(M)
    
    
    
    
    
    
    
    
    
class MyRelationExtractor():
    
    def __init__(self):
        pass
    
    
    
    def extract_hyponyms(self, sentence):
        # Let spaCy do its magic
        doc = nlp(sentence)
        # Generate and record all hyponyms
        hyponyms = []
        hyponyms.extend(self.extract_such_as_hyponyms(doc))      # Here we look only at "such as" constructs
        #hyponyms.extend(self.extract_or_other_hyponyms(doc))    # and only show here how the class could
        #hyponyms.extend(self.extract_including_hyponyms(doc))   # easily be extended to accomodate other
        #hyponyms.extend(self.extract_especially_hyponyms(doc))  # constructs as well (ignoring performance)
        # Return all found hyponyms
        return hyponyms
        
        
        
    def extract_such_as_hyponyms(self, doc):
        hyponyms = []
        
        ################################################################################################
        ##
        ## IMPORTANT: There are many and probably better ways to implement this method. 
        ## You are free to completely ignore the the skeleton code below!
        ##
        ################################################################################################
        
        valid_as_tokens = [] 
        
        #########################################################################################
        ### Your code starts here ###############################################################          
        
        # We first need to identify the valid tokens
        
        valid_as_tokens = [token for token in doc if not token.is_punct]
        ### Your code ends here #################################################################
        #########################################################################################     

        for i, token in enumerate(valid_as_tokens):
            try:

                #########################################################################################
                ### Your code starts here ###############################################################  

                # We are looking for patterns like this (quite easy with the dependency tree)
                # Y such as X ((, X)* (, and|or) X)
                if token.text.lower() == 'such' and valid_as_tokens[i+1].text.lower() == 'as':
                    hypernym = self.get_compound(valid_as_tokens[i-1])
                    hypernym = [word.lemma_ for word in hypernym]
                    hypernym = ' '.join(hypernym)
                    children = list(valid_as_tokens[i+1].children)
                    children = [child for child in children if child.text.lower() != 'such']
                    conjuncts = [s.conjuncts for s in children]
                    conjuncts.append(children)
                    uniqueConjuncts = []
                    for j in conjuncts:
                        for k in j:
                            if k not in uniqueConjuncts:
                                uniqueConjuncts.append(k)
                    for h in uniqueConjuncts:
                        hypo = self.get_compound(h)
                        hypo = [word.lemma_ for word in hypo]
                        hypo = ' '.join(hypo)
                        hyponyms.append((hypo.lower(), hypernym.lower()))
                    break

                ### Your code ends here #################################################################
                #########################################################################################     
                
                pass
        
            except Exception as e:
                pass
    
        return hyponyms
    
    
    
    def get_compound(self, token, to_string=True):
        # Collect all the parts the form the compound word
        compound_parts = []

        # Loop over all children of the token
        for child in token.children:
            # We are only interested in the "compound" relationship
            if child.dep_ == "compound":
                # If we found a relevant child, we add it to our list
                compound_parts.append(child)

        # At the end, we also need to add the head word itself
        compound_parts.append(token)
        
        return compound_parts


    def get_wikipedia_urls(self, search, topk=1):
        try:
            urls = []
            data = requests.get(f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={search}&language=en&format=json").json()
            qids = [ c['id'] for c in data['search'][:topk] ]
            for qid in qids:
                try:
                    wikipedia = requests.get(f"https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&props=sitelinks/urls&ids={qid}&sitefilter=enwiki").json()
                    urls.append(wikipedia["entities"][qid]["sitelinks"]["enwiki"]["url"])
                except:
                    pass
            return urls
        except Exception as e:
            print(e)
            return []