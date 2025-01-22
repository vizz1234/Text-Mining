import re


class MyTokenizer:
    
    def __init__(self):
        pass


    def tokenize(self, text):

        # Split by whitespace
        tokens = text.split(' ')

        idx = 0
        while True:
            # Stop if we reach the end of the token list
            if idx >= len(tokens):
                break

            # Grab the next token
            token = tokens[idx]

            # Split the token into 2 or more subtokens; might not be necessary!
            components = self.split_token(token)

            # Just for safety, we ignore any empty component
            # (this might happen during the split and it's more convenient to handle it here)
            components = [ c.strip() for c in components if len(c.strip()) > 0]

            # Check if the current token was indeed split into 2 more components
            if len(components) > 1:
                # If the token was split create a new token list
                tokens = tokens[:idx] + components + tokens[idx+1:]
            else:
                # If the token was NOT split; just go to the next token in the list
                idx +=1

        # Return final list of tokens
        return tokens

    
    def split_token(self, token):

        # Contains all the subtokens that might result from any splitting
        components = []

        # Example criteria: split tokens containing a repeated sequence (e.g., "bonbon", "papa")
        # NOTE: This is only to give a example how to approach this tasks
        for match in re.finditer(r"^([a-zA-Z]+)\1$", token):
            # 1st component: all characters from the beginning of the token  until the end of the first group
            components.append(token[:match.span(1)[1]])
            # 2nd component: all characters from the end of the first group until the end of the token
            components.append(token[match.span(1)[1]:])
            return components            

        #########################################################################################
        ### Your code starts here ###############################################################    
        if token.startswith('\"') or token.startswith('('):
            return [token[0], token[1:]]
        if '-' in token:
            tokenSplit = token.split('-')
            finalToken = []
            for i in range(len(tokenSplit)-1):
                finalToken.extend([tokenSplit[i], '-'])
            finalToken.append(tokenSplit[-1])
            return finalToken
        if '/' in token:
            tokenSplit = token.split('/')
            finalToken = []
            for i in range(len(tokenSplit)-1):
                finalToken.extend([tokenSplit[i], '/'])
            finalToken.append(tokenSplit[-1])
            return finalToken
        if '...' in token:
            tokenSplit = token.split('...')
            finalToken = []
            for i in range(len(tokenSplit)-1):
                finalToken.extend([tokenSplit[i], '...'])
            finalToken.append(tokenSplit[-1])
            return finalToken   
        if '\'s' in token or '\'d' in token:
            return [token[:-2], token[-2:]]
        if '\'re' in token or '\'ve' in token or 'n\'t' in token:
            return [token[:-3], token[-3:]]
        if token.endswith('\"') or token.endswith(')') or token.endswith('!'):
            return [token[:-1], token[-1]]
        if '.' in token:
            tokenSplit = token.split('.')
            if len(tokenSplit) == 2:
                return [token[:-1], '.']
            else:
                return [token]
        
        ### Your code ends here #################################################################
        #########################################################################################    

        # If we didn't find any reason to split, return the token as the only subtoken itself
        return [token]
