import numpy as np
import spacy
import model

nlp = spacy.load('en')
RANK_MAP = {
    'first': 0,
    'second': 1,
    'third': 2,
    'fourth': 3,
    'fifth': 4,
    'sixth': 5,
    'seventh': 6,
    'eighth': 7,
    'ninth': 8,
    'tenth': 9
}
MAP_RANK = {v: k for k, v in RANK_MAP.items()}

class Conversation:

    def __init__(self):
        '''
        The init function: Here, you should load any
        PyTorch models / word vectors / etc. that you
        have previously trained.
        '''
        self.notes = []
        self.start = False
        
    """
    Intent Handlers.
    """
    def start_note(self):
        return "OK, what would you like me to remember?"

    def take_note(self, note):
        self.notes.append(note)
        return "Alright, I've noted that."

    def retrieve(self, n=-1):
        return "Your {0} note was: {1}".format(MAP_RANK[n] if n > -1 else 'last', self.notes[n])

    def delete(self, n=-1):
        self.notes.pop(n)
        return "OK, I've deleted it."

    def total_num_notes(self):
        return "You have {0} notes.".format(len(self.notes))
    
    def respond(self, sentence):
        '''
        This is the only method you are required to support
        for the Conversation class. This method should accept
        some input from the user, and return the output
        from the chatbot.
        '''
        if self.start == True:
            self.start = False
            return self.take_note(sentence)
        else:
            INTENTS = ['start', 'retrieve', 'delete', 'total']
            intent = INTENTS[np.argmax(model.predict(sentence))]
            entities = nlp(sentence).ents
            if intent == 'start':
                self.start = True
                return self.start_note()
            elif intent == 'retrieve':
                try:
                    return self.retrieve(RANK_MAP[str(entities[0])])
                except:
                    return self.retrieve()
            elif intent == 'delete':
                try:
                    return self.delete(RANK_MAP[str(entities[0])])
                except:
                    return self.delete()
            else:
                return self.total_num_notes()

