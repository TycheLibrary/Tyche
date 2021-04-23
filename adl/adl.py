'''
classes for representing adl formulas
'''

class Concept:

    self.symbol
    self.description

    def __init__(self, symbol, description="Generic Concept"):
        if(symbol):
            self.symbol = symbol
        else:
            #throw erro if syntax wrong
            return False

    def __repr__(self):
      return self.symbol +":"+ self.description

    def __str__(self):
        return self.symbol





