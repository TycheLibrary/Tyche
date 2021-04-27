'''
The base ADL language module, with classes for 
representing and reasoning with ADL formulas.

ADL is designed to support both mathematical notion and a formal english notion.
'''

class ADLException(Exception):
    '''
    Class for detailing ADL exceptions
    '''

    def __init__(self, message):
        self.message= "ADLException:" + message


class Concept:
    '''
    An abstract base class for representing all ADL concepts.

    This class should not be instantiated and only exists to be subclassed.
    It describes the methods that all ADL concepts shold have avilable.
    '''
    
    def __init__(self):
        '''Should not execute'''

        raise ADLException("Concept is an abstract class and should not be instatiated")

    def __str__(self):
        '''Returns a representation of the formula as it would appear in a formula'''
        pass

    def __repr__(self):
        '''Returns the representation of the formula, along an optional descriptive string'''
        pass

    def normal_form(self) -> 'Concept':
        '''Returns the tree normal form of a Concept, where atoms are ordered alphabetically.'''
        pass

    def is_equal(self, adl_form: 'Concept') -> bool:
        '''Returns true if this Concept is provably equivalent to adl_form'''
        pass

    def is_weaker(self, adl_form: 'Concept') -> bool:
        '''Returns true if the probability of this Concept is provably 
        necesssarily less than or equal to the probabiliy of adl_form'''
        pass
 
    def is_stronger(self, adl_form: 'Concept') -> bool:
        '''Returns true if the probability of this concept is provably
        necessarily greater than or equal to the probability of adl_form'''
        pass
 
    '''
    Other methods to include later, are methods to generate a proof string, and aleatroic polynomials
    '''


class Atom(Concept):

    def __init__(self, symbol: str, description: str):
        '''Represents an Atomic Concept

           User defined Atoms should have symbol names that begin with a lower case letter 
           and then consist of upper or lower case latters, digits, or underscore.
        '''
        if(symbol==None or len(symbol)<1):
            raise ADLException("Naming error, atomic concept symbols must begin with lower case letters")
        self.symbol = symbol
        self.description = description


    def __repr__(self):
      return self +":"+ self.description

    def __str__(self):
        return self.symbol

    def normal_form(self) -> Concept:
        '''Returns the tree normal form of a Concept, where atoms are ordered alphabetically.'''
        return self

    def is_equal(self, adl_form: Concept) -> bool:
        '''Returns true if this Concept is provably equivalent to adl_form'''
        return self.symbol == adl_form.symbol

    def is_weaker(self, adl_form: Concept) -> bool:
        '''Returns true if the probailty of this Concept is provably 
        necessarily less than or equal to the probability of adl_form'''
        return not type(adl_form)==type(self) and adl_form.is_stronger(Yes)
 
    def is_greater_than(self, adl_form: Concept) -> bool:
        '''Returns true if this Concept is provably greater than adl_form'''
        return not type(adl_form)==type(self) and adl_form.is_less_than(No)
    #need to sort out a base base here....
 
Yes = Atom('Y','Yes, a concept with probability always 1.0')
No = Atom('N', 'No, a concept with probability always 0.0')


