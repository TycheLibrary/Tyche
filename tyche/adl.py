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


class Concept(tuple):
    '''
    An class for representing all ADL concepts.

    Concepts are immutable and encoded via tuples, where the first term of each tuple indicates the type of formula.
    The constructor builds atomic concepts, and complex concepts are constructed using the static methods provided.
    '''

    #seems like a waste of space to have strings in the tuples
    #use constants???
    #everything can b
    op2code = {'Yes':0,'No':1,'Atom':2,'Conditional':3,'Marginal':4}
    code2op = {0:'Yes',1:'No',2:'Atom',3:'Conditional',4:'Marginal'}

    def __new__(cls, operator :str, symbol=None: str, term1=None:Concept, term2=None:Concept, term3=None:Concept, role=None:Role):
        '''Constructs all formulas as tuples'''
        constructors = {#lambdas to construct formulas
            'Yes': lambda c, s, x, y, z, r: tuple.__new__(c, (0))
            'No': lambda c, s, x, y, z, r: tuple.__new__(c, (1))
            'Atom': lambda c, s, x, y, z, r : tuple.__new__(c, (2, x)),
            'Conditional': lambda c, s, x, y, z, r: tuple.__new__(c, (3,x,y,z)),
            'Marginal': lambda c, s, x, y, z, r: tuple.__new__(c, (4, r,x,y))
            }#language core
        if symbol is not None not is_valid_symbol:
            raise ADLExcpetion('Symbols must begin with lowercase letter, and contain only alphanumeric characters or underscores')
        if role is not None not is_valid_symbol:
            raise ADLExcpetion('Roles must begin with lowercase letter, and contain only alphanumeric characters or underscores')
        if not operator in ['Yes','No','Atom','Conditinal','Marginal']:
            raise ADLExcpetion('Operator must be Yes, No, Atom, Conditional or Marginal')
        return constructors.get(operator)(symbol, term1, term2, term3, role)

    def is_valid_symbol(s:str):
        '''Checks a string contains only alphanumeric chacaters or underscore'''
        valid = s not None and len(s)<1 or not s[0].islower()
        for c in s[1:]:
            valid = valid and ((c.isalnum) or c=='_')
        return valid    

    #static constructors for different types of formulas
    yes = Concept('Yes')
    no = Concept('No')
    def atom(symbol:str) -> Concept:
        return Concept('Atom',symbol=symbol)
    def conditional(condition: Concept, positive: Concept, negative: Concept)-> Concept:
        return Concept('Conditional', term1=condition, term2=positive, term3=negative)
    def marginal(role: Role, event: Concept, margin: Concept)-> Concept:
        return Concept('Marginal', term1=condition, term2=margin, role=role)
    #some abbreviations provided too
    def negation(term:Concept) -> Concept:
        return Concept('Conditional', term1=term, term2=no, term3=yes)
    def conjunction(left:Concept, right: Concept) -> Concept:
        return Concept('Conditional', term1=left, term2=right, term3=no)
    def disjunction(left:Concept, right: Concept) -> Concept:
        return Concept('Conditional', term1=left, term2=yes, term3=right)
    def implication(antecedent: Concept, consequent: Concept) -> Concept:
        return Concept('Conditional', term1=antecedent, term2=consequent, term3=yes)
    def equivalence(left: Concept, right Concept) -> Concept:
        return Concept('Conditional', term1=left, term2=right,term3=negation(right))
    def expectation(role: Role, term: Concept):
        return Concept('Marginal', term1=term, term2=yes, role=role)
    def necessity(role: Role, term: Concept):
        return Concept('Marginal', term1=no, term2=negation(term), role=role)
    def possibility(role: Role, term: Concept):
        return negation(Concept('Marginal', term1=no, term2=term, role=role))

    #properties included via decorators
    @property
    def operator(self):
        return __tuple__.getItem(self,0)
    @property
    def symbol(self):
        if self.operator()=='Atom':
            return __tuple__.getItem(self,1)
        else:
            return None
    @property
    def condition(self):
        if self.operator()=='Conditional':
            return __tuple__.getItem(self,1)
        else:
            return None
    @property
    def positive(self):
        if self.operator()=='Conditional':
            return __tuple__.getItem(self,2)
        else:
            return None
    @property
    def negative(self):
        if self.operator()=='Conditional':
            return __tuple__.getItem(self,3)
        else:
            return None
    @property
    def role(self):
        if self.operator()=='Marginal':
            return __tuple__.getItem(self,1)
        else:
            return None
    @property
    def event(self):
        if self.operator()=='Marginal':
            return __tuple__.getItem(self,2)
        else:
            return None
    @property
    def margin(self):
        if self.operator()=='Marginal':
            return __tuple__.getItem(self,3)
        else:
            return None
    
    def __str__(self):
        '''Returns a formatted readable representation of the formula.
        Just return __repr__ for now
        '''
        return self.__repr__()

    def __repr__(self):
        '''Returns a concise mathematical representation of the formula'''
        reps ={
            'Atom': lambda x: x[1],
            'Yes': lambda x: 'Y',
            'No': lambda x: 'N',
            'Conditional': lambda x: '({1}?{2}:{3})'.format(x[1].__repr__(), x[2].__repr__(), x[3].__repr__()),
            'Marginal': lambda x: '.{1}({2}|{3})'.format(x[1].__repr__(),x[2].__repr__(),x[3].__repr__())
            }
        return reps.get(x[0])(x)

    def pretty(self, indent=0):
        '''Returns a multiline formatted version of the formula.
        Returning repr for now'''
        return __repr__()

    #Proof theoretic functions here.
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

class Role(tuple):        
    '''
    An class for representing all ADL roles.

    Roles are immutable and encoded via tuples. 
    We currently just use atomic roles, but use these structures to allow for dynamic modalities in future.
    '''
    def __new__(cls, role: str) -> Role:
        '''Creates an atomic concept, with the symbol symbol
           
           symbol should be an alpha-numeric+underscore string, starting with a lower case letter.
           a '.' is prepended to the symbol to distinguish it from a concept.
        '''
        if(role==None or len(role)<1 or not role[0].is_lower()):
            raise ADLException("Naming error, atomic concept symbols must begin with lower case letters")
        return tuple.__new__(cls, ('.'+role))

    @property
    def role(self):
        return __tuple__.getitem(self,0)

    def __repr__(self):
        '''Gives the concise mathematical representation of the concept
        In this case, just the symbol
        '''
      return self.role

    def __str__(self):
        '''Gives a formatted version of the concept, using English operators, and pretty indenting
        In this case it is just the symbol
        '''
        return self.role






