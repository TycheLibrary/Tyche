'''
The base ADL language module, with classes for 
representing and reasoning with ADL formulas.

ADL is designed to support both mathematical notion and a formal english notion.
'''

class TycheException(Exception):
    '''
    Class for detailing ADL exceptions
    '''

    def __init__(self, message):
        self.message= "TycheException:" + message

'''
Refactor
Simpler class system. No need for abstract classes of formula.
The base classes are: No, Yes, Concept, Variable, Conditional, Expectation, LeastFixedPoint.
Each formula class must provide the methods 
__str__, 
__repr__, 
eval (acts on lambdas, returns a lambda)
normal_form (produces a normal form of the formula)
__equal__ (for an ordering of formulas)
__lt__ (for ordering)

'''
class Concept
   '''
   Abstract class for representing formulas in the language
   Maybe need to add in these methods at the bottom.
   '''

   def when(self, condition):
       '''constructor for inline conditional formulas.
       a.when(b) is the same as a-> b or Conditional(b,a,Yes)
       a.when(b).otherwise(c) is the same as (b?a:c)

       '''
       cond = Conditional(condition, self, Yes)
       cond.otherwise = lambda if_no:#add an otherwise clause if desired. 
           cond.if_no = if_no
           delattr(cond, otherwise)
           return cond

   def is_not(self, concept):
       '''
       inline negation operator
       '''
       return No.when(concept).otherwise(Yes)

   def and(self, concept):
       '''
       inline conjunction operator,
       note ordering for lazy evaluation
       '''
       return concept.when(self).otherwise(No)

   def or(self, concept):
       '''
       inline disjunction operator
       '''
       return Yes.when(self).otherwise(concept)




class Atom(Concept):
    '''
    This class represents the atomic concepts Yes, No, and named concepts.
    '''
    def __init__(self, symbol):
        if is_valid_symbol(symbol):
            self.symbol = symbol

    def is_valid_atom(s):
        '''Checks a string contains only alphanumeric characters or underscore'''
        valid = s is not None and len(s)>0 and (s[0].islower() or s=='Yes' or s=='No')
        if valid:
            for c in s[1:]:
                valid = valid and ((c.isalnum) or c=='_')
        return valid    

    '''
    Class variables for Yes and No
    '''
    yes = Concept('Yes')
    no = Concept('No')


    def __str__(self):
        return self.symbol

    def __repr__(self):
        return self.symbol

    def __eq__(self, other):
        return self.symbol == other.symbol
        
    def __lt__(self, other):
        if self == no:
            return True
        if other == no:
            return False
        if self == yes:
            return True
        if other == yes:
            return False
        return self.symbol < other.symbol
                        
    def eval(self, concepts, roles):
        '''returns the probability of a concept, 
        given the lambda evaluation of roles and concepts.
        It is assumed concepts is a function that 
        maps concept symbols to probabilities, 
        and roles is a function that maps role symbols
        to probability ditributions of tuples of concept functions and role functions'''
        if self == no:
            return 0
        if self == yes:
            return 1
        return concepts(symbol)
        

    #Proof theoretic functions here.
    def normal_form(self):
        '''Returns the tree normal form of a Atom, where atoms are ordered alphabetically.'''
        return self

    def is_equivalent(self, concept):
        '''Returns true if this Concept is provably equivalent to adl_form'''
        return self == concept

    def is_weaker(self, concept):
        '''Returns true if the probability of this Concept is provably 
        necesssarily less than or equal to the probabiliy of adl_form'''
        return concept == yes
 
    def is_stronger(self, concept):
        '''Returns true if the probability of this concept is provably
        necessarily greater than or equal to the probability of adl_form'''
        return concept == no

######End of class Atom#####


class Conditional(Concept):
    '''class for representng the aleatoric if then else construct in the language'''





####End of refactor######


class Concept(tuple):
    '''
    A class for representing all ADL concepts.

    Concepts are immutable and encoded via tuples, where the first term of each tuple indicates the type of formula.
    The constructor builds atomic concepts, and complex concepts are constructed using the static methods provided.
    '''

    def __new__(cls, operator, *kwargs):
        '''Constructs all formulas as tuples.
        Usage is: 
            c = Concept('No')
            c = Concept('Yes')
            c = Concept('Atom',symbol=s)
            c = Concept('Conditional', condition=x, if_yes=y, if_no=n)
            c = Concept('Expectation', role=r, concept=c)
            c = Concept('LeastFixedPoint', variable=v, concept=c)
        '''
        constructors = {#lambdas to construct formulas
            'No': lambda c, keys: tuple.__new__(c, ('No',)),
            'Yes': lambda c, keys: tuple.__new__(c, ('Yes',)),
            'Atom': lambda c, keys: tuple.__new__(c, ('Atom', keys['symbol'])),
            'Conditional': lambda c, keys: tuple.__new__(c, ('Conditional', keys['condition'], keys['if_yes'], keys['if_no'])),
            'Expectation': lambda c, keys: tuple.__new__(c, ('Expectation', keys['role'], keys['concept'] )
            'LeastFixedPoint': lambda c, keys: tuple.__new__(c, ('LeastFixedPoint', keys['variable'], keys['concept'] )
            }#language core
        if operator=='Atom' and not cls.is_valid_symbol(symbol):
            raise ADLException('Symbols must begin with lowercase letter, and contain only alphanumeric characters or underscores')
        if not operator in ['Yes','No','Atom','Conditional','Expectation','LeastFixedPoint']:
            raise ADLException('Operator must be Yes, No, Atom, Conditional or Marginal')
        return constructors.get(operator)(cls, kwargs)


    def is_valid_symbol(s):
        '''Checks a string contains only alphanumeric chacaters or underscore'''
        valid = s is not None and len(s)>0  and s[0].islower()
        if valid:
            for c in s[1:]:
                valid = valid and ((c.isalnum) or c=='_')
        return valid    


    #properties included via decorators
    @property
    def operator(self):
        return self[0]

    @property
    def symbol(self):
        if self.operator=='Atom':
            return self[1]
        else:
            return None

    @property
    def condition(self):
        if self.operator=='Conditional':
            return self[1]
        else:
            return None

    @property
    def if_yes(self):
        if self.operator=='Conditional':
            return self[2]
        else:
            return None

    @property
    def if_no(self):
        if self.operator=='Conditional':
            return self[3]
        else:
            return None

    @property
    def role(self):
        if self.operator=='Expectation':
            return self[1]
        else:
            return None

    @property
    def concept(self):
        if self.operator=='Expectation' or self.operator=="LeastFixedPoint":
            return self[2]
        else:
            return None

    @property
    def variable(self):
        if self.operator=='LeastFixedPoint':
            return self[1]
        else:
            return None
    
    def __str__(self):
        '''Returns a formatted readable representation of the formula.
        Just return __repr__ for now
        '''
        return self.__repr__()

    def __repr__(self):
        '''Returns a concise mathematical representation of the formula'''
        reps = {
            'No': lambda x: 'N',
            'Yes': lambda x: 'Y',
            'Atom': lambda x: x.symbol,
            'Conditional': lambda x: '({0}?{1}:{2})'.format(x.condition.__repr__(), x.if_yes.__repr__(), x.if_no.__repr__()),
            'Expectation': lambda x: '[{0}]({1})'.format(x.role.__repr__(), x.concept.__repr__()),
            'LeastFixedPoint': lambda x: 'LFP_{0}_({1})'.format(x.variable.__repr(), x.concept.__repr__()) 
            }
        return reps.get(self.operator)(self)

    def pretty(self, indent=0):
        '''Returns a multiline formatted version of the formula.
        Returning repr for now'''
        return __repr__(self)

    def __eq__(self,other):
        eqs = {
                'No': lambda x, y: True,
                'Yes': lambda x, y: True,
                'Atom': lambda x, y: x.symbol == y.symbol,
                'Conditional': lambda x, y: \
                        x.condition==y.condition and \
                        x.if_yes==y.if_yes and \
                        x.if_no==y.if_no,
                'Marginal': lambda x, y:\
                        x.role==y.role and \
                        x.concept==y.concept
                'LeastFixedPoint': lambda x, y:\
                        x.variable==y.variable and \
                        x.concept==y.concept
                }
        return self.operator==other.operator and \
                eqs[self.operator](self,other)
                                
                        
    def eval(self):
        '''returns a lambda to evaluate the Concept, 
        given the lambda evaluation of roles and concepts'''
        pass

    #Proof theoretic functions here.
    def normal_form(self):
        '''Returns the tree normal form of a Concept, where atoms are ordered alphabetically.'''
        pass

    def is_equivalent(self, adl_form: 'Concept'):
        '''Returns true if this Concept is provably equivalent to adl_form'''
        pass

    def is_weaker(self, adl_form: 'Concept'):
        '''Returns true if the probability of this Concept is provably 
        necesssarily less than or equal to the probabiliy of adl_form'''
        pass
 
    def is_stronger(self, adl_form: 'Concept'):
        '''Returns true if the probability of this concept is provably
        necessarily greater than or equal to the probability of adl_form'''

class Role(tuple):        
    '''
    An class for representing all ADL roles.

    Roles are immutable and encoded via tuples. 
    We currently just use atomic roles, but will also represent dynamic roles.
    '''
    def __new__(cls, role):
        '''Creates an atomic concept, with the symbol symbol
           
           symbol should be an alpha-numeric+underscore string, starting with a lower case letter.
           a '.' is prepended to the symbol to distinguish it from a concept.
        '''
        if(not Concept.is_valid_symbol(role)):
            raise ADLException("Naming error, atomic concept symbols must begin with lower case letters")
        return tuple.__new__(cls, (role))

    @property
    def role(self):
        return self[0]

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

#static constructors for different types of formulas
yes = Concept('Yes')
no = Concept('No')
#define a concept to be a fair coin flip by convention?
coin = Concept('Atom','fair_coin')
#static functions for generating functions
def atom(symbol):
    return Concept('Atom',symbol=symbol)
def conditional(condition, if_yes, if_no):
    return Concept('Conditional', condition=condition, if_yes=if_yes, if_no=if_no)
def expectation(role, concept):
    return Concept('Exepctation', role=role, concept=concept)
def least_fixed_point(variable, concept):
    return Concept('LeastFixedPoint', variable=variable, concept=concept)
#some abbreviations provided too
def negation(term):
    return conditional(term, no, yes)
def conjunction(left, right):
    return conditional(left, right, no)
def disjunction(left, right):
    return conditional(left, yes, right)
def implication(antecedent, consequent):
    return conditional(antecedent, consequent, yes)
def equivalence(left, right):
    return conditional(left, right, negation(right))

#ongoing
def marginal(role, concept, margin):#hmmm need a way to avoid name clashes and check for monotonicity.
    variable = atom('x')
    return least_fixed_point(atom('x'),conditional(coin,expectation(conjunction(concept,margin))


def necessity(role, term):
    return Concept('Marginal', term1=no, term2=negation(term), role=role)
def greatest_fixed_point(variable,  concept):
    return negation(Concept('Marginal', term1=no, term2=term, role=role))

