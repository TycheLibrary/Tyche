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

   def __str__(self):
       '''
       gives a compact string representation of the structure of the formula
       in terms of primitive operators
       '''
       pass

   def __repr__(self):
       '''
       gives the constructor string of the object
       '''
       pass

   #maybe also include a function for giving an optimised inline string representation of a formula.
       
    def __eq__(self, other):
        '''return true if formulas are identical'''
        pass
        
    def __lt__(self, other):
        '''establishes a syntactic ordering over formula'''
        pass               

    def eval(self, concepts, roles):
        '''returns the probability of a concept, 
        given the lambda evaluation of roles and concepts.
        It is assumed concepts is a function that 
        maps concept symbols to probabilities, 
        and roles is a function that maps role symbols
        to probability distributions of tuples of concept functions and role functions'''
        pass

    #Proof theoretic functions here.
    def normal_form(self):
        '''Returns the tree normal form of the formula, where atoms are ordered alphabetically.'''
        pass

    def is_equivalent(self, concept):
        '''
        Returns true if this Concept is provably equivalent to concept
        delegates to normal form function.
        Two concepts have the same normal form if and only if 
        they have the same evaluation function
        '''
        return self.normal_form() == concept.normal_form()

    def is_weaker(self, concept):
        '''Returns true if the probability of this Concept is provably 
        necesssarily less than or equal to the probabiliy of concept'''
        #need to consider how to effectively implement this
        #I think move everything to normal form and then some traversal?
        #However, it is more than just inclusion. 
        #eg x/\~x is always weaker than y\/~y
        #need to factor out inclusion, then find separating constants
        pass
 

    def is_stronger(self, concept):
        '''Returns true if the probability of this concept is provably
        necessarily greater than or equal to the probability of adl_form'''
        return concept.is_weaker(self)

   '''
   Class variables for Yes, No and Flip
   '''
   yes = Concept('Yes')
   no = Concept('No')
   flip = Concept('Flip')

   def when(self, condition):
       '''constructor for inline conditional formulas.
       a.when(b) is the same as a-> b or Conditional(b,a,Yes)
       a.when(b).otherwise(c) is the same as (b?a:c)

       '''
       cond = Conditional(condition, self, yes)
       cond.otherwise = lambda if_no:#add an otherwise clause if desired. 
           cond.if_no = if_no
           delattr(cond, otherwise)
           return cond

   def is_not(self, concept):
       '''
       inline negation operator
       '''
       return no.when(concept).otherwise(yes)

   def and(self, concept):
       '''
       inline conjunction operator,
       note ordering for lazy evaluation
       '''
       return concept.when(self).otherwise(no)

   def or(self, concept):
       '''
       inline disjunction operator
       '''
       return yes.when(self).otherwise(concept)

   '''
   Other inline operators to define:
   concept.for(role)
   concept.for(role).given(margin)
   concpet.necessary_for(role)
   concept.possible_for(role)
   others...
   '''


#Concept class for indivisible concepts
class Atom(Concept):
    '''
    This class represents the atomic concepts Yes, No, and named concepts.
    '''
    def __init__(self, symbol):
        if is_valid_symbol(symbol):
            self.symbol = symbol

    def is_valid_atom(s):
        '''Checks a string contains only alphanumeric characters or underscore'''
        valid = s is not None and len(s)>0 and \
                (s[0].islower() or s=='Yes' or s=='No' or s=='Flip'))
        if valid:
            for c in s[1:]:
                valid = valid and ((c.isalnum) or c=='_')
        return valid    

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return self.symbol

    def __eq__(self, other):
        return self.symbol == other.symbol
        
    def __lt__(self, other):
        if other == no: return False
        if self == no: return True
        if self == yes: return False
        if other == yes: return True
        if other.symbol == None: return True
        return self.symbol < other.symbol
                        
    def eval(self, concepts, roles):
        if self == no: return 0
        if self == yes: return 1
        return concepts(symbol)
        

    #Proof theoretic functions here.
    def normal_form(self):
        '''Returns the tree normal form of a Atom, where atoms are ordered alphabetically.'''
        return self

    def is_equivalent(self, concept):
        return self == concept

    def is_weaker(self, concept):
        return concept == yes or self == no
 

######End of class Atom#####

##Class for conditional constructs
class Conditional(Concept):
    '''class for representng the aleatoric if then else construct in the language'''

    def __init__(self, condition, if_yes, if_no):
        self.condition = condition
        self.if_yes = if_yes
        self.if_no = if_no


    def __str__(self):
        return '('+str(self.condition)+'?'+str(self.if_yes)+':'+str(self.if_no)+')'

    def __repr__(self):
        return 'Conditional(condition='+self.condition+', if_yes='+self.if_yes+', if_no='+self.is_no+')'

    def __eq__(self, other):
        try:
            return self.condition == other.condition and\
                    self.if_yes == other.if_yes and\
                    self.if_no == other.if_no
        except:
            return False
        
    def __lt__(self, other):
        if other.symbol != None: return False #all terminals are less than conditionals
        if other.role != None: return False #all expectations are less than conditionals
        if other.condition != None: # comparing conditionals
            if self.if_yes < other.if_yes: return True
            elif self.if_yes == other.if_yes:
                if self.if_no < other.if_no: return True
                elif self.if_no==other.if_no: return self.condition < other.conditon
        return True # conditonals are less than fixed points (for now)        


    def eval(self, concepts, roles):
        cond = self.condition.eval(concepts, roles)
        return cond*if_yes.eval(concepts, roles) + (1-cond)*if_no.eval(concepts, roles)

    #Proof theoretic functions here.
    def normal_form(self):
        '''Returns the tree normal form of the conditional, 
        by recursively calling normal for on sub elements.'''
        pass #too come. Long hack


    def is_weaker(self, concept):
        pass
 
####End of class conditional#####
        
class Expectation(Concept):
    '''class for representng the aleatoric expectation  construct in the language'''

    def __init__(self, role, concept):
        self.role = role #role object
        self.concept = concept #concept object

    def __str__(self):
        return str(self.role)+'.'+ self.concept

    def __repr__(self):
        return 'Expectation(role='+repr(self.role)+', concept='+repr(self.concept)+')'

    def __eq__(self, other):
        try:
            return self.role == other.role and\
                    self.concept == other.concept
        except:
            return False
        
    def __lt__(self, other):
        '''
        Expectations are greater than terminals, but less than conditionals?
        '''
        if other.symbol != None: return False #all terminals are less than expectations
        if other.role != None: #comparing expectations
            if self.role < other.role: return True
            elif self.role == other.role: return self.condition < other.condition

    def eval(self, concepts, roles):
        '''
        Complex one, need to extract a distribution of roles:
        '''
        dist = roles(self.role)#dist is a dictionary mapping (concepts, roles) lambda to probabilities
        prob = 0.0 
        for (c,r) in dist.keys:
            prob = prob + dist[(c,r)]*self.concept.eval(c,r)
        return prob    

    #Proof theoretic functions here.
    def normal_form(self):
        '''Returns the tree normal form of the conditional, 
        by recursively calling normal for on sub elements.'''
        pass

    def is_equivalent(self, concept):
        pass

    def is_weaker(self, concept):
        pass
 
####End of class expectation#####

        
class LeastFixedPoint(Concept):
    '''class for representng the aleatoric fixed point construct in the language'''

    def __init__(self, variable, concept):
        if is_linear(variable, concept):
            self.variable = variable #role object
            self.concept = concept #concept object
            else: raise TycheException('Variable '+variable+'not linear in '+concept)    

    def __str__(self):
        '''
        Use X as the fixed point quantifier, 
        if least and greatest not relavant?
        or is assignment appropriate x<=(father.(bald?YES:x)) (GFP is x>=(father.(bald?x:NO)) "all bald on the male line")
        eg LFP-x(father.(bald?YES:x)) the probability of having a bald ancestor on the male line.
        '''
        return self.variable+'<=('+ self.concept+')'

    def __repr__(self):
        return 'LeastFixedPoint(variable='+repr(self.variable)+', concept='+repr(self.concept)+')'

    def __eq__(self, other):
        try:
            return self.variable == other.variableand\
                    self.concept == other.concept
        except:
            return False
        
    def __lt__(self, other):
        '''
        FixedPoints are greater than everything else
        '''
        if other.variable == None: return False #everything else is less than fixed_points
        if self.concept < other.concept: return True
            elif self.concept == other.concept: return self.variable < other.variable

    def eval(self, concepts, roles):
        '''
        Complex one, needs iteration or equation solving.
        '''
        pass

    #Proof theoretic functions here.
    def normal_form(self):
        '''Returns the tree normal form of the conditional, 
        by recursively calling normal for on sub elements.'''
        pass

    def is_equivalent(self, concept):
        pass

    def is_weaker(self, concept):
        pass
 
    def is_linear(variable, concept):
        '''
        class method to test whether variable is linear in concept
        '''
        pass


class Role:        
    '''
    An class for representing all ADL roles.
    Abstract class laying out the methods.

    We currently just use atomic roles. 
    Dynamic roles will be realised as abbreviations using complex concepts..
    '''
####refactor below####?

    def __init__(self, symbol):
        '''
        Creates an atomic concept, with the symbol symbol
        symbol should be an alpha-numeric+underscore string, starting with a lower case letter.
        a '.' is prepended to the symbol to distinguish it from a concept.
        '''
        if(not Concept.is_valid_symbol(role)):
            raise ADLException("Naming error, atomic concept symbols must begin with lower case letters")
        self.symbol = symbol

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

