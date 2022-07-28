<p align="center">
  <img src="https://raw.githubusercontent.com/TycheLibrary/Tyche/master/docs/banner.png" alt="Tyche Logo" height="200" width="480" />
</p>

--------------------------------------------------------------------

Tyche is a Python library to represent and reason about aleatoric information. 
The system includes syntax for describing aleatoric information (in `tyche.language`),
Aleatoric knowledge bases are a generalisation of description logic knowledge bases,
where concepts and roles are aleatoric, in that they are assumed to be determined by the role of a die.
This allows fine-grained modelling of probabilistic belief systems, with a rigorous mathematical foundation.


The *tyche* package consists of the following modules:
- **language**. This is the aleatoric description logic langauge module for representing formulas, and performing basic transformations.
  It consists of the following classes:
     - the logical operators *Concept*, *Conditional*, *Expectation* and *LeastFixedPoint*
     - and the role operators *Role*, *Test*, *Iterate*, and *Combine*
     - *Proof*, theorem proving utilities.   
- **logic**. This is the aleatoric description logic module for reasoning. It consists of the classes:
  - *Axiom*, a class for representing intensional beliefs, or TBooks.
  - *Assertion*, a class for representing extensional beliefs, or ABooks.
  - *Theory*, a class for inference given axioms and assertions, satisfiability and consistency checking
  - this class will have multiple utility classes to support the extensive algorithms required for satisfiability testing
- **models**. This is the module for representing interpretations of an aleatoric theory, 
    which is essentially a belief model for decision support.
    It provides a framework for writing python classes that represent aleatoric information constrained by aleatoric theories.
    It also permits learning and database synchronisation operations, and consists of the classes:
  - *Distribution*, various classes for representing distributions over fields that an individual can have
  - *Individual*, for representing aleatoric individuals. 
    These may be subclassed and decorated with aleatoric concepts, roles and axioms.
  - *BeliefBase*, a module for synchronising individuals with databases,
     checking consistency with theories, handling queries, and performing learning operations given observations.
  - *SqlAleatory* a wrapper for sqlalchemy to handle aleatoric information.

The Tyche project also consists of the package *test* which will contain unit tests as they are written,
and an *examples* directory containing prototypes of python classes that use the tyche package.

This page will be updated as these modules and packages are fleshed out.
