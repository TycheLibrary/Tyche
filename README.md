#tyche

Python library to represent and reason about aleatoric information. 
The system includes syntax for describing aleatoric information (in package langauge),
Aletaoric knowledge bases are a generalisation of description logic knowledge bases,
where concepts and roles are aleatoric, in that they are assumed to be determined by the role of a die.
This allows fine grained modelling of probabilistic belief systems, with a rigorous mathematical foundation.


The *tyche* package consists of the following subpackages and modules:
- **language**. This is the aleatoric description logic langauge subpackage for representing formulas, and performing basic transformations.
  It consists of the following modules:
  - **adl**, a module for representing the syntax of the language, including:
     - the logical operators *Concept*, *Conditional*, *Expectation* and *LeastFixedPoint*
     - and the role operators *Role*, *Test*, *Iterate*, and *Combine*
  - **proof**, a module consisting of theorem proving utilities.   
- **logic**. This is the aleatoric description logic package for intensional reasoning. It consists of the modules:
  - **axiom**, a module for representing beliefs of a universal nature, or TBooks.
  - **assertion**, a module for representing beliefs pertsaiing to a specific individual, or ABooks.
  - **theory**, a module for inference given axioms and assertions.
- **models**. This is the module for representing extensional information, or a specific realisation of an aleatoric theory.
  It provides a framework for writing python classes that represent aleatoric information constrained by aleatoric theories.
  It also permits learning and database synchronisation operations.
- **test** package will contain unit tests as they are written.

This page will be updated as these modules and packages are fleshed out.


