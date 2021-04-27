#tyche

Python implementation of an aleatoric knowledge base, and association utilities.
Aletaoric knowledge bases are a generalisation of description logic knowledge bases,
where concepts and roles are aleatoric, in that they are assumed to be determined by the role of a die.
This allows fine grained modelling of probabilistic belief systems, with a rigorous mathematical foundation.


The *tyche* package consists of the following modules:
- **adl** this is the aleatoric description logic langauge module for representing formulas, performing proofs etc.
- **models** this is the models module for creating schemas for individuals, assigning values to interpretations, serialising with a database, model checking, and learning.
- **akb** this is the aleatoric knowledge base module for defining axiom schemas over the model schema, consistentcy checking and synthesis of interpretations.
- **api** this is the abstract programming interface module for specifying models and axioms, perform learning and execute queries.

The *test* package will contain unit tests as they are written.

This page will be updated as these modeuls and packages are fleshed out.


