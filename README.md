<p align="center">
  <img src="https://raw.githubusercontent.com/TycheLibrary/Tyche/master/docs/banner.png" alt="Tyche Logo" width="480" />
</p>

Tyche is a Python library to support the representation of, and the reasoning about, aleatoric information.
Aleatoric information is information that has an independent probability of being true each time it is observed
(i.e., each observation is treated as a roll of the dice). For example, every text message that people send _may_
include emojis, with a different probability for each individual that sent the message. Tyche provides functionality
to reason about this aleatoric information using aleatoric description logic. This allows the probability of truth
of logical statements to be efficiently queried, and allows the probability of the tendencies of individuals to
be learnt through observation.

Tyche provides its main functionality through its `tyche.individuals` module, which facilitates the
construction of ontological knowledge bases with probabilistic beliefs. This allows the simple
representation of individuals, the probabilistic beliefs about them (termed concepts), and the
probabilistic relationships between them (termed roles). Aleatoric description logic sentences
may then be constructed using the `tyche.language` module to be used to query a knowledge base
for a probability, or to be used as an observation to update a knowledge base. This allows
fine-grained modelling of probabilistic belief systems, with a rigorous mathematical foundation.

**Documentation:** [tychelibrary.github.io](https://tychelibrary.github.io)

**Tyche Paper:**
- "Tyche: A library for probabilistic reasoning and belief modelling in Python"
  by Padraig Lamont (2022): [link](https://doi.org/10.1007/978-3-031-22695-3_27)

**Related Publications:**
- "Aleatoric Description Logic for Probabilistic Reasoning" by
  Tim French and Thomas Smoker (2021): [arXiv link](https://arxiv.org/abs/2108.13036)
- "A modal aleatoric calculus for probabilistic reasoning" by
  Tim French, Andrew Gozzard and Mark Reynolds (2018): [arXiv link](https://arxiv.org/abs/1812.11741)


## Usage

Tyche may be installed via pip from [PyPi](https://pypi.org/project/Tyche/).
```sh
pip install Tyche
```

The functions and classes of Tyche can then be imported from the tyche package.
```python
from tyche.language import *
from tyche.individuals import *
from tyche.distributions import *
```

Tyche does not yet have tutorials or documentation for its usage. However, example uses of Tyche are
provided in the [examples directory](https://github.com/TycheLibrary/Tyche/tree/main/examples), and
the [source code](https://github.com/TycheLibrary/Tyche/tree/main/tyche) contains doc comments to
explain the functionality of classes and methods. In the future, more extensive documentation of
Tyche is planned.


## Structure

The *tyche* package consists of the following main modules:
- **language**. This is the aleatoric description logic language module for representing sentences. The language
  module also contains the representation used for the value of roles.
- **individuals**. This is the ontological knowledge base individuals module for representing ontologies of
  individuals, the probabilistic beliefs about them (concepts), and the probabilistic relationships between
  them (roles). The individuals module also contains classes and functions that may be used to learn from
  aleatoric description logic observations.
- **distributions**. This module contains utility classes for representing and manipulating probability
  distributions. These distributions may be used to convert continuous quantities into probabilities,
  and to learn the probability distributions from aleatoric description logic observations.

The Tyche project also consists of the package *test* that contains unit tests for the functionality of
Tyche, and the package *examples* that contains example uses of Tyche.

## License

Tyche is licensed under the [MIT License](/LICENSE).
