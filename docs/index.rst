Welcome to Tyche's documentation!
=================================

Tyche is a Python library to support the representation of, and the reasoning about, aleatoric information.
Aleatoric information is information that has an independent probability of being true each time it is observed
(i.e., each observation is a roll of the dice). For example, every text message that people send _may_ include
emojis, with a different probability for each individual that sent the message. Tyche provides functionality to
reason about this aleatoric information using aleatoric description logic. This allows the probability of truth
of logical statements to be efficiently queried, and allows the probability of the tendencies of individuals to
be learnt through observation.

Tyche provides its main functionality through its `tyche.individuals` module, which facilitates the
construction of ontological knowledge bases with probabilistic beliefs. This allows the simple
representation of individuals, the probabilistic beliefs about them (termed concepts), and the
probabilistic relationships between them (termed roles). Aleatoric description logic sentences
may then be constructed using the `tyche.language` model to be used to query a knowledge base
for a probability, or to be used as an observation to update a knowledge base. This allows
fine-grained modelling of probabilistic belief systems, with a rigorous mathematical foundation.

**GitHub:** `github.com/TycheLibrary/Tyche
<https://github.com/TycheLibrary/Tyche>`_

**PyPi:** `pypi.org/project/Tyche/
<https://pypi.org/project/Tyche>`_

**Related Publications:**

- "Aleatoric Description Logic for Probabilistic Reasoning" by
  Tim French and Thomas Smoker (2021):
  `arXiv link <https://arxiv.org/abs/2108.13036>`_
- "A modal aleatoric calculus for probabilistic reasoning" by
  Tim French, Andrew Gozzard and Mark Reynolds (2018):
  `arXiv link <https://arxiv.org/abs/1812.11741>`_


Table of Contents
#################

.. toctree::
   :maxdepth: 2
   :caption: Modules:

   language
   individuals
   distributions
   probability
   references
   string_utils


Indices and tables
#################

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
