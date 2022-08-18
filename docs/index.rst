Welcome to Tyche's documentation!
=================================

Tyche is a Python library to support the representation of, and the reasoning about, aleatoric information.
Aleatoric information is information that has an independent probability of being true each time it is observed
(i.e., each observation is treated as a roll of the dice). For example, every text message that people send _may_
include emojis, with a different probability for each individual that sent the message. Tyche provides functionality
to reason about this aleatoric information using aleatoric description logic. This allows the probability of truth
of logical statements to be efficiently queried, and allows the probability of the tendencies of individuals to
be learnt through observation.

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

   language
   individuals
   distributions
   probability
   references
   string_utils


Indices and tables
#################

- :ref:`genindex`
- :ref:`modindex`
- :ref:`search`
