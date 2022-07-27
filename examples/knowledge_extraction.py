"""
An example use of Tyche for knowledge extraction from a set of messages,
where the source that sent each message is unknown, but the recipient
is known. The properties of the messages can be used to learn the
different tendencies of the possible sources of the message, without
explicitly knowing which sources sent any of the messages. For example,
we can learn that Bob uses emojis often, without knowing what messages
Bob sent. This is achieved through the use of the observation learning
mechanisms of the individuals module of Tyche.
"""
from tyche.individuals import *
from tyche.language import *
import functools


# We want to decay earlier observations faster than later observations.
# This is done to improve convergence as more observations are added,
# while allowing the model to change more during early learning.
decaying_learning_strat = StatisticalLearningStrategy(
    decay_rate=0.95, decay_rate_for_decay_rate=0.95
)


class Person(Individual):
    """
    An example person that has a set of preferences when writing messages.
    """
    name: str

    def __init__(
            self, name: str,
            uses_emoji: TycheConceptValue = 0.5,
            capitalises_first_word: TycheConceptValue = 0.5,
            is_positive: TycheConceptValue = 0.5):

        super().__init__(name)
        self._conversed_with = ExclusiveRoleDist()
        self._uses_emoji = uses_emoji
        self._capitalises_first_word = capitalises_first_word
        self._is_positive = is_positive

    @role()
    def conversed_with(self):
        return self._conversed_with

    @concept(learning_strat=decaying_learning_strat)
    def uses_emoji(self):
        return self._uses_emoji

    @uses_emoji.setter
    def uses_emoji(self, uses_emoji: float):
        self._uses_emoji = uses_emoji

    @concept(learning_strat=decaying_learning_strat)
    def capitalises_first_word(self):
        return self._capitalises_first_word

    @capitalises_first_word.setter
    def capitalises_first_word(self, capitalises_first_word: float):
        self._capitalises_first_word = capitalises_first_word

    @concept(learning_strat=decaying_learning_strat)
    def is_positive(self):
        return self._is_positive

    @is_positive.setter
    def is_positive(self, is_positive: float):
        self._is_positive = is_positive

    def sample_message(self) -> tuple[bool, bool, bool]:
        """ Randomly samples this individual to a Person with known properties. """
        uses_emoji = random.uniform(0, 1) < self._uses_emoji
        capitalises_first_word = random.uniform(0, 1) < self._capitalises_first_word
        is_positive = random.uniform(0, 1) < self._is_positive
        return uses_emoji, capitalises_first_word, is_positive


if __name__ == "__main__":
    # Construct a target model. This is the ground truth that we will use to randomly
    # generate observations to evaluate how well we can estimate the model from them.
    target_bob = Person("Bob", uses_emoji=0.9, capitalises_first_word=0.4, is_positive=0.15)
    target_alice = Person("Alice", uses_emoji=0.1, capitalises_first_word=0.8, is_positive=0.4)
    target_jeff = Person("Jeff", uses_emoji=0.5, capitalises_first_word=0.5, is_positive=0.5)
    target_bob.conversed_with.add(target_alice)#, 1/3)
    target_bob.conversed_with.add(target_jeff)#, 2/3)
    target_alice.conversed_with.add(target_bob)#, 2/9)
    target_alice.conversed_with.add(target_jeff)#, 7/9)
    target_jeff.conversed_with.add(target_alice)#, 2/10)
    target_jeff.conversed_with.add(target_bob)#, 8/10)

    target_people = [target_bob, target_alice, target_jeff]

    def create_initial_learn_model() -> tuple[Person, Person, Person]:
        """ Constructs the initial model to train from. """
        bob = Person("Bob")
        alice = Person("Alice")
        jeff = Person("Jeff")
        bob.conversed_with.add(alice)
        bob.conversed_with.add(jeff)
        alice.conversed_with.add(bob)
        alice.conversed_with.add(jeff)
        jeff.conversed_with.add(alice)
        jeff.conversed_with.add(bob)
        return bob, alice, jeff

    # The parameters for our evaluation of the knowledge extraction.
    # We run multiple trials to obtain mean & std dev.
    no_trials = 10
    no_observations = 1000

    # We generate 'conversations' of a small number of messages.
    min_messages = 1
    max_messages = 3

    print(f"Running with {no_trials} trials, {no_observations} observations")
    trial_results = []
    example_observations = []
    for trial_no in range(no_trials):
        print(f".. running trial {trial_no + 1}")

        learned_people = list(create_initial_learn_model())
        learned_people_by_name: dict[str, Person] = {person.name: person for person in learned_people}

        # Generate random indirect observations about Bob, Alice, and Jeff.
        uses_emoji = Atom("uses_emoji")
        capitalises_first_word = Atom("capitalises_first_word")
        is_positive = Atom("is_positive")
        for _ in range(no_observations):
            # The context person is the person that we observed having a conversation with someone.
            target_ctx = target_people[random.randint(0, len(target_people) - 1)]
            learned_ctx = learned_people_by_name[target_ctx.name]

            # Sample the messages of the conversation.
            partner = cast(Person, target_ctx.conversed_with.sample())
            messages = []
            for message_no in range(random.randint(min_messages, max_messages)):
                m_uses_emoji, m_capitalises, m_is_positive = partner.sample_message()
                o_uses_emoji = uses_emoji if m_uses_emoji else uses_emoji.complement()
                o_capitalises = capitalises_first_word if m_capitalises else capitalises_first_word.complement()
                o_is_positive = is_positive if m_is_positive else is_positive.complement()
                messages.append(o_uses_emoji & o_capitalises & o_is_positive)

            # Construct the observation.
            obs_messages = functools.reduce(lambda a, b: a & b, messages)
            observation = Expectation("conversed_with", obs_messages)
            if len(example_observations) < 10:
                example_observations.append(f"Observe at {target_ctx.name}: {str(observation)}")

            # Apply the observation to the learned ctx person.
            learned_ctx.observe(observation)

        trial_results.append(learned_people_by_name)

    print()
    print("Example Observations:")
    print("- " + "\n- ".join(str(obs) for obs in example_observations))
    print()

    print("Target People:")
    print("- " + "\n- ".join(str(p) for p in target_people))
    print()

    # Collate trial results.
    print("Learned People:")
    for target in target_people:
        learned = [people[target.name] for people in trial_results]
        uses_emoji = [p.uses_emoji for p in learned]
        capitalises_first_word = [p.capitalises_first_word for p in learned]
        is_positive = [p.is_positive for p in learned]
        print(f"- {target.name}("
              f"capitalises_first_word={np.mean(capitalises_first_word):.3f} ± {np.std(capitalises_first_word):.3f}, "
              f"is_positive={np.mean(is_positive):.3f} ± {np.std(is_positive):.3f}, "
              f"uses_emoji={np.mean(uses_emoji):.3f} ± {np.std(uses_emoji):.3f})")
    print()

    print("Initial People that were Trained into the Learned People:")
    print("- " + "\n- ".join(str(p) for p in create_initial_learn_model()))
    print()