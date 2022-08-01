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
    # Construct the ground-truth model. This is the model that we will use to randomly
    # generate observations to evaluate how well we can estimate the model from them.
    target_bob = Person("Bob", uses_emoji=0.9, capitalises_first_word=0.4, is_positive=0.15)
    target_alice = Person("Alice", uses_emoji=0.1, capitalises_first_word=0.8, is_positive=0.4)
    target_jeff = Person("Jeff", uses_emoji=0.5, capitalises_first_word=0.5, is_positive=0.5)
    target_bob.conversed_with.add(target_alice)
    target_bob.conversed_with.add(target_jeff)
    target_alice.conversed_with.add(target_bob)
    target_alice.conversed_with.add(target_jeff)
    target_jeff.conversed_with.add(target_alice)
    target_jeff.conversed_with.add(target_bob)

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
    min_messages = 2
    max_messages = 4
    uses_emoji = Concept("uses_emoji")
    capitalises_first_word = Concept("capitalises_first_word")
    is_positive = Concept("is_positive")

    def generate_conversation_observation(person: Person, message_count: int) -> ADLNode:
        # Sample the properties of random messages.
        messages = []
        for message_no in range(message_count):
            m_uses_emoji, m_capitalises, m_is_positive = person.sample_message()
            o_uses_emoji = uses_emoji if m_uses_emoji else uses_emoji.complement()
            o_capitalises = capitalises_first_word if m_capitalises else capitalises_first_word.complement()
            o_is_positive = is_positive if m_is_positive else is_positive.complement()
            messages.append(o_uses_emoji & o_capitalises & o_is_positive)

        # Combine the messages into one observation.
        return functools.reduce(lambda a, b: a & b, messages)

    print(f"Running with {no_trials} trials, {no_observations} observations, "
          f"and conversations of {min_messages} to {max_messages} messages")
    trial_results = []
    example_observations = []
    for trial_no in range(no_trials):
        print(f".. running trial {trial_no + 1}")

        learned_people = list(create_initial_learn_model())
        learned_people_by_name: dict[str, Person] = {person.name: person for person in learned_people}

        # Generate random indirect observations about Bob, Alice, and Jeff.
        for _ in range(no_observations):
            # The context person is the person that we observed having a conversation with someone.
            target_ctx = target_people[random.randint(0, len(target_people) - 1)]
            learned_ctx = learned_people_by_name[target_ctx.name]

            # Sample the messages of the conversation.
            partner = cast(Person, target_ctx.conversed_with.sample())
            conversation = generate_conversation_observation(partner, random.randint(min_messages, max_messages))
            observation = Expectation("conversed_with", conversation)
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
        learned_uses_emoji = [p.uses_emoji for p in learned]
        learned_capitalises = [p.capitalises_first_word for p in learned]
        learned_is_positive = [p.is_positive for p in learned]
        print(f"- {target.name}("
              f"capitalises_first_word={np.mean(learned_capitalises):.3f} ± {np.std(learned_capitalises):.3f}, "
              f"is_positive={np.mean(learned_is_positive):.3f} ± {np.std(learned_is_positive):.3f}, "
              f"uses_emoji={np.mean(learned_uses_emoji):.3f} ± {np.std(learned_uses_emoji):.3f})")
    print()

    print("Initial People that were Trained into the Learned People:")
    print("- " + "\n- ".join(str(p) for p in create_initial_learn_model()))
    print()

    print("Evaluating the accuracy of the target model to predict the author of random sets of messages")
    no_tests = 2000
    for no_messages in range(1, 11):
        correct_per_person = {p.name: 0 for p in target_people}
        for person in target_people:
            for test in range(no_tests):
                # Generate a random conversation from the current person.
                conversation = generate_conversation_observation(person, no_messages)

                # Calculate who the model would predict to have written the conversation.
                highest_prob = -1
                highest_prob_person = None
                for potential_person in target_people:
                    prob = potential_person.eval(conversation)
                    if prob > highest_prob:
                        highest_prob = prob
                        highest_prob_person = potential_person

                # Check if the prediction was correct.
                if highest_prob_person.name == person.name:
                    correct_per_person[person.name] += 1

        print(f".. {no_messages} message{'s' if no_messages > 1 else ''}: ".ljust(16) + ", ".join(
            [f"{name} = {100 * correct / no_tests:.1f}%" for name, correct in correct_per_person.items()]
        ))
