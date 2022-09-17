"""
Contains the code used to test both the Tyche implementation and
the ProbLog implementation of the anonymous messages example.
"""
from typing import cast

import numpy as np

from examples.anonymous_messages.base_model import AnonymousMessagesImplementation, Message, Model, ModelPerson
from examples.anonymous_messages.tyche_model import Person, TycheImplementation


def construct_target_model() -> Model:
    """
    Constructs the ground-truth target model that is used for this example.
    This is the model that we will use to randomly generate observations to
    evaluate how well we can estimate the model from them.
    """
    bob = ModelPerson("Bob", uses_emoji=0.9, capitalises_first_word=0.4, is_positive=0.15)
    alice = ModelPerson("Alice", uses_emoji=0.1, capitalises_first_word=0.8, is_positive=0.4)
    jeff = ModelPerson("Jeff", uses_emoji=0.5, capitalises_first_word=0.5, is_positive=0.5)
    bob.conversed_with[alice.name] = 1.5
    bob.conversed_with[jeff.name] = 1
    alice.conversed_with[bob.name] = 1
    alice.conversed_with[jeff.name] = 3
    jeff.conversed_with[alice.name] = 3
    jeff.conversed_with[bob.name] = 1.5
    return Model(bob, alice, jeff)


def create_initial_learn_model() -> Model:
    """
    Constructs the initial model to train from.
    """
    bob = ModelPerson("Bob")
    alice = ModelPerson("Alice")
    jeff = ModelPerson("Jeff")
    bob.conversed_with[alice.name] = 1
    bob.conversed_with[jeff.name] = 1
    alice.conversed_with[bob.name] = 1
    alice.conversed_with[jeff.name] = 1
    jeff.conversed_with[alice.name] = 1
    jeff.conversed_with[bob.name] = 1
    return Model(bob, alice, jeff)


def generate_conversation(rng: np.random.Generator, person: ModelPerson, message_count: int) -> list[Message]:
    """
    Samples a random set of messages written by the given person.
    """
    return [person.sample_message(rng) for _ in range(message_count)]


def run_inference_test(
        imp: AnonymousMessagesImplementation, *,
        no_tests: int = 2000, min_messages: int = 1, max_messages: int = 10):
    """
    Runs tests on the given implementation to evaluate its performance at inferring authors of messages.
    """
    rng = np.random.default_rng()
    target_model = construct_target_model()
    imp.set_model(target_model)

    print(f"Evaluating the accuracy of the {imp.name} implementation to predict the author of random sets of messages")
    for no_messages in range(min_messages, max_messages + 1):
        correct_per_person = {p.name: 0 for p in target_model.all}
        for person in target_model.all:
            for test in range(no_tests):
                # Generate a random conversation from the current person.
                conversation = generate_conversation(rng, person, no_messages)

                # Calculate who the model would predict to have written the conversation.
                predicted_author_probs = imp.query_author_probabilities(conversation)
                max_prob: float = -1
                max_prob_author: str = ""
                for author_name, prob in predicted_author_probs.items():
                    if prob > max_prob:
                        max_prob = prob
                        max_prob_author = author_name

                # Check if the prediction was correct.
                if max_prob_author == person.name:
                    correct_per_person[person.name] += 1

        print(f".. {no_messages} message{'s' if no_messages > 1 else ''}: ".ljust(16) + ", ".join(
            [f"{name} = {100 * correct / no_tests:.1f}%" for name, correct in correct_per_person.items()]
        ))


def run_learning_test(
        imp: AnonymousMessagesImplementation, *,
        no_trials: int = 10, no_observations: int = 5_000, repetitions: int = 2,
        min_messages: int = 2, max_messages: int = 4):
    """
    Runs tests on the given implementation to evaluate its performance at
    learning the ground-truth model.
    """
    rng = np.random.default_rng()
    tyche_print_imp = TycheImplementation()
    target_model = construct_target_model()
    imp.set_model(target_model)

    print(
        f"Running with {no_trials} trials, {no_observations} observations per trial, "
        f"{repetitions} repetitions of those observations, "
        f"and conversations of {min_messages} to {max_messages} messages"
    )
    trial_results: list[Model] = []
    example_observations = None
    for trial_no in range(no_trials):
        print(f".. running trial {trial_no + 1}")

        initial_learned_model = create_initial_learn_model()
        imp.set_model(initial_learned_model)

        # Generate random indirect observations about Bob, Alice, and Jeff.
        observations: list[tuple[str, list[Message]]] = []
        for index, person in enumerate(target_model.all):
            # Evenly distribute the observations between each person.
            from_no = round(index * no_observations / len(target_model.all))
            to_no = round((index + 1) * no_observations / len(target_model.all))
            for _ in range(from_no, to_no):
                conversation_partner_name = person.sample_conversed_with(rng)
                conversation_partner = target_model.by_name[conversation_partner_name]
                conversation = generate_conversation(
                    rng, conversation_partner, rng.integers(min_messages, max_messages, endpoint=True)
                )
                observations.append((person.name, conversation))

        rng.shuffle(observations)
        if example_observations is None:
            example_observations = []
            for person_name, received_messages in observations[0:4]:
                obs = tyche_print_imp.build_received_messages_observation(received_messages)
                example_observations.append(f"Observe at {person_name}: {str(obs)}")

        # Observe the generated observations in the model.
        for _ in range(repetitions):
            rng.shuffle(observations)
            for person_name, received_messages in observations:
                imp.apply_received_messages_observation(person_name, received_messages)

        # Record the results.
        trial_results.append(imp.get_model())

    print()
    print("Example Observations:")
    print("- " + "\n- ".join(str(obs) for obs in example_observations))
    print()

    print("Ground-Truth People:")
    tyche_print_imp.set_model(target_model)
    print("- " + "\n- ".join(str(p) for p in tyche_print_imp.all))
    print()

    # Collate trial results.
    print("Learnt People:")
    tyche_print_imp.set_model(target_model)
    target_people = tyche_print_imp.all
    for target in target_people:
        learned = []
        for model in trial_results:
            tyche_print_imp.set_model(model)
            learned.append(tyche_print_imp.by_name[target.name])

        learned_uses_emoji = [p.uses_emoji() for p in learned]
        learned_capitalises = [p.capitalises_first_word() for p in learned]
        learned_is_positive = [p.is_positive() for p in learned]

        learned_conversed_with: dict[str, list[float]] = {}
        for p in learned:
            for ctx, prob in p.conversed_with():
                name = cast(Person, ctx).name
                if name not in learned_conversed_with:
                    learned_conversed_with[name] = []
                learned_conversed_with[name].append(100 * prob)

        conversed_with_entries = [
            f"{np.mean(p):.1f}% ± {np.std(p):.1f}%: {n}" for n, p in learned_conversed_with.items()
        ]

        print(f"- {target.name}("
              f"capitalises_first_word={np.mean(learned_capitalises):.3f} ± {np.std(learned_capitalises):.3f}, "
              f"is_positive={np.mean(learned_is_positive):.3f} ± {np.std(learned_is_positive):.3f}, "
              f"uses_emoji={np.mean(learned_uses_emoji):.3f} ± {np.std(learned_uses_emoji):.3f}, "
              f"conversed_with={{" + ", ".join(conversed_with_entries) + "})")
    print()

    print("Initial People that were Trained into the Learned People:")
    tyche_print_imp.set_model(create_initial_learn_model())
    print("- " + "\n- ".join(str(p) for p in tyche_print_imp.all))
    print()


if __name__ == "__main__":
    print("--------------------------")
    print("   Tyche Implementation   ")
    print("--------------------------")
    tyche_imp = TycheImplementation()
    run_inference_test(tyche_imp)
    print()
    run_learning_test(tyche_imp)
