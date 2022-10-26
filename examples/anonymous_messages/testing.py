"""
Contains the code used to test both the Tyche implementation and
the ProbLog implementation of the anonymous messages example.
"""
import logging
import time

import numpy as np

from examples.anonymous_messages.base_model import AnonymousMessagesImplementation, Message, Model, ModelPerson
from examples.anonymous_messages.problog_model import ProbLogImplementation
from examples.anonymous_messages.tyche_model import TycheImplementation


def construct_target_model() -> Model:
    """
    Constructs the ground-truth target model that is used for this example.
    This is the model that we will use to randomly generate observations to
    evaluate how well we can estimate the model from them.
    """
    bob = ModelPerson("Bob", uses_emoji=0.9, capitalises_first_word=0.4, is_positive=0.15)
    alice = ModelPerson("Alice", uses_emoji=0.1, capitalises_first_word=0.8, is_positive=0.4)
    jeff = ModelPerson("Jeff", uses_emoji=0.5, capitalises_first_word=0.5, is_positive=0.5)
    bob.conversed_with[alice.name] = 60
    bob.conversed_with[jeff.name] = 40
    alice.conversed_with[bob.name] = 25
    alice.conversed_with[jeff.name] = 75
    jeff.conversed_with[alice.name] = 70
    jeff.conversed_with[bob.name] = 30
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
        no_tests: int = 100, min_messages: int = 1, max_messages: int = 10,
        rng_seed: int = 42
) -> tuple[list[int], dict[str, list[float]], list[float]]:
    """
    Runs tests on the given implementation to evaluate its performance at inferring authors of messages.
    """
    rng = np.random.default_rng(rng_seed)
    target_model = construct_target_model()
    imp.set_model(target_model)

    print(
        f"Evaluating the accuracy of the {imp.name} implementation to "
        f"predict the author of random sets of messages ({no_tests} tests per person, per message count)"
    )
    result_no_messages: list[int] = []
    result_correct_percentages: dict[str, list[float]] = {person.name: [] for person in target_model.all}
    result_durations_ms: list[float] = []
    for no_messages in range(min_messages, max_messages + 1):
        start_time = time.time()

        correct_per_person = {p.name: 0 for p in target_model.all}
        for recipient in target_model.all:
            for test in range(no_tests):
                # Sample who sent a message to the recipient.
                person_name = recipient.sample_conversed_with(rng)
                person = target_model.by_name[person_name]

                # Generate a random conversation sent from the current person to the recipient.
                conversation = generate_conversation(rng, person, no_messages)

                # Calculate who the model would predict to have written the conversation.
                predicted_author_probs = imp.query_author_probabilities(recipient.name, conversation)
                max_prob: float = -1
                max_prob_author: str = ""
                for author_name, prob in predicted_author_probs.items():
                    if prob > max_prob:
                        max_prob = prob
                        max_prob_author = author_name

                # Check if the prediction was correct.
                if max_prob_author == person_name:
                    correct_per_person[recipient.name] += 1

        correct_percentages = {name: 100 * correct / no_tests for name, correct in correct_per_person.items()}
        duration_ms = (time.time() - start_time) * 1000 / no_tests

        result_no_messages.append(no_messages)
        for name, correct_percentage in correct_percentages.items():
            result_correct_percentages[name].append(correct_percentage)
        result_durations_ms.append(duration_ms)

        print(f".. {no_messages} message{'s' if no_messages > 1 else ''}: ".ljust(17) + ", ".join(
            [f"{name} = {correct_percentage:.1f}%" for name, correct_percentage in correct_percentages.items()]
        ).ljust(45) + f"({duration_ms:.2f} ms per evaluation)")

    return result_no_messages, result_correct_percentages, result_durations_ms


def print_learning_results(trial_results: list[Model]):
    """
    Collates the results from all the trials and prints them out.
    """
    print("Learnt People:")
    target_model = construct_target_model()
    for target in target_model.all:
        learned: list[ModelPerson] = []
        for model in trial_results:
            learned.append(model.by_name[target.name])

        learned_uses_emoji: list[float] = [p.uses_emoji for p in learned]
        learned_capitalises: list[float] = [p.capitalises_first_word for p in learned]
        learned_is_positive: list[float] = [p.is_positive for p in learned]

        learned_conversed_with: dict[str, list[float]] = {}
        for p in learned:
            for name, prob in p.conversed_with.items():
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


def run_learning_test(
        imp: AnonymousMessagesImplementation, *,
        no_trials: int = 1, no_observations: int = 5000, repetitions: int = 2,
        min_messages: int = 2, max_messages: int = 4,
        rng_seed: int = 42
):
    """
    Runs tests on the given implementation to evaluate its performance at
    learning the ground-truth model.
    """
    rng = np.random.default_rng(rng_seed)
    target_model = construct_target_model()

    print(
        f"Running with {no_trials} trials, {no_observations} observations per trial, "
        f"{repetitions} repetitions of those observations, "
        f"and conversations of {min_messages} to {max_messages} messages"
    )
    trial_results: list[Model] = []
    example_observations = None
    for trial_no in range(no_trials):
        print(f".. running trial {trial_no + 1}")
        trial_start_time = time.time()

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
                # Use Tyche's functionality to print the observation.
                tyche_print_imp = TycheImplementation()
                obs = tyche_print_imp.build_received_messages_observation(received_messages)
                example_observations.append(f"Observe at {person_name}: {str(obs)}")

        # Observe the generated observations in the model.
        for _ in range(repetitions):
            rng.shuffle(observations)
            for person_name, received_messages in observations:
                imp.apply_received_messages_observation(person_name, received_messages)

        # Record the results.
        trial_results.append(imp.get_model())
        print(f"  * took {time.time() - trial_start_time:.2f} seconds")

    print()
    print("Example Observations:")
    print("- " + "\n- ".join(str(obs) for obs in example_observations))
    print()

    print(f"Ground-Truth People:\n{target_model}")
    print()

    print_learning_results(trial_results)

    print(f"Initial People that were Trained into the Learned People:\n{create_initial_learn_model()}")
    print()


class FilteringStreamHandler(logging.StreamHandler):
    """
    Filters out unnecessary information from the ProbLog logging.
    """
    IGNORE_LINES_CONTAINING: list[str] = ["Cycle breaking:", "Clark's completion:", "DSharp compilation:", "Grounding:"]

    def emit(self, record):
        message = record.getMessage()
        for text in FilteringStreamHandler.IGNORE_LINES_CONTAINING:
            if text in message:
                return

        super(FilteringStreamHandler, self).emit(record)


if __name__ == "__main__":
    # Enables ProbLog's debugging output.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[FilteringStreamHandler()]
    )

    print("--------------------------")
    print("   Tyche Implementation   ")
    print("--------------------------")
    tyche_imp = TycheImplementation()
    run_inference_test(tyche_imp)
    print()
    run_learning_test(tyche_imp)

    print()
    print("----------------------------")
    print("   ProbLog Implementation   ")
    print("----------------------------")
    problog_imp = ProbLogImplementation()
    run_inference_test(problog_imp)
    print()
    run_learning_test(problog_imp)

