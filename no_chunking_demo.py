from datetime import timedelta
from pyClarion import (
    Event, Agent, Priority, Input, Pool, Choice, ChunkStore,
    Family, NumDict, Atoms, Atom, Chunk, ks_crawl
)
import matplotlib.pyplot as plt


""" Keyspace Definition """
class Number(Atoms):
    _0: Atom
    _1: Atom
    _2: Atom
    _3: Atom
    _4: Atom
    _5: Atom
    _6: Atom
    _7: Atom
    _8: Atom
    _9: Atom

class Value(Atoms):
    value: Atom


class Memory(Family):
    number: Number
    value: Value


""" Model Construction """
class Participant(Agent):
    d: Memory
    input: Input
    store: ChunkStore
    pool: Pool
    choice: Choice

    def __init__(self, name: str) -> None:
        p = Family()
        e = Family()
        d = Memory()
        super().__init__(name, p=p, e=e, d=d)
        self.d = d
        with self:
            self.input = Input("input", (d, d))
            self.store = ChunkStore("store", d, d, d)
            self.pool = Pool("pool", p, self.store.chunks, func=NumDict.sum)
            self.choice = Choice("choice", p, self.store.chunks)
        self.store.bu.input = self.input.main
        self.pool["store.bu"] = self.store.bu.main

        self.pool["inhibition"] = (
            self.store.bu.main,
            lambda d: d.neg()
        )
        self.choice.input = self.pool.main

    def resolve(self, event: Event) -> None:
        if event.source == self.store.bu.update:
            self.choice.trigger()

    def start_trial(
        self,
        dt: timedelta,
        priority: Priority = Priority.PROPAGATION
    ) -> None:
        self.system.schedule(self.start_trial, dt=dt, priority=priority)

    def finish_trial(
        self,
        dt: timedelta,
        priority: Priority = Priority.PROPAGATION
    ) -> None:
        self.system.schedule(self.finish_trial, dt=dt, priority=priority)


# This is the condition with no chunking/grouping
""" Knowledge Initialization (without chunking) """
def init_stimuli_no_chunking(memory: Memory, list_data: list[str]) -> list[Chunk]:
    number, value = memory.number, memory.value

    chunks = []

    # Create individual chunks for each digit without grouping
    for i, digit in enumerate(list_data[0]):
        chunk = f"digit_{i}" ^ +value.value**number[f"_{digit}"]
        chunks.append(chunk)

    return chunks


def build_recall_cue(memory: Memory, target_digit: str, position: int) -> Chunk:
    number, value = memory.number, memory.value

    cue = (
        f"recall_cue_{position}" ^
        +value.value**number[f"_{target_digit}"]
    )
    return cue


""" Event Processing (No Chunking) """
participant = Participant("participant_no_chunking")
number_list = [
    "123456789", "987654321", "456789123", "321654987",
    "789123456", "654321789", "234567890"
]
results_no_chunking = []

# Loop through sequences
for trial_idx, sequence in enumerate(number_list):
    # --- Study Phase ---
    stimuli = init_stimuli_no_chunking(participant.d, [sequence])

    for chunk in stimuli:
        participant.store.compile(chunk)

    # Advance system after compilation (ensure everything is processed)
    while participant.system.queue:
        participant.system.advance()

    # --- Recall Phase ---
    recalled = ""

    for i, target_digit in enumerate(sequence):
        cue = build_recall_cue(participant.d, target_digit, i)
        participant.input.send(cue)

        while participant.system.queue:
            participant.system.advance()

        try:
            response_key = participant.choice.poll()[~participant.store.chunks]

            if response_key is not None:
                response = ks_crawl(participant.system.root, response_key)

                if response and "digit_" in response._name_:
                    digit = str(response)[-1]
                    recalled += digit
                else:
                    recalled += "?"
            else:
                recalled += "?"
        except Exception as e:
            print(f"Error during recall: {e}")
            recalled += "?"

    # --- Store Results ---
    correct = recalled == sequence[:len(recalled)]
    correct_rate = sum(1 for a, b in zip(recalled, sequence) if a == b) / len(sequence)
    correct_digits = sum(1 for digit in set(sequence) if digit in recalled) / len(sequence)
    results_no_chunking.append({
        "trial": trial_idx + 1,
        "stim": sequence,
        "response": recalled,
        "correct order rate": correct_rate,
        "correct digits rate": correct_digits
    })

# --- Print Results ---
for result in results_no_chunking:
    print(
        f"Trial {result['trial']}: Stimulus: {result['stim']}, Response: {result['response']}, "
        f"Correct Order Rate: {result['correct order rate']:.2f}, Correct Digits Rate: {result['correct digits rate']:.2f}"
    )

# --- Plot Results ---
trial_numbers = [r['trial'] for r in results_no_chunking]
correct_order_rates = [r['correct order rate'] for r in results_no_chunking]
correct_digits_rates = [r['correct digits rate'] for r in results_no_chunking]

plt.figure(figsize=(10, 6))

plt.plot(trial_numbers, correct_order_rates, label="Correct Recall Rate (with order)", marker='o')
plt.plot(trial_numbers, correct_digits_rates, label="Correct Recall Rate (without order)", marker='s')

plt.title("Recall Performance (No Chunking)")
plt.xlabel("Trial Number")
plt.ylabel("Recall Rate")
plt.ylim(0, 1.1)
plt.xticks(trial_numbers)
plt.legend()
plt.grid(True)

plt.show()
