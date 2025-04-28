from datetime import timedelta
import math
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


class Group(Atoms):
    group: Atom


class List(Atoms):
    list: Atom


class Value(Atoms):
    value: Atom


class Memory(Family):
    number: Number
    value: Value
    group: Group
    list: List


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
        self.pool["store.bu"] = (self.store.bu.main)

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


""" Knowledge Initialization (with chunking) """
def init_stimuli(memory: Memory, list_data: list[str]) -> list[tuple[Chunk, Chunk]]:
    number, value, group, lst = memory.number, memory.value, memory.group, memory.list

    list_chunk = "list" ^ +lst.list ** number[f"_1"]
    chunks = [list_chunk]

    num_groups = math.ceil(len(list_data[0]) / 3)

    # Make the groups
    for i in range(1, num_groups + 1):
        x = round(1.3 - (0.1 * (3 - i)), 1)
        y = round((2.0 - x), 1)
        g_i = f"g_{i}" ^ +x * list_chunk + y * group.group ** number[f"_{i}"]
        chunks.append(g_i)

    # Same process for values
    for i in range(1, num_groups + 1):
        count = 0
        for j in range(3 * (i - 1), len(list_data[0])):
            if count < 3:
                x = 1.0
                y = round(1.1 - (0.1 * (count - 1)), 1)
                z = round((2.0 - y), 1)
                v_i = (
                    f"v_{count}" ^
                    +x * list_chunk +
                    y * group.group ** number[f"_{i}"] +
                    z * value.value ** number[f"_{list_data[0][j]}"]
                )
                chunks.append(v_i)
                count += 1
            else:
                break
    return chunks


def build_recall_cue(memory: Memory, target_digit: str, position: int) -> Chunk:
    number, value, group, lst = memory.number, memory.value, memory.group, memory.list

    cue = (
        f"recall_cue_{position}" ^
        +value.value ** number[f"_{target_digit}"]
    )
    return cue


""" Event Processing """
participant = Participant("participant")
number_list = [
    "123456789", "987654321", "456789123", "321654987",
    "789123456", "654321789", "234567890"
]
results = []

# Loop through sequences
for trial_idx, sequence in enumerate(number_list):
    # --- Study Phase ---
    stimuli = init_stimuli(participant.d, [sequence])

    for chunk in stimuli:
        participant.store.compile(chunk)

    # Advance system after compilation
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
                if response and "v_" in response._name_:
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
    correct_order = sum(1 for a, b in zip(sequence, recalled) if a == b)
    correct_digits = sum(1 for digit in set(sequence) if digit in recalled) / len(sequence)
    results.append({
        "trial": trial_idx + 1,
        "stim": sequence,
        "response": recalled,
        "correct rate": correct_order / len(sequence),
        "correct digits": correct_digits / len(sequence)
    })

# --- Print Results ---
for result in results:
    print(f"Trial {result['trial']}:")
    print(f"Stimulus: {result['stim']}")
    print(f"Response: {result['response']}")
    print(f"Correct Rate (with order): {result['correct rate']:.2f}")
    print(f"Correct Digits Rate (without order): {result['correct digits']:.2f}")
    print("-" * 30)

# --- Plot Results ---
trials = [r['trial'] for r in results]
correct_rates = [r['correct rate'] for r in results]
correct_digits = [r['correct digits'] for r in results]

plt.figure(figsize=(10, 6))

plt.plot(trials, correct_rates, label="Correct Recall Rate (with order)", marker='o')
plt.plot(trials, correct_digits, label="Correct Recall Rate (without order)", marker='x')

plt.title("Recall Performance Across Trials")
plt.xlabel("Trial")
plt.ylabel("Recall Rate")
plt.ylim(0, 1.1)
plt.xticks(trials)
plt.legend()
plt.grid(True)

plt.show()
