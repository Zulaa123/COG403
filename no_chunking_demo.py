from datetime import timedelta
import matplotlib.pyplot as plt
from pyClarion import (
    Event, Agent, Priority, Input, Pool, Choice, ChunkStore,
    Family, NumDict, Atoms, Atom, Chunk, ks_crawl
)

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


class List(Atoms):
    list: Atom


class Value(Atoms):
    value: Atom


class Memory(Family):
    number: Number
    value: Value
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
            self.store = ChunkStore("store", d, d, d)
            self.input = Input("input", self.store.chunks)
            self.inhibition = Input("inhib", self.store.chunks)
            self.pool = Pool("pool", p, self.store.chunks, func=NumDict.sum)
            self.choice = Choice("choice", p, self.store.chunks, sd=1e-3)
        self.store.td.input = self.input.main
        self.store.bu.input = self.store.td.main
        self.pool["store.bu"] = self.store.bu.main

        self.pool["inhibition"] = (
            self.inhibition.main,
            lambda d: d.neg()
        )
        self.choice.input = self.pool.main

    def resolve(self, event: Event) -> None:
        if event.source == self.store.bu.update:
            self.choice.trigger()


""" Knowledge Initialization (NO Chunking) """

def init_stimuli_no_chunking(memory: Memory, list_data: list[str], num: int) -> list[Chunk]:
    """ Initialize stimuli without chunking.
    """
    number, value, lst = memory.number, memory.value, memory.list

    # Create a list chunk for the number
    list_chunk = f"list_{num}" ^ +lst.list ** number[f"_{num}"]
    chunks = [list_chunk]

    # For each digit, directly associate it with the list
    for i, digit in enumerate(list_data[0]):
        c = (
            f"v_{i}" ^ +list_chunk
            + value.value ** number[f"_{digit}"]
        )
        chunks.append(c)

    return chunks


""" Event Processing """
# Initialize participant and number list
participant = Participant("participant")
number_list = [
    "123456789", "222333444", "555666777", "888999000",
    "098178322", "111122233", "632876343"
]
results = []
list_cues = []


# Loop through sequences for chunk compile
for trial_idx, sequence in enumerate(number_list):
    # --- Study Phase ---
    stimuli = init_stimuli_no_chunking(participant.d, [sequence], trial_idx)
    list_cues.append(stimuli[0])

    for chunk in stimuli:
        participant.store.compile(chunk)

    # Advance system after compilation
    while participant.system.queue:
        participant.system.advance()

# Loop through sequences for recall
for trial_idx, sequence in enumerate(number_list):
    # --- Recall Phase ---
    recalled = ""
    cue = list_cues[trial_idx]

    participant.input.send({cue: 1})
    participant.inhibition.send({cue: 1})

    while participant.system.queue:
        event = participant.system.advance()
        if event.source == participant.choice.select:
            response_key = participant.choice.poll()[~participant.store.chunks]
            response = ks_crawl(participant.system.root, response_key)

            if response and response._name_.startswith("v"):
                digit = str(response)[-1]
                recalled += digit

            if response != participant.store.chunks.nil:
                wm = {response_key: 1, **participant.input.main[0].d}
                participant.inhibition.send(wm)
                participant.input.send(wm)

    # --- Store Results ---
    correct = recalled == sequence[:len(recalled)]
    correct_order = sum(1 for a, b in zip(sequence, recalled) if a == b)
    correct_digits = sum(1 for digit in sequence if digit in recalled)
    results.append({
        "trial": trial_idx + 1,
        "stim": sequence,
        "response": recalled,
        "correct recall": correct_order / len(sequence),
        "correct order recall": correct_digits / len(sequence)
    })


# --- Print Results ---
for result in results:
    print(f"Trial {result['trial']}:")
    print(f"Stimulus: {result['stim']}")
    print(f"Response: {result['response']}")
    print(f"Correct Rate (with order): {result['correct recall']:.2f}")
    print(f"Correct Digits Rate (without order): {result['correct order recall']:.2f}")
    print("-" * 30)

  
# --- Plot Results ---
trials = [r['trial'] for r in results]
correct_rates = [r['correct recall'] for r in results]
correct_digits = [r['correct order recall'] for r in results]

plt.figure(figsize=(6, 4))

plt.plot(trials, correct_rates, label="With Order", marker='o')
plt.plot(trials, correct_digits, label="Without Order", marker='x')

plt.title("Recall Accuracy Across Trials (No Chunking)", fontsize=12, fontname="Arial")
plt.xlabel("Trial", fontsize=11, fontname="Arial")
plt.ylabel("Correct Recall Rate", fontsize=11, fontname="Arial")
plt.xticks(trials, fontsize=10, fontname="Arial")
plt.yticks(fontsize=10, fontname="Arial")

plt.ylim(0, 1.1)
plt.legend(title="Recall Type", fontsize=10, title_fontsize=11, prop={"family": "Arial"})
plt.grid(True, linestyle='--', linewidth=0.5)

plt.show()

