from datetime import timedelta
import math
from pyClarion import (
    Event, Agent, Input, Pool, Choice, ChunkStore,
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
            self.store = ChunkStore("store", d, d, d)
            self.input = Input("input", self.store.chunks)
            self.inhibition = Input("inhib", self.store.chunks)
            self.pool = Pool("pool", p, self.store.chunks, func=NumDict.sum)
            self.choice = Choice("choice", p, self.store.chunks, sd = 1e-3)
        self.store.td.input = self.input.main    
        self.store.bu.input = self.store.td.main
        self.pool["store.bu"] = (self.store.bu.main)

        self.pool["inhibition"] = (
            self.inhibition.main, 
            lambda d: d.neg()
        )
        self.choice.input = self.pool.main

    def resolve(self, event: Event) -> None:
        if event.source == self.store.bu.update:
            self.choice.trigger()


""" Knowledge Initialization (with chunking) """

def init_stimuli(memory: Memory, list_data: list[str], num: int) -> list[tuple[Chunk, Chunk]]:
    """
    Initialize the stimuli for the chunking model.
    This function creates chunks for the list and its corresponding values.
    """
    number, value, group, lst = memory.number, memory.value, memory.group, memory.list

    # Create the list chunk
    list_chunk = f"list_{num}" ^ + lst.list ** number[f"_{num}"]
    chunks = [list_chunk]

    # Calculate the number of groups based on the length of the list
    num_groups = math.ceil(len(list_data[0]) / 3)

    # Make the groups
    for i in range(1, num_groups + 1):
        x = round(1.3 - (0.1 * (i-1)), 1)
        y = round((2.0 - x), 1)
        g_i = f"g_{i}_list{num}" ^ + x * list_chunk + y * group.group ** number[f"_{i}"]
        chunks.append(g_i)

    # Same process for values
    idx = 0
    for i in range(1, num_groups + 1):
        count = 0
        for j in range(3 * (i - 1), len(list_data[0])):
            if count < 3:
                # Calculate the weights for the features
                x = 1.0
                y = round(1.1 - (0.1 * (count - 1)), 1)
                z = round((2.0 - y), 1)
                v_i = (
                    f"v_{idx}_g{i}_list{num}" ^
                    +x * list_chunk +
                    y * group.group ** number[f"_{i}"] +
                    z * value.value ** number[f"_{list_data[0][j]}"]
                )
                chunks.append(v_i)
                count += 1
                idx += 1
            else:
                break
    return chunks



""" Event Processing """

# Initialize the participant and the number list
participant = Participant("participant")
number_list = [
    "123456789", "222333444", "555666777", "888999000",
    "098178322", "111122233", "632876343"
]
results = []

# Initialize the list of cues (the list chunks)
list_cues = []

# Loop through sequences
for trial_idx, sequence in enumerate(number_list):
    # --- Study Phase ---
    stimuli = init_stimuli(participant.d, [sequence], trial_idx)
    list_cues.append(stimuli[0])

    for chunk in stimuli:
        participant.store.compile(chunk)
    
    # Advance system after compilation
    while participant.system.queue:
        participant.system.advance()

for trial_idx, sequence in enumerate(number_list):
    # --- Recall Phase ---
    recalled = ""
    cue = list_cues[trial_idx]

    # Send cue to the system
    participant.input.send({cue: 1})
    # Inhibit the cue to prevent reactivation
    participant.inhibition.send({cue: 1})

    while participant.system.queue:
        event = participant.system.advance()
        # Check if the event is a choice selection
        if event.source == participant.choice.select:
            response_key = participant.choice.poll()[~participant.store.chunks]
            response = ks_crawl(participant.system.root, response_key)

            # Reconstruct the recalled digit from the response
            if response and response._name_.startswith("v_"):
                digit = str(response)[-1]
                recalled += digit
            
            # If the response is not nil, send it to the system
            if response != participant.store.chunks.nil:
                wm = {response_key : 1, **participant.input.main[0].d}
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

plt.title("Recall Accuracy Across Trials", fontsize=12, fontname="Arial")
plt.xlabel("Trial", fontsize=11, fontname="Arial")
plt.ylabel("Correct Recall Rate", fontsize=11, fontname="Arial")
plt.xticks(trials, fontsize=10, fontname="Arial")
plt.yticks(fontsize=10, fontname="Arial")

plt.ylim(0, 1.1)
plt.legend(title="Recall Type", fontsize=10, title_fontsize=11, prop={"family": "Arial"})
plt.grid(True, linestyle='--', linewidth=0.5)

plt.show()

