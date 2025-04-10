from datetime import timedelta
import random
import math
from pyClarion import (Event, Agent, Priority, Input, Pool, Choice, ChunkStore, BaseLevel, Family, NumDict, Atoms, Atom, Chunk, ks_crawl)

# This is the condition with no chunking/grouping

""" Keyspace Definition """

class Number(Atoms):
    _0: Atom; _1: Atom; _2: Atom; _3: Atom; _4: Atom; _5: Atom; _6: Atom; _7: Atom; _8: Atom; _9: Atom

class Chunk(Atoms):
    Num1:Atom; Num2:Atom; Num3:Atom; Num4:Atom; Num5:Atom; Num6:Atom; Num7:Atom

class Memory(Family):
    chunk: Chunk
    number: Number

""" Model Construction """
class Participant(Agent):  
    d: Memory 
    input: Input
    store: ChunkStore
    choice: Choice

    def __init__(self, name: str) -> None:
        d = Memory()
        p = Family()
        e = Family()
        super().__init__(name, d=d, p=p, e=e)
        self.d = d
        with self:
            self.input = Input("input", (d, d))
            self.store = ChunkStore("store", d, d, d)
            self.choice = Choice("choice", p, self.store.chunks)
        self.store.bu.input = self.input.main

    def resolve(self, event: Event) -> None:
        if event.source == self.input:
            self.store.bu.input = self.input.main
            self.store.bu.update()
        if event.source == self.store.bu.update:
            self.choice.trigger()

    def start_trial(self, dt: timedelta, priority: Priority = Priority.PROPAGATION) -> None:
        self.system.schedule(self.start_trial, dt=dt, priority=priority)

    def finish_trial(self, dt: timedelta, priority: Priority = Priority.PROPAGATION) -> None:
        self.system.schedule(self.finish_trial, dt=dt, priority=priority)

""" Initialize stimuli with ungrouped digits (no chunking) """
def init_stimuli_no_chunking(d: Memory, number_list: list[str]) -> list[tuple[Chunk, Chunk]]:
    number, chunk = d.number, d.chunk

    stimuli = []
    for number_str in number_list:
        chunk_instance = (
            + chunk.Num1 ** number[f"_{number_str[0]}"]
            + chunk.Num2 ** number[f"_{number_str[1]}"]
            + chunk.Num3 ** number[f"_{number_str[2]}"]
            + chunk.Num4 ** number[f"_{number_str[3]}"]
            + chunk.Num5 ** number[f"_{number_str[4]}"]
            + chunk.Num6 ** number[f"_{number_str[5]}"]
            + chunk.Num7 ** number[f"_{number_str[6]}"]
        )
        query = (
            + chunk.Num1 ** number[f"_{number_str[0]}"]
            + chunk.Num2 ** number[f"_{number_str[1]}"]
            + chunk.Num3 ** number[f"_{number_str[2]}"]
            + chunk.Num4 ** number[f"_{number_str[3]}"]
            + chunk.Num5 ** number[f"_{number_str[4]}"]
            + chunk.Num6 ** number[f"_{number_str[5]}"]
            + chunk.Num7 ** number[f"_{number_str[6]}"]
        )
        stimuli.append((chunk_instance, query))

    return stimuli

""" Event Processing"""

number_list = ['123456789', '222333444', '555666777', '987654321', '098123456', '123321789', '445566779']

all_results = []  # To hold results for each input string

for num_str in number_list:
    # Create a new participant agent per string (optional, but clean)
    participant = Participant("participant")
    
    # Reinitialize stimuli and chunks for this number
    stimuli = init_stimuli_no_chunking(participant.d, [num_str])

    for target, _ in stimuli:
        participant.store.compile(target)

    # Simulate recall
    results = {"trial": [], "stim": [], "response": [], "correct": [], "recall": []}
    original_string = ""

    for trial, (target_chunk, query_input) in enumerate(stimuli):
        participant.input.send(query_input)

        # Process event queue
        while participant.system.queue:
            participant.system.advance()

        # Poll recall
        response_key = participant.choice.poll()[~participant.store.chunks]
        response_chunk = ks_crawl(participant.system.root, response_key)
        print(f"Response Key: {str(response_chunk)}")

        target_name = str(target_chunk._name_)
        response_name = str(response_chunk._name_) if response_chunk else "None"
        print(f"Trial {trial} → Target: {target_name}, Response: {response_name}")

        results["trial"].append(trial)
        results["stim"].append(target_name)
        results["response"].append(response_name)
        results["correct"].append(response_chunk == target_chunk)

        # Extract digits from the chunk
        if response_chunk:
            digits = []
            for line in str(response_chunk).splitlines():
                if "number._" in line:
                    digit = line.split("number._")[1].strip()
                    digits.append(digit)
            original_string += ''.join(digits)
        else:
            original_string += "   "  # 3 spaces for a missing chunk (for alignment)

    results["recall"].append(original_string)
    all_results.append({
        "input": num_str,
        "reconstructed": original_string,
        "details": results
    })

total = 0
correct = 0
# Print final recap
for entry in all_results:
    print("\n==== Recall Test ====")
    print(f"Input:        {entry['input']}")
    print(f"Response:     {entry['reconstructed']}")
    print(f"Accuracy:     {sum(entry['details']['correct'])}/{len(entry['details']['correct'])}")
    total += len(entry['details']['correct'])
    correct += sum(entry['details']['correct'])
    
# Overall accuracy
print(f"\nOverall Accuracy: {correct}/{total} ({(correct/total)*100:.2f}%)")

