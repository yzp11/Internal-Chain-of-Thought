import json
import copy
import numpy as np
import torch as t
from torch.utils.data import Dataset


class ICLSequence:
    """
    Represents a sequence of word pairs (x, y, t1, t2) and provides methods
    to build prompts and completions for in-context learning.
    """
    def __init__(self, word_pairs: list[tuple[str, str, str, str]]):
        """
        Initialize the sequence with a list of word pairs.

        Args:
            word_pairs (List[Tuple[str, str, str, str]]): 
                Each tuple is of the form (x, y, t1, t2).
        """
        self._word_pairs = word_pairs
        self.x, self.y, self.t1, self.t2 = zip(*word_pairs)

    def __len__(self) -> int:
        """Return the number of word pairs in this sequence."""
        return len(self._word_pairs)

    def __getitem__(self, idx: int) -> tuple[str, str, str, str]:
        """Get the word pair at the specified index."""
        return self._word_pairs[idx]

    def prompt(self) -> str:
        """
        Build and return the prompt string. This prompt includes all Q/A pairs 
        except for the last answer (i.e., it omits the second element of the last pair).
        """
        full_prompt = "\n\n".join(
            [f"Q: {x}\nA: {y}" for x, y, t1, t2 in self._word_pairs]
        )
        # Remove the portion corresponding to the final pair's answer.
        return full_prompt[:-len(self.completion())]

    def completion(self) -> str:
        """
        Return the second element of the last word pair (y[-1]),
        prefixed by a space.
        """
        return " " + self.y[-1]

    def completion_t1(self) -> str:
        """Return t1 from the last word pair, prefixed by a space."""
        return " " + self.t1[-1]

    def completion_t2(self) -> str:
        """Return t2 from the last word pair, prefixed by a space."""
        return " " + self.t2[-1]

    def query(self) -> str:
        """Return x from the last word pair, prefixed by a space."""
        return " " + self.x[-1]


class ICLDataset(Dataset):
    """
    A dataset that generates multiple in-context learning prompts from 
    a larger list of word pairs. Each entry in the dataset can contain:
      - A normal prompt
      - A zero-shot prompt (if enabled)
      - A corrupted prompt (if enabled)
      - A self-corrupted prompt (if enabled)
      - The associated completions and queries
    """
    def __init__(
        self,
        word_pairs: list[tuple[str, str, str, str]],
        size: int,
        n_prepended: int,
        seed: int = 0,
        generate_corrupted: bool = False,
        generate_zero: bool = False,
        generate_self_corrupted: bool = False
    ):
        """
        Initialize the ICLDataset.

        Args:
            word_pairs (List[Tuple[str, str, str, str]]): 
                A list of word pairs: (x, y, t1, t2).
            size (int): 
                Number of samples to generate in this dataset.
            n_prepended (int): 
                Number of examples to prepend as context in each prompt.
            seed (int, optional): 
                Random seed for reproducibility.
            generate_corrupted (bool, optional): 
                Whether to generate a corrupted prompt (random answers).
            generate_zero (bool, optional): 
                Whether to generate a prompt with zero examples (i.e., only the last one).
            generate_self_corrupted (bool, optional): 
                Whether to generate a prompt corrupted by shuffling 
                existing answers among the prepended examples.
        """
        assert (
            n_prepended + 1 <= len(word_pairs)
        ), "Not enough antonym pairs in dataset to create prompt."

        self.word_pairs = word_pairs
        self.answer_list = [wp[1] for wp in word_pairs]
        self.size = size
        self.n_prepended = n_prepended
        self.seed = seed

        # Pre-generate all prompts/completions to avoid repeated computation
        self.prompts = []
        self.corrupted_prompts = []
        self.zero_prompts = []
        self.self_corrupted_prompts = []
        self.completions = []
        self.completions_t1 = []
        self.completions_t2 = []
        self.queries = []

        for i in range(size):
            # Set a seed to ensure reproducibility across runs
            np.random.seed(seed * size + i)

            # Select random pairs for constructing the sequence
            selected_indices = np.random.choice(
                len(self.word_pairs), n_prepended + 1, replace=False
            )
            # Copy pairs so original data is never modified
            selected_pairs = [copy.deepcopy(self.word_pairs[idx]) for idx in selected_indices]

            seq = ICLSequence(selected_pairs)
            # Store normal prompt and completions
            self.prompts.append(seq.prompt())
            self.completions.append(seq.completion())
            self.completions_t1.append(seq.completion_t1())
            self.completions_t2.append(seq.completion_t2())
            self.queries.append(seq.query())

            # Optional: zero-shot prompt 
            if generate_zero:
                zero_seq = ICLSequence(selected_pairs[-1:])
                self.zero_prompts.append(zero_seq.prompt())
            else:
                self.zero_prompts.append('none')

            # Optional: self-corrupted prompt
            if generate_self_corrupted:
                # Shuffle answers among the first n_prepended pairs
                sequence_answers = [wp[1] for wp in selected_pairs][:-1]
                np.random.shuffle(sequence_answers)

                seq_words_modified = copy.deepcopy(selected_pairs)
                for idx_corrupt in range(len(seq_words_modified) - 1):
                    seq_words_modified[idx_corrupt][1] = sequence_answers[idx_corrupt]

                self_corrupted_seq = ICLSequence(seq_words_modified)
                self.self_corrupted_prompts.append(self_corrupted_seq.prompt())
            else:
                self.self_corrupted_prompts.append('none')

            # Optional: random-corrupted prompt
            if generate_corrupted:
                seq_words_modified = copy.deepcopy(selected_pairs)
                for idx_corrupt in range(len(seq_words_modified) - 1):
                    # Assign a random answer from the entire dataset
                    seq_words_modified[idx_corrupt][1] = np.random.choice(self.answer_list)
                corrupted_seq = ICLSequence(seq_words_modified)
                self.corrupted_prompts.append(corrupted_seq.prompt())
            else:
                self.corrupted_prompts.append('none')

    def __len__(self) -> int:
        """Return the total number of samples in this dataset."""
        return self.size

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve a dictionary containing prompts, completions, and queries.

        Returns:
            dict: {
                'prompt': str,
                'zero_prompt': str,
                'corrupted_prompt': str,
                'self_corrupted_prompt': str,
                'completion': str,
                'completion_t1': str,
                'completion_t2': str,
                'query': str
            }
        """
        return {
            'prompt': self.prompts[idx],
            'zero_prompt': self.zero_prompts[idx],
            'corrupted_prompt': self.corrupted_prompts[idx],
            'self_corrupted_prompt': self.self_corrupted_prompts[idx],
            'completion': self.completions[idx],
            'completion_t1': self.completions_t1[idx],
            'completion_t2': self.completions_t2[idx],
            'query': self.queries[idx],
        }


def load_data(data_path: str) -> list[tuple[str, str, str, str]]:
    """
    Load data from JSON and return it in the form of a list of tuples:
    (input, output, t1, t2).

    If the loaded data only has 'input' and 'output' fields, the placeholders
    for t1 and t2 will be 'none'.

    Args:
        data_path (str): Path to the JSON file.

    Returns:
        List[Tuple[str, str, str, str]]: List of tuples containing the data.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If each item has only (input, output), fill t1 and t2 with 'none'.
    if len(data[0]) == 2:
        return [[d['input'], d['output'], 'none', 'none'] for d in data]
    else:
        return [[d['input'], d['output'], d['t1'], d['t2']] for d in data]



def load_data_with_instructions(
    data_dir: str,
    required_instructions: list,
) -> list[dict]:
    data_list = []

    with open(data_dir, "r") as file:
        for line in file:
            json_obj = json.loads(line)
            if set(required_instructions).issubset(set(json_obj["levels2"])):
                data_list.append(json_obj)

    return data_list