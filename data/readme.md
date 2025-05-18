# Pun Datasets Collection

This folder contains all the datasets used in our work, except for JOKER, which could not be released per the authors'
instructions.

## Format

Each dataset is provided as a JSON file, where each sample is formatted as follows:

```json5
{
  "text": "str",
  // Text of the pun or non-pun

  "w_p": "str|null",
  // Pun word for puns or null for non-puns

  "w_a": "str|null",
  // Alternative word for puns or null for non-puns

  "s_p": "str|null",
  // Interpretation of w_p (pun sense) for puns or null for non-puns

  "s_a": "str|null",
  // Interpretation of w_a (alternative sense) for puns or null for non-puns

  "c_w": "list[str]|null",
  // Optional list of contextual words supporting both senses or null for non-puns

  "explanation": "str|null",
  // Human-annotated explanation (only for PunEval)

  "label": "int",
  // Target label: 1 (pun) or 0 (non-pun)

  "is_het": "bool",
  // true for het-puns, false for hom-puns, null for non-puns

  "id": "str",
  // Unique ID of the sample within the current JSON file
}
```

## Dataset list

- **PunEval**: contains the three splits we used;
- **NAP**: the Newly Annotated Puns dataset;
- **PunnyPattern**: each JSON file contains 200 samples with a different language pattern;
- **PunBreak**: dataset of altered puns. The additional `type` attribute in each sample object specifies the
  substitution category:
    - `pos`: a pun from NAP or PunEval;
    - `ns`: nonsensical phonetic substitution;
    - `sp`: pun word synonym;
    - `sa`: alternative word synonym;
    - `ra`: nonsensical random word;
    - `neg`: control group of 100 generated non-pun sentences.