import re
from typing import Dict, Any, List

import torch
from tqdm import tqdm

from utils.text_processing import preprocessing


def _match_assistant_output(generated_text: str, pattern: str, logger, example) -> str:
    # Extract only the assistant's response using the EOS token
    match = re.search(pattern, generated_text, re.DOTALL)
    if match is not None:
        assistant_response = match.group(1).strip()  # Get the content between the markers
    else:
        logger.warning("Wrong output match", generated_text, example)
        assistant_response = generated_text.strip()  # Fallback if no match is found
    return assistant_response


def get_model_predictions(model, tokenizer, examples, system_prompt: str, user_wrap: str, config, logger) \
        -> List[Dict[str, Any]]:
    outputs = list()
    for example in tqdm(examples):
        text = example["text"]
        message = [
            {
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": user_wrap.format(preprocessing(text))
            }
        ]
        inputs = tokenizer.apply_chat_template(
            message,
            # tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt").to("cuda")

        with torch.no_grad():  # Disable gradient calculation for inference
            if config.temp > 0:
                kwargs = dict(temperature=config.temp, do_sample=True, num_beams=2)
            else:
                kwargs = dict(do_sample=False)
            generated_ids = model.generate(inputs, max_new_tokens=config.max_tokens,
                                           pad_token_id=tokenizer.eos_token_id, **kwargs)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Extract only the assistant's response using the EOS token
        pattern = rf"{re.escape(config.boa)}(.*)"
        assistant_response = _match_assistant_output(generated_text, pattern, logger, example)
        outputs.append({
            "id": example["id"],
            "text": text,
            "label": example["label"],
            "input": message,  # parsed input
            "output": assistant_response,  # parsed output
            "raw_output": generated_text,  # raw generated str,
            "example": example
        })
    return outputs
