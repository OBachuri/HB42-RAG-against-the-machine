# import argparse
import sys
import os
import logging
from pathlib import Path
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
from transformers import PreTrainedModel
import torch

from src.r_data_model import RetrievedChunk

from typing import cast, Callable
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.__main__ import RagCLI


_SYSTEM_PROMPT = (
    """You are an expert assistant.
Use ONLY the provided context to answer the question.
If the answer is not completely contained in the context,
say:
"I don't have enough information."
""")


class R_LLM():
    """ Class for work with LLM to get answer on question
        using retrieved information """

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        """ Init LLM model """

        self.model_name = model_name

        # Disable the missing token warning (hides the warning)
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HUB_VERBOSITY"] = "error"
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

        # Load the tokenizer and the model
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
            )

    def query(self,
              question: str,
              chunks: list[RetrievedChunk],
              param: RagCLI,
              print_answer: bool = True
              ) -> str:
        """ Get answer on one question """

        # read chunks
        chunk_txt = ""

        for ch_ in chunks:
            file = ch_.file_path
            chunk_id = ch_.chunk_id

            try:
                # f_path = Path(param.data_raw_path) / Path(file)
                f_path = Path(file)
                with open(f_path) as f:
                    source = f.read()
            except Exception as ex:
                print(f"Error: can't read file {file} \n({ex})",
                      file=sys.stderr)
                continue
            chunk_txt += f"""------
<chunk
 File: {file}
 Chunk_id_in_file: {chunk_id}
 from_char: {ch_.first_character_index}
 to_char: {ch_.last_character_index}
>
{source[ch_.first_character_index:ch_.last_character_index+1]}
</chunk>"""

        # prepare the model input
        messages = [
            {"role": "system",
             "content": _SYSTEM_PROMPT + chunk_txt},
            {"role": "user",
             "content": question}
            ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
            )

        #  print("txt:", text)

        model_inputs = self._tokenizer(
            [text], return_tensors="pt").to(self._model.device)

        if print_answer:

            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = dict(
                **model_inputs,
                streamer=streamer,
                max_new_tokens=4000  # 32768
            )

            # for mypy
            generate_func = cast(Callable[..., torch.Tensor],
                                 self._model.generate)

            thread = Thread(target=generate_func,
                            kwargs=generation_kwargs)
            thread.start()

            answer = ""

            for chunk in streamer:
                answer += chunk
                print(chunk, end="", flush=True)

        else:
            # for mypy
            generate_func = cast(Callable[..., torch.Tensor],
                                 self._model.generate)
            generated_ids_ = generate_func(
                **model_inputs,
                max_new_tokens=4000,
                use_cache=True
                )

            # for mypy
            # generated_ids_ = cast(torch.Tensor, generated_ids)

            output_ids = generated_ids_[0][
                len(model_inputs.input_ids[0]):].tolist()

            answer = cast(str, self._tokenizer.decode(
                output_ids,
                skip_special_tokens=True))

        return answer.strip("\n")
