import os
import pickle
import random
import asyncio
import argparse
import torch
import tiktoken
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from common import TEMPLATE_MAP, OPTION_MAP, generate_few_shot_examples
from gpt_labeling import get_client, generate


def parse_args():
    parser = argparse.ArgumentParser(description="Get executor data for train and evaluation task")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="path to directory containing model weights and config file"
    )
    parser.add_argument(
        "--trust_remote_code", type=bool, default=True
    )
    parser.add_argument(
        "--n_shot", type=int, default=0, help="number of shots, default to zero-shot"
    )
    parser.add_argument(
        "--cot", action="store_true", default=False,
    )
    parser.add_argument(
        "--early_stopping", type=bool, default=True
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256, help="number of shots, default to zero-shot"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95
    )
    parser.add_argument(
        "--n", type=int, default=1
    )
    parser.add_argument(
        "--batch_size", type=int, default=10240
    )
    parser.add_argument(
        "--input", type=str, help="Path of input data", required=True
    )
    parser.add_argument(
        "--output", type=str, help="Path of output data", required=True
    )
    parser.add_argument(
        "--refine", type=str, help="Path of output data"
    )
    parser.add_argument(
        "--template", type=str, help="Template to prompt", required=True
    )
    parser.add_argument(
        '--use_chat_template', action="store_true", help="Enable the use of chat template"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initializing training."
    )
    parser.add_argument(
        "--option_map", type=int, default=0, help="option mapping."
    )
    parser.add_argument(
        "--enhance", action="store_true"
    )
    parser.add_argument(
        "--delete", action="store_true"
    )
    parser.add_argument(
        "--prob", action="store_true"
    )
    parser.add_argument(
        "--strategy", type=str, default=None, help='prompting strategy'
    )
    args = parser.parse_args()
    return args


def construct_chat(user_query: str, system_prompt: str=''):
    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_query}
    ]


def truncate_data(data_str: str, tokenizer: AutoTokenizer, max_len: int = 2500):
    max_len = min(max_len, 4096)
    data_tokens = tokenizer.encode(data_str)
    is_trunc = len(data_tokens) > max_len
    if is_trunc:
        # print(f'Data too long {len(data_tokens)}, truncating into {max_len}')
        data_tokens = data_tokens[:max_len]
        data_str = tokenizer.decode(data_tokens) + '\n...'
    return data_str, is_trunc

if __name__ == '__main__':
    args = parse_args()

    # set seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    use_openai = 'gpt' in args.model_name_or_path
    # enable the flashinfer backend for gemma-2
    if not use_openai and 'gemma-2' in args.model_name_or_path:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

    # # login hf
    # from huggingface_hub import login
    # access_token_read = "your_token"
    # login(token=access_token_read)

    # load data
    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    # run
    template = TEMPLATE_MAP[args.template]
    cot_str = '_cot' if 'cot' in args.template else '_'
    model_name_sanitized = args.model_name_or_path.split('/')[-1].replace('-', '_')
    output_tag = f"{args.n_shot}_shot{cot_str}{model_name_sanitized}"
    data_idx = list(data.keys())
    if output_tag not in data[data_idx[0]]:
        print(f"===================\n\n*** Data-{output_tag} start running...")
        print(f"Loading model {args.model_name_or_path}")
        if args.enhance:
            enhance_df = pd.read_csv('data/combine_enhance_v3.6.csv')
        if args.template == 'refine':
            with open(args.refine, 'rb') as f:
                data_refine = pickle.load(f)

        if use_openai:
            tokenizer = tiktoken.encoding_for_model(args.model_name_or_path)
        else:
            model = LLM(
                model=args.model_name_or_path,
                trust_remote_code=args.trust_remote_code,  # mandatory for hf models
                tensor_parallel_size=torch.cuda.device_count(),
            )
            max_len = model.llm_engine.model_config.max_model_len
            tokenizer = model.get_tokenizer()
            sampling_params = SamplingParams(
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                min_tokens=2,
                n=args.n,
                logprobs=1 if args.prob else None,
            )

        lst_batch = []
        lst_batch_ids = []
        num_trunc = 0
        for idx, id_ in enumerate(tqdm(data.keys(), desc="Preprocess prompt", mininterval=10.)):
            question = data[id_]['QUESTION']
            challenge = data[id_]['CHALLENGE']

            prompt_args = {'question': question}
            # load example
            if args.n_shot > 0:
                answer_map = {v: k.split('_')[-1].upper() for k, v in OPTION_MAP[args.option_map].items()}
                prompt_args['examples'] = generate_few_shot_examples(args.n_shot, cot='cot' in args.template, answer_map=answer_map)
            # load evidence
            if 'wo_data' not in args.template:
                envidence_key = 'NEW_EVIDENCE' if args.delete else 'EVIDENCE'
                stripped_list = [ele.strip() for ele in data[id_][envidence_key]]
                data[id_]["len_evidence"] = len(stripped_list)
                evidence = '\n'.join(stripped_list)
                prompt_args['data'], is_trunc = truncate_data(evidence, tokenizer, max_len=2500)
            # load enhanced text
            if args.enhance:
                prompt_args['data'] += f"\n\nReference prompt: \n{enhance_df[enhance_df['index']==id_]['label'].tolist()[0]}\n"
                num_trunc += 1 if is_trunc else 0
            if args.template == 'refine':
                prompt_args['answer'] =  data_refine[id_][output_tag]

            # generate prompt
            prompt = template.format(**prompt_args, **OPTION_MAP[args.option_map])
            if not use_openai and args.use_chat_template and tokenizer.chat_template is not None:
                chat_prompt = construct_chat(prompt)
                if 'gemma-2' in args.model_name_or_path:
                    # gemma not support sys role
                    chat_prompt = chat_prompt[1:]
                prompt = tokenizer.apply_chat_template(
                    chat_prompt, add_generation_prompt=True, tokenize=False)

            lst_batch.append(prompt)
            lst_batch_ids.append(id_)
            if len(lst_batch) < args.batch_size and idx < len(data) - 1:
                continue

            print(f"truncating {num_trunc} data.")
            # inference
            if use_openai:
                client = get_client()
                if args.prob:
                    args.logprobs = 1
                    args.temperature = 1.0
                    all_outputs = asyncio.run(generate(args, client, '', lst_batch, return_raw=True))
                    lst_outputs = []
                    lst_logprobs = []
                    for idx, out in enumerate(all_outputs):
                        is_match = False
                        logprob_item = out.choices[0].logprobs.content[0]
                        lst_logprobs.append(logprob_item.logprob)
                        text = logprob_item.token
                        lst_outputs.append(text)
                elif args.strategy == 'self-consistent':
                    lst_outputs = asyncio.run(generate(args, client, '', lst_batch, return_raw=True))
                    lst_outputs = [[choice.message.content for choice in out.choices] for out in lst_outputs]
                else:
                    lst_outputs = asyncio.run(generate(args, client, '', lst_batch))
            else:
                preds = model.generate(lst_batch, sampling_params=sampling_params)
                if args.prob:
                    logprob_item = [list(p.outputs[0].logprobs[0].values())[0] for p in preds]
                    lst_outputs = [i.decoded_token for i in logprob_item]
                    lst_logprobs = [i.logprob for i in logprob_item]
                elif args.strategy == 'self-consistent':
                    lst_outputs = [[o.text for o in p.outputs] for p in preds]
                else:
                    lst_outputs = [output.text for p in preds for output in p.outputs[:1]]

            if args.strategy == 'self-consistent':
                refined_outputs = []
                for out in lst_outputs:
                    const_out = max(out, key=out.count)
                    refined_outputs.append(const_out)
                lst_outputs = refined_outputs

            # reload the data to avoid missing any updates
            with open(args.input, 'rb') as f:
                data = pickle.load(f)
            for id_, out in zip(lst_batch_ids, lst_outputs):
                data[id_][output_tag] = out
                if args.prob:
                    data[id_][f"{output_tag}_logprob"] = lst_logprobs[id_]

        print('-----------------THE END----------------------')

        # save data
        with open(args.output, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Data-{output_tag} saved to {args.output}")
    else:
        print(f"Data-{output_tag} already exists in the input data, skip running")
