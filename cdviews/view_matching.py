import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re
from qa_utils import load_and_update, get_scanqa, get_sqa


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048,
                    system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    # if roles[source[0]["from"]] != roles["human"]:
    #     source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids


def eval_model(args):

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.LVLM_ckpt)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Data
    print("Auto-labeling all the views as positive/negative/uncertain ... \n"
          "We STRONGLY RECOMMEND directly downloading the prepared files with the link in README.md")

    print("preparing data...")
    if args.dataset == 'ScanQA':
        qa_data = get_scanqa(args, mode='train')  # only labeling the views for the training set
    elif args.dataset == 'SQA':
        qa_data = get_sqa(args, mode='train')

    caption_file = os.path.expanduser(args.caption_file).format(args.dataset)
    assert os.path.exists(caption_file), \
        'the caption file {} does not exist! '.format(caption_file) + \
        'please reformat the QA_pair into captions first, or download it from the provided link (Strongly Recommended)'
    captions_all = json.load(open(caption_file))

    view_labeling_file = os.path.expanduser(args.view_labeling_file).format(args.dataset)
    view_labeling_info = {}

    for iter, line in enumerate(tqdm(qa_data)):
        scene_id = line['scene_id']
        image_files = os.listdir(os.path.join(args.image_folder, scene_id, 'color'))
        question_id = str(line['question_id'])
        captions = captions_all[question_id]
        view_labeling_info[question_id] = {'Positive': [], 'Negative': [], 'Uncertain': []}

        for image_name in image_files:
            line['from'] = 'human'
            line['value'] = '<image>' + 'Caption: {}'.format(captions[0])

            input_ids = preprocess_qwen([line, {'from': 'gpt', 'value': None}], tokenizer,
                                        has_image=True, system_message=args.prompt_M + args.context_M).cuda()

            image_tensors = []
            image = Image.open(os.path.join(args.image_folder, scene_id, 'color', image_name))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_tensors.append(image_tensor.half().cuda())
            image_tensors = torch.cat(image_tensors, dim=0)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensors,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if 'A' in outputs:
                view_labeling_info[question_id]['Positive'].append(image_name)
            elif 'B' in outputs:
                view_labeling_info[question_id]['Negative'].append(image_name)
            elif 'C' in outputs:
                view_labeling_info[question_id]['Uncertain'].append(image_name)
            else:
                print('providing answer not in the options {} / {}'.format(scene_id, question_id))
        # early save to avoid accident corrupt
        if iter % 50 == 0:
            json.dump(view_labeling_info, open(view_labeling_file, 'w'))

    json.dump(view_labeling_info, open(view_labeling_file, 'w'))
    print('Done! Save the labeling results to {}'.format(view_labeling_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--cfg_file", type=str, default="../cfgs/QA.yaml")
    args = parser.parse_args()

    args = load_and_update(args)

    eval_model(args)
