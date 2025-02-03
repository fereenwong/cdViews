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
from qa_utils import load_and_update, get_scanqa, get_sqa, custom_collate_fn
from dataset import ViewLabelDataset
from torch.utils.data import DataLoader
from ViewSelector import ViewSelector, BCELoss
from torch.nn.utils.rnn import pad_sequence


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


def train_epoch(pair_dataloader, tokenizer, model, view_selector, optimizer, epoch,
                save_path):
    view_selector.train()
    criterion = BCELoss()
    for iter, (_, qs_list, image_embeds, labels, image_file_list) in enumerate(tqdm(pair_dataloader)):
        text_embeds_list = []
        for qs, image_files in zip(qs_list, image_file_list):
            line = {'from': 'human',
                    'value': qs}
            input_ids = preprocess_qwen([line, {'from': 'gpt', 'value': None}], tokenizer, has_image=False).cuda()
            text_embed = model.get_model().embed_tokens(input_ids)
            text_embeds_list += [text_embed.squeeze(0), ] * len(image_files)

        padded_text_embeds = pad_sequence(text_embeds_list, batch_first=True)

        text_embeds, image_embeds = view_selector(image_embeds.float().cuda(), padded_text_embeds.float())

        loss = criterion(text_embeds, image_embeds, labels.cuda())
        print('epoch: {} -- iteration: {} -- loss: {}'.format(epoch, iter, loss))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.save({'epoch': epoch,
                'model': view_selector.state_dict(),
                'optimizer': optimizer.state_dict()}, save_path)


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.LVLM_ckpt)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    # view selector model
    view_selector = ViewSelector().to(model.device)

    # Data
    print("Training the view selector... ")
    pair_dataset_train = ViewLabelDataset(args, mode='train')
    pair_dataloader_train = DataLoader(pair_dataset_train, batch_size=args.batch_size, shuffle=True,
                                       collate_fn=custom_collate_fn)

    learning_rate, num_epoches = 0.00005, 10
    optimizer = torch.optim.Adam(view_selector.parameters(), lr=learning_rate, weight_decay=1e-5)

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    for epoch in range(num_epoches):
        save_ckpt_path = os.path.join(args.work_dir, 'model_epoch_{}.pth'.format(epoch + 1))
        train_epoch(pair_dataloader_train, tokenizer, model, view_selector,
                        optimizer, epoch, save_path=save_ckpt_path)


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
