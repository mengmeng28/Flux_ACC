import json

import torch
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
from diffusers import StableDiffusionPipeline
from train_classifier_residual_reweight_next_pred_serial_module import MultiTaskClassifierImproved
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
import random, string
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision import transforms
import modeling
import os
from huggingface_hub import hf_hub_download

from transformers import CLIPModel, CLIPProcessor, AutoTokenizer
from urllib.request import urlretrieve

import open_clip

class AestheticsScorer:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化所有评分模型
        :param device: 指定运行设备 (cuda/cpu)
        """
        self.device = device
        self._init_PIQA()
        self._init_LAIONAES()

    def _init_PIQA(self):
        """初始化PerceptCLIP_IQA模型"""
        self.PIQA_model = modeling.clip_lora_model().to(self.device)
        # A. load from local file
        self.PIQA_model.load_state_dict(torch.load('/data6/fxm/ckpt/PerceptCLIP/perceptCLIP_IQA.pth', map_location=self.device))
        # B. load online
        # model_path = hf_hub_download(repo_id="PerceptCLIP/PerceptCLIP_IQA", filename="perceptCLIP_IQA.pth")
        # self.PIQA_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.PIQA_model.eval()
        self.PIQA_processor = transforms.Compose(
            [
                transforms.Resize((512, 384)),
                transforms.RandomCrop(size=(224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466,0.4578275,0.40821073),
                                     std=(0.26862954,0.26130258,0.27577711))
            ]
        )

    def _init_LAIONAES(self):
        """初始化LAION-AES模型"""
        # A. load from local file
        self.laionaes_model, _, self.laionaes_preprocess = open_clip.create_model_and_transforms('ViT-L-14',
                                                                                        pretrained='/data6/fxm/ckpt/LAION-AES/ViT-L-14.pt',
                                                                                        cache_dir='/data6/fxm/ckpt/LAION-AES/',
                                                                                        device=self.device)
        self.laionaes_head = torch.nn.Linear(768, 1)
        self.laionaes_head.load_state_dict(torch.load('/data6/fxm/ckpt/LAION-AES/sa_0_4_vit_l_14_linear.pth'))
        self.laionaes_head.eval()
        self.laionaes_head.to(self.device)
        # B. load online
        # self.laionaes_model, _, self.laionaes_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        # head_folder = './aes_head'
        # head_model_path = head_folder + '/sa_0_4_vit_l_14_linear.pth'
        # if not os.path.exists(head_model_path):
        #     os.makedirs(head_folder, exist_ok=True)
        #     url_model = (
        #         "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"
        #     )
        #     urlretrieve(url_model, head_model_path)
        # self.laionaes_head = torch.nn.Linear(768, 1)
        # self.laionaes_head.load_state_dict(torch.load(head_model_path))
        # self.laionaes_head.eval()
        # self.laionaes_head.to(self.device)


    def calculate_PIQA(self, image_path):
        """
        计算PIQA PerceptCLIP_IQA
        """
        image = Image.open(image_path).convert("RGB")
        image = self.PIQA_processor(image).to(self.device).unsqueeze(0)
        with torch.no_grad():
            score = self.PIQA_model(image).cpu().numpy()
        min_pred = -6.52
        max_pred = 3.11

        normalized_score = ((score[0][0] - min_pred) / (max_pred - min_pred))
        return normalized_score

    def calculate_LAIONAES(self, image_path):
        """
        计算LAION-AES
        """
        image = self.laionaes_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.laionaes_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            prediction = self.laionaes_head(image_features)
        return prediction[0][0].item()

    def calculate_all_scores(self, image_path):
        """一次性计算所有分数"""
        return {
            "PIQA": self.calculate_PIQA(image_path),
            "LAION-AES": self.calculate_LAIONAES(image_path),
        }




class AlignmentScorer:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化所有评分模型
        :param device: 指定运行设备 (cuda/cpu)
        """
        self.device = device
        self._init_clip()
        self._init_pickscore()
        self._init_hpsv2()

    def _init_clip(self):
        """初始化CLIP模型"""
        # A. load from local file
        self.clip_score_fn = CLIPScore(model_name_or_path="/data/fxm/ckpt/CLIP-Base/").to(self.device)
        # B. load online
        # self.clip_score_fn = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(self.device)

    def _init_pickscore(self):
        """初始化ImageReward模型"""
        # A. load from local file
        self.pickscore_model = CLIPModel.from_pretrained('/data6/fxm/ckpt/PickScore/').to(self.device)
        self.pickscore_processor = CLIPProcessor.from_pretrained('/data6/fxm/ckpt/PickScore/processor/')
        # B, load online
        # self.pickscore_model = CLIPModel.from_pretrained('yuvalkirstain/PickScore_v1').to(self.device)
        # self.pickscore_processor = CLIPProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

    def _init_hpsv2(self):
        """初始化HPSv2模型"""
        # A. load from local file
        self.hps_model = CLIPModel.from_pretrained('/data/fxm/ckpt/HPSv2/').to(self.device)
        self.hps_processor = CLIPProcessor.from_pretrained('/data/fxm/ckpt/HPSv2/')
        # B. load online
        # self.hps_model = CLIPModel.from_pretrained('adams-story/HPSv2-hf')
        # self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    def calculate_clipscore(self, image_path, prompt):
        """
        计算CLIPScore
        """
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            clip_score = self.clip_score_fn(image, prompt).detach()
        return float(clip_score)

    def calculate_pickscore(self, image_path, prompt):
        """
        计算pickscore分数
        """
        image = Image.open(image_path).convert("RGB")

        image_inputs = self.pickscore_processor(
            images=image,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt',
        ).to(self.device)

        text_inputs = self.pickscore_processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt',
        ).to(self.device)

        with torch.no_grad():
            image_embs = self.pickscore_model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = self.pickscore_model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            scores = self.pickscore_model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

        return scores.item()

    def calculate_hpscore(self, image_path, prompt):
        """
        计算HPSv2分数
        """
        image = Image.open(image_path)

        inputs = self.hps_processor([prompt], images=image, return_tensors='pt', truncation=True, max_length=77, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.hps_model(**inputs)
        image_embeds, text_embeds = outputs['image_embeds'], outputs['text_embeds']
        logits_per_image = image_embeds @ text_embeds.T

        hps_score = logits_per_image.cpu().numpy()

        return float(hps_score[0][0])

    def calculate_all_scores(self, image_path, prompt):
        """一次性计算所有分数"""
        return {
            "CLIPScore": self.calculate_clipscore(image_path, prompt),
            "PickScore": self.calculate_pickscore(image_path, prompt),
            "HPScore": self.calculate_hpscore(image_path, prompt)
        }



if __name__ == "__main__":

    Alignment_Scorer = AlignmentScorer()
    Aesthetics_Scorer = AestheticsScorer()

    # -----------！！！！加载超参！！！！----------------
    with open('config_template.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    device = config['device']
    image_dir = config['image_save_dir']
    json_dir = config['json_save_dir']

    with open(json_dir, 'r', encoding='utf-8') as f:
        data = json.load(f)


    scores = []

    for item in data:
        base_filename = item['base_filename']
        speedup_filename = item['speedup_filename']
        prompt = item['text_condition']

        base_path = os.path.join(image_dir, base_filename)
        speedup_path = os.path.join(image_dir, speedup_filename)

        base_aes_score = Aesthetics_Scorer.calculate_all_scores(base_path)
        base_align_score = Alignment_Scorer.calculate_all_scores(base_path, prompt)

        speedup_aes_score = Aesthetics_Scorer.calculate_all_scores(speedup_path)
        speedup_align_score = Alignment_Scorer.calculate_all_scores(speedup_path, prompt)

        item_score = {'base_filename': item['base_filename'],
                'speedup_filename': item['speedup_filename'],
                'text_condition': item['text_condition'],
                'base_time': item['base_time'],
                'speedup_time': item['speedup_time'],
                'cfg': item['cfg'],
                'num_timesteps': item['num_timesteps'],
                'base_MACs': item['base_MACs'],
                'speedup_MACs': item['speedup_MACs'],
                'base_aes_score': base_aes_score,
                'base_align_score': base_align_score,
                'speedup_aes_score': speedup_aes_score,
                'speedup_align_score': speedup_align_score}
        scores.append(item_score)

    try:
        with open(config['score_save_dir'], 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)
    except Exception as e:
            print(f'error {str(e)}')



        # clip_score = Alignment_Scorer.calculate_clipscore(base_filename, prompt)
        # hpscore = Alignment_Scorer.calculate_hpscore(base_filename, prompt)
        # pickscore = Alignment_Scorer.calculate_pickscore(base_filename, prompt)

