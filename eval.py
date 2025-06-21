import json
import os

import torch
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
from diffusers import StableDiffusionPipeline
from train_classifier_residual_reweight_next_pred_serial_module import MultiTaskClassifierImproved
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
import random, string


###########################################
# 辅助函数 —— 根据 prompt_type 获取条件分支 token embedding
###########################################
def prompt_type2embed(prompt_type, prompt_embeds, prompt, pipe):
    """
    根据 prompt_type 选择对应的 token embedding。
    参数 prompt_embeds 为 pipe.encode_prompt 得到的完整嵌入，形状为 [1, n_tokens, embed_dim]。

    prompt_type 的取值说明：
      'no': 返回全零向量；
      's': 仅使用首 token；
      't': 使用中间文本 token（不含首尾）；
      'e': 仅使用尾 token；
      'p': 使用 padding 部分；
      'st': 拼接首 token 和中间 token；
      'se': 拼接首 token 和尾 token；
      'sp': 拼接首 token 和 padding；
      'te': 拼接中间 token 和尾 token；
      'tp': 拼接中间 token 和 padding；
      'ep': 拼接尾 token 和 padding；
      's_last': 拼接首 token 和最后一个 padding token；
      对于特殊形式如 "st-1", "st-2", … 表示首 token 拼接中间 token 前 N 个；
      'ste': 拼接首、中间、尾 token；
      'step': 使用完整嵌入。
    """
    text_inputs = pipe.tokenizer(
        prompt, padding='max_length', max_length=77, truncation=True, return_tensors='pt'
    )
    seq_length = torch.sum(text_inputs.data['attention_mask']).item()

    prompt_start = prompt_embeds[:, 0:1, :]
    prompt_text = prompt_embeds[:, 1:seq_length - 1, :]
    prompt_end = prompt_embeds[:, seq_length - 1:seq_length, :]
    prompt_padding = prompt_embeds[:, seq_length:, :]

    if prompt_type == 'no':
        return torch.zeros_like(prompt_start)
    elif prompt_type == 's':
        return prompt_start
    elif prompt_type == 't':
        return prompt_text
    elif prompt_type == 'e':
        return prompt_end
    elif prompt_type == 'p':
        return prompt_padding
    elif prompt_type == 'st':
        return torch.cat([prompt_start, prompt_text], dim=1)
    elif prompt_type == 'se':
        return torch.cat([prompt_start, prompt_end], dim=1)
    elif prompt_type == 'sp':
        return torch.cat([prompt_start, prompt_padding], dim=1)
    elif prompt_type == 'te':
        return torch.cat([prompt_text, prompt_end], dim=1)
    elif prompt_type == 'tp':
        return torch.cat([prompt_text, prompt_padding], dim=1)
    elif prompt_type == 'ep':
        return torch.cat([prompt_end, prompt_padding], dim=1)
    elif prompt_type == 's_last':
        return torch.cat([prompt_start, prompt_padding[:, -1:, :]], dim=1)
    elif prompt_type.startswith("st-"):
        try:
            n_token = int(prompt_type.split("-")[1])
            return torch.cat([prompt_start, prompt_text[:, :n_token, :]], dim=1)
        except Exception as e:
            raise ValueError(f"解析 prompt_type {prompt_type} 失败: {e}")
    elif prompt_type == 'ste':
        return torch.cat([prompt_start, prompt_text, prompt_end], dim=1)
    elif prompt_type == 'step':
        return prompt_embeds
    else:
        raise ValueError(f"Unsupported prompt type: {prompt_type}")


###########################################
# 辅助函数 —— 根据 empty_type 获取非条件（空白）分支 token embedding
###########################################
def empty_type2embed(empty_type, empty_embeds):
    """
    根据 empty_type 选择对应的空白 token embedding。
    empty_embeds 由 pipe.encode_prompt 得到，形状为 [1, n_tokens, embed_dim]。

    empty_type 的取值：
      'no': 返回全零嵌入；
      's': 使用第一个 token；
      'e': 使用第二个 token；
      'p': 使用剩余区域；
      'se': 拼接前两个 token；
      'sp': 拼接第一个 token 与后续 token；
      'ep': 拼接第二个 token 与后续 token；
      's_last': 拼接第一个 token 和最后一个 token；
      'sep': 拼接全部 token（前两个和后续部分）。
    """
    prompt_start = empty_embeds[:, 0:1, :]
    prompt_end = empty_embeds[:, 1:2, :]
    prompt_padding = empty_embeds[:, 2:, :]

    if empty_type == 'no':
        return torch.zeros_like(prompt_start)
    elif empty_type == 's':
        return prompt_start
    elif empty_type == 'e':
        return prompt_end
    elif empty_type == 'p':
        return prompt_padding
    elif empty_type == 'se':
        return torch.cat([prompt_start, prompt_end], dim=1)
    elif empty_type == 'sp':
        return torch.cat([prompt_start, prompt_padding], dim=1)
    elif empty_type == 'ep':
        return torch.cat([prompt_end, prompt_padding], dim=1)
    elif empty_type == 's_last':
        return torch.cat([prompt_start, prompt_padding[:, -1:, :]], dim=1)
    elif empty_type == 'sep':
        return torch.cat([prompt_start, prompt_end, prompt_padding], dim=1)
    else:
        raise ValueError(f"Unsupported empty type: {empty_type}")



exp_labels_map = [
    "no_no",
    "s_no",
    "s_s",
    "s_se",
    "s_sep",
    "se_no",
    "se_s",
    "se_se",
    "se_sep",
    "st_s",
    "st_se",
    "st_sep",
    "st-1_sep",
    "st-2_sep",
    "st-3_s",
    "st-3_sep",
    "st-4_s",
    "st-4_se",
    "st-4_sep",
    "ste_s",
    "ste_se",
    "ste_sep",
    "step_no",
    "step_s",
    "step_se",
    "step_sep"
]


def generate_image_name(use_timestamp=False, random_length=6, extension="jpg"):
    """
    生成随机图片文件名
    参数：
        use_timestamp (bool): 是否包含时间戳（默认包含）
        random_length (int): 随机字符串长度（默认6位）
        extension (str): 文件扩展名（默认"jpg"）
    返回：
        str: 类似 "IMG_20231022_8fT3xv.jpg" 的文件名
    """
    # 处理扩展名格式
    if not extension.startswith("."):
        extension = f".{extension.lower()}"

    # 生成随机字符串（字母+数字）
    characters = string.ascii_letters + string.digits
    random_str = ''.join(random.choice(characters) for _ in range(random_length))

    # 组合最终文件名
    return f"IMG{random_str}{extension}"


def save_latent(latent, save_path, vae):
    with torch.no_grad():
        image_tensor = vae.decode(latent / vae.config.scaling_factor).sample

    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_tensor = image_tensor.cpu().permute(0, 2, 3, 1).numpy()
    image = (image_tensor * 255).astype(np.uint8)

    pil_image = Image.fromarray(image[0])
    pil_image.save(save_path)


def predict_for_prompt(classifier_model, prompt, timesteps, device):
    """
    对给定 prompt 及 51 个 timestep 进行预测，
    返回 active_flag、exp、next_exp 三个分支（均为 numpy 数组）。
    """
    prompt_list = [prompt] * len(timesteps)
    if isinstance(timesteps, torch.Tensor):
        timesteps_tensor = timesteps.clone().detach().float().to(device)
    else:
        timesteps_tensor = torch.tensor(timesteps, dtype=torch.float, device=device)
    with torch.no_grad():
        active_logits, exp_logits, next_exp_logits = classifier_model(prompt_list, timesteps_tensor)
    active_preds = active_logits.argmax(dim=1).cpu().numpy()
    exp_preds = exp_logits.argmax(dim=1).cpu().numpy()
    next_exp_preds = next_exp_logits.argmax(dim=1).cpu().numpy()
    return active_preds, exp_preds, next_exp_preds



class ModelEvaluator:
    def __init__(
            self,
            base_model: torch.nn.Module,
            base_version: str,
            classifier: Optional[torch.nn.Module] = None,
            input_shape: Tuple[int, ...] = (3, 512, 512),
            device: str = "cuda",
            image_save_path = None,
            cfg: float = 7.0,
            num_timesteps: int = 50
    ):
        self.base_model = base_model.to(device)
        self.base_version = base_version
        self.classifier = classifier.to(device)
        self.classifier.eval()
        self.device = device
        self.input_shape = input_shape
        self.image_save_path = image_save_path
        self.cfg = cfg
        self.num_timsteps = num_timesteps

        # 预计算基础模型FLOPs
        self.base_macs = self.calculate_MACs(self.base_model.unet)  # 示例SD模型UNet输入
        self.dict_macs = self.init_dict_macs()


    def calculate_MACs(self, model):
        from thop import profile

        latent = torch.randn(1, 4, 64, 64).to(self.device)

        timestep = torch.tensor([0], dtype=torch.long).to(self.device)

        # sd 1系列是768
        # sd 2系列是1024
        if '2' in self.base_version:
            text_embeddings = torch.randn(1, 77, 1024).to(self.device)
        elif '1' in self.base_version:
            text_embeddings = torch.randn(1, 77, 768).to(self.device)

        macs, _ = profile(
            model,
            inputs=(latent, timestep, text_embeddings),
            verbose=False
        )
        return macs

    def init_dict_macs(self):
        from thop import profile

        latent = torch.randn(1, 4, 64, 64).to(self.device)

        timestep = torch.tensor([0], dtype=torch.long).to(self.device)

        mac_dict = {}


        for i in range(77):
            if '2' in self.base_version:
                text_embeddings = torch.randn(1, i + 1, 1024).to(self.device)
            elif '1' in self.base_version:
                text_embeddings = torch.randn(1, i + 1, 768).to(self.device)

            macs, _ = profile(self.base_model.unet,
                              inputs=(latent, timestep, text_embeddings),
                              verbose=False)
            mac_dict[i+1]=macs
        return mac_dict

    def generate_image(self,
                       text: str,
                       ):
        output_path = self.image_save_path
        cfg = self.cfg
        num_steps = self.num_timsteps

        timestep_list, num_inference_steps = retrieve_timesteps(self.base_model.scheduler, num_steps, self.device, None)
        global_latents = self.base_model.prepare_latents(1, 4, 512, 512, torch.float32, self.device, None, None)

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        texts = [text]

        for text in texts:
            # 随机生成图像名称
            image_name = generate_image_name()
            # 直接生成
            print('----------直接生成----------')
            time1 = time.time()
            prompt_embeds, empty_embeds = self.base_model.encode_prompt(
                text, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=True
            )
            latents = global_latents.clone()
            for step_idx, t in enumerate(timestep_list):
                latent_model_input = self.base_model.scheduler.scale_model_input(latents, t)
                with torch.no_grad():
                    noise_pred_text = self.base_model.unet(
                        latent_model_input, t, encoder_hidden_states=prompt_embeds, return_dict=False
                    )[0]
                    noise_pred_empty = self.base_model.unet(
                        latent_model_input, t, encoder_hidden_states=empty_embeds, return_dict=False
                    )[0]
                    noise_pred = noise_pred_empty + cfg * (noise_pred_text - noise_pred_empty)
                    latents = self.base_model.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            time2 = time.time()
            base_time = time2-time1
            print(time2-time1)

            # 保存直接生成的结果
            save_latent(latents, os.path.join(output_path, 'base_' + image_name), self.base_model.vae)

            # 加速生成
            print('----------加速生成----------')
            time1 = time.time()

            prompt_embeds, empty_embeds = self.base_model.encode_prompt(
                text, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True
            )

            active_preds, exp_preds, _ = predict_for_prompt(self.classifier, text, timestep_list, device)
            active_indices = np.where(active_preds == 1)[0]
            if active_indices.size == 0:
                continue
            last_active_index = active_indices[-1]
            effective_timesteps = timestep_list[: last_active_index + 1]
            # print(f"    有效 diffusion 步数：{len(effective_timesteps)}（最后 active_flag=1 出现在第 {last_active_index} 步）")

            # 为该 prompt 获取条件与空白分支的 token embedding
            # 使用全局 latent 的克隆，保证所有 prompt 起始 latent 一致
            latents = global_latents.clone()

            for step_idx, t in enumerate(effective_timesteps):
                token_idx = exp_preds[step_idx]
                if token_idx < 0 or token_idx >= len(exp_labels_map):
                    # print(f"    timestep {step_idx}: 无效的 exp 预测 index {token_idx}，跳过此次 timestep。")
                    continue
                token_comb = exp_labels_map[token_idx]  # 如 "st_se"
                try:
                    p_type, e_type = token_comb.split("_")
                except Exception as e:
                    print(f"    token_comb 格式错误 {token_comb}: {e}")
                    continue

                # 根据 exp 预测获得相应的 token embedding
                # p_type = 'step'
                # e_type = 'sep'
                current_prompt_embed = prompt_type2embed(p_type, prompt_embeds, text, self.base_model)
                current_empty_embed = empty_type2embed(e_type, empty_embeds)
                # current_prompt_embed = prompt_embeds
                # current_empty_embed = empty_embeds

                # 执行单步 diffusion
                latent_model_input = self.base_model.scheduler.scale_model_input(latents, t)
                with torch.no_grad():
                    noise_pred_text = self.base_model.unet(
                        latent_model_input, t, encoder_hidden_states=current_prompt_embed, return_dict=False
                    )[0]
                    noise_pred_empty = self.base_model.unet(
                        latent_model_input, t, encoder_hidden_states=current_empty_embed, return_dict=False
                    )[0]
                noise_pred = noise_pred_empty + cfg * (noise_pred_text - noise_pred_empty)
                latents = self.base_model.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            time2 = time.time()
            acc_time=time2-time1
            print(time2-time1)

            alpha_prod_t = self.base_model.scheduler.alphas_cumprod[t.item()]
            beta_prod_t = (1 - alpha_prod_t) ** 0.5
            pred_x0 = (latents - beta_prod_t * noise_pred) / alpha_prod_t
            latents = pred_x0

            # 计算MACs
            MACs = 0
            for step_idx, t in enumerate(effective_timesteps):
                token_idx = exp_preds[step_idx]
                token_comb = exp_labels_map[token_idx]
                p_type, e_type = token_comb.split("_")
                current_prompt_embed = prompt_type2embed(p_type, prompt_embeds, text, self.base_model)
                current_empty_embed = empty_type2embed(e_type, empty_embeds)
                prompt_number = current_prompt_embed.shape[1]
                empty_number = current_empty_embed.shape[1]
                MACs = MACs + self.dict_macs[prompt_number] + self.dict_macs[empty_number]

            save_latent(latents, os.path.join(output_path, 'speedup_' + image_name), self.base_model.vae)

        return {'base_filename': 'base_' + image_name,
                'speedup_filename': 'speedup_' + image_name,
                'text_condition': text,
                'base_time': base_time,
                'speedup_time': acc_time,
                'cfg': cfg,
                'num_timesteps': num_steps,
                'base_MACs': self.base_macs * 51 * 2,
                'speedup_MACs': MACs}





if __name__ == "__main__":

    # -----------！！！！加载超参！！！！----------------
    with open('config_template.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 设置随机种子
    SEED = config['random_seed']
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # 初始化模型
    device = config['device']
    sd_model = StableDiffusionPipeline.from_pretrained(config['basemodel_path'])
    classifier = MultiTaskClassifierImproved(
        stable_diff_model=sd_model,
        device=device,
        prompt_feature_dim=1024*77,
        timestep_feature_dim=1280,
        prompt_feature_out=800,
        timestep_feature_out=640,
        fusion_dim=960,
        num_exp_classes=26,
        n_tokens=77
    )
    classifier.load_state_dict(torch.load(config['classifier_path'], map_location=device))
    classifier = classifier.to(device)
    classifier.eval()

    # 创建评估器
    evaluator = ModelEvaluator(
        base_model=sd_model,
        base_version=config['basemodel_version'],
        classifier=classifier,
        input_shape=(3, 512, 512),
        device=device,
        image_save_path=config['image_save_dir'],
        cfg=config['cfg'],
        num_timesteps=config['num_timesteps']
    )

    # 测试数据集
    with open(config['dataset_path'], 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)

    prompt_data = prompt_data[:5]

    result = []

    for data_item in prompt_data:
        data = evaluator.generate_image(data_item['prompt'])
        result.append(data)

    try:
        with open(config['json_save_dir'], 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f'error {str(e)}')


