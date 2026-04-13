import torch
from . import CLIPAD
from torch.nn import functional as F
from .ad_prompts import *
from PIL import Image

valid_backbones = ['ViT-B-16-plus-240']
valid_pretrained_datasets = ['laion400m_e32']

from torchvision import transforms

mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]

def _convert_to_rgb(image):
    return image.convert('RGB')

class WinClipAD(torch.nn.Module):
    def __init__(self, out_size_h, out_size_w, device, backbone, pretrained_dataset, scales, precision='fp32', **kwargs):
        '''

        :param out_size_h:
        :param out_size_w:
        :param device:
        :param backbone:
        :param pretrained_dataset:
        '''
        super(WinClipAD, self).__init__()

        self.out_size_h = out_size_h
        self.out_size_w = out_size_w
        self.precision =  'fp16' #precision  -40% GPU memory (2.8G->1.6G) with slight performance drop 

        self.device = device
        self.get_model(backbone, pretrained_dataset, scales)
        self.phrase_form = '{}'

        # version v1: no norm for each of linguistic embedding
        # version v1:    norm for each of linguistic embedding
        self.version = 'V1' # V1:
        # visual textual, textual_visual
        self.fusion_version = 'textual_visual'

        self.transform = transforms.Compose([
            transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.BICUBIC),
            transforms.CenterCrop(kwargs['img_cropsize']),
            _convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train, std=std_train)])

        self.gt_transform = transforms.Compose([
            transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.NEAREST),
            transforms.CenterCrop(kwargs['img_cropsize']),
            transforms.ToTensor()])

        print(f'fusion version: {self.fusion_version}')

    def get_model(self, backbone, pretrained_dataset, scales):

        assert backbone in valid_backbones
        assert pretrained_dataset in valid_pretrained_datasets

        model, _, _ = CLIPAD.create_model_and_transforms(model_name=backbone, pretrained=pretrained_dataset, scales=scales, precision = self.precision)
        tokenizer = CLIPAD.get_tokenizer(backbone)
        model.eval().to(self.device)

        self.masks = model.visual.masks
        self.scale_begin_indx = model.visual.scale_begin_indx
        self.model = model
        self.tokenizer = tokenizer
        self.normal_text_features = None
        self.abnormal_text_features = None
        self.grid_size = model.visual.grid_size
        self.visual_gallery = None
        print("self.grid_size",self.grid_size)

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):

        if self.precision == "fp16":
            image = image.half()
        image_features = self.model.encode_image(image)
        return [f / f.norm(dim=-1, keepdim=True) for f in image_features]

    @torch.no_grad()
    def encode_text(self, text: torch.Tensor):
        text_features = self.model.encode_text(text)
        return text_features

    def build_text_feature_gallery(self, category: str):
        normal_phrases = []
        abnormal_phrases = []

        for template_prompt in template_level_prompts:
            for normal_prompt in state_level_normal_prompts:
                phrase = template_prompt.format(normal_prompt.format(category))
                normal_phrases.append(phrase)

            for abnormal_prompt in state_level_abnormal_prompts:
                phrase = template_prompt.format(abnormal_prompt.format(category))
                abnormal_phrases.append(phrase)

        self.normal_phrases = normal_phrases
        self.abnormal_phrases = abnormal_phrases

        normal_tokens = self.tokenizer(normal_phrases).to(self.device)
        abnormal_tokens = self.tokenizer(abnormal_phrases).to(self.device)

        if self.version == "V1":
            normal_text_features = self.encode_text(normal_tokens)
            abnormal_text_features = self.encode_text(abnormal_tokens)
        elif self.version == "V2":
            normal_text_features = []
            for phrase_id in range(normal_tokens.size()[0]):
                normal_text_feature = self.encode_text(normal_tokens[phrase_id].unsqueeze(0))
                normal_text_feature = normal_text_feature / normal_text_feature.norm(dim=-1, keepdim=True)
                normal_text_features.append(normal_text_feature)
            normal_text_features = torch.cat(normal_text_features, 0).half()

            abnormal_text_features = []
            for phrase_id in range(abnormal_tokens.size()[0]):
                abnormal_text_feature = self.encode_text(abnormal_tokens[phrase_id].unsqueeze(0))
                abnormal_text_feature = abnormal_text_feature / abnormal_text_feature.norm(dim=-1, keepdim=True)
                abnormal_text_features.append(abnormal_text_feature)
            abnormal_text_features = torch.cat(abnormal_text_features, 0).half()
        else:
            raise NotImplementedError

        # store individual prompt embeddings
        self.normal_text_features = normal_text_features / normal_text_features.norm(dim=-1, keepdim=True)
        self.abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)

        avr_normal_text_features = torch.mean(self.normal_text_features, dim=0, keepdim=True)
        avr_abnormal_text_features = torch.mean(self.abnormal_text_features, dim=0, keepdim=True)

        self.avr_normal_text_features = avr_normal_text_features
        self.avr_abnormal_text_features = avr_abnormal_text_features
        self.text_features = torch.cat(
            [self.avr_normal_text_features, self.avr_abnormal_text_features], dim=0
        )
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def build_image_feature_gallery(self, normal_images):

        self.visual_gallery = []
        visual_features = self.encode_image(normal_images)

        for scale_index in range(len(self.scale_begin_indx)):
            if scale_index == len(self.scale_begin_indx) - 1:
                scale_features = visual_features[self.scale_begin_indx[scale_index]:]
            else:
                scale_features = visual_features[self.scale_begin_indx[scale_index]:self.scale_begin_indx[scale_index+1]]

            self.visual_gallery += [torch.cat(scale_features, dim=0)]


    def calculate_textual_anomaly_score(self, visual_features):
        N = visual_features[0].shape[0]
        scale_anomaly_scores = []
        token_anomaly_scores = torch.zeros((N,self.grid_size[0] * self.grid_size[1]))
        token_weights = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
        for indx, (features, mask) in enumerate(zip(visual_features, self.masks)):
            normality_and_abnormality_score = (100.0 * features @ self.text_features.T).softmax(dim=-1)
            normality_score = normality_and_abnormality_score[:, 0]
            abnormality_score = normality_and_abnormality_score[:, 1]
            normality_score = normality_score.cpu()

            mask = mask.reshape(-1)
            cur_token_anomaly_score = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
            if self.precision == "fp16":
                cur_token_anomaly_score = cur_token_anomaly_score.half()
            cur_token_anomaly_score[:, mask] = (1. / normality_score).unsqueeze(1)
            # cur_token_anomaly_score[:, mask] = (1. - normality_score).unsqueeze(1)
            cur_token_weight = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
            cur_token_weight[:, mask] = 1.

            if indx in self.scale_begin_indx[1:]:
                # deal with the first two scales
                token_anomaly_scores = token_anomaly_scores / token_weights
                scale_anomaly_scores.append(token_anomaly_scores)

                # another scale, calculate from scratch
                token_anomaly_scores = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
                token_weights = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))

            token_weights += cur_token_weight
            token_anomaly_scores += cur_token_anomaly_score

        # deal with the last one
        token_anomaly_scores = token_anomaly_scores / token_weights
        scale_anomaly_scores.append(token_anomaly_scores)

        scale_anomaly_scores = torch.stack(scale_anomaly_scores, dim=0)
        scale_anomaly_scores = torch.mean(scale_anomaly_scores, dim=0)
        scale_anomaly_scores = 1. - 1. / scale_anomaly_scores

        anomaly_map = scale_anomaly_scores.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)
        return anomaly_map

    def calculate_visual_anomaly_score(self, visual_features):
        N = visual_features[0].shape[0]
        scale_anomaly_scores = []
        token_anomaly_scores = torch.zeros((N,self.grid_size[0] * self.grid_size[1]))
        token_weights = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))

        cur_scale_indx = 0
        cur_visual_gallery = self.visual_gallery[cur_scale_indx]

        for indx, (features, mask) in enumerate(zip(visual_features, self.masks)):
            normality_score = 0.5 * (1 - (features @ cur_visual_gallery.T).max(dim=1)[0])
            normality_score = normality_score.cpu()

            mask = mask.reshape(-1)
            cur_token_anomaly_score = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
            if self.precision == "fp16":
                cur_token_anomaly_score = cur_token_anomaly_score.half()
            cur_token_anomaly_score[:, mask] = normality_score.unsqueeze(1)
            # cur_token_anomaly_score[:, mask] = (1. - normality_score).unsqueeze(1)
            cur_token_weight = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
            cur_token_weight[:, mask] = 1.

            if indx in self.scale_begin_indx[1:]:
                cur_scale_indx += 1
                cur_visual_gallery = self.visual_gallery[cur_scale_indx]
                # deal with the first two scales
                token_anomaly_scores = token_anomaly_scores / token_weights
                scale_anomaly_scores.append(token_anomaly_scores)

                # another scale, calculate from scratch
                token_anomaly_scores = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))
                token_weights = torch.zeros((N, self.grid_size[0] * self.grid_size[1]))

            token_weights += cur_token_weight
            token_anomaly_scores += cur_token_anomaly_score

        # deal with the last one
        token_anomaly_scores = token_anomaly_scores / token_weights
        scale_anomaly_scores.append(token_anomaly_scores)

        scale_anomaly_scores = torch.stack(scale_anomaly_scores, dim=0)
        scale_anomaly_scores = torch.mean(scale_anomaly_scores, dim=0)

        anomaly_map = scale_anomaly_scores.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)
        return anomaly_map

    @torch.no_grad()
    def calculate_zero_shot_image_score(self, visual_features):
        last_features = visual_features[-1]  # expected shape: [N, D]

        if last_features.ndim != 2:
            raise ValueError(f"Unexpected last feature shape: {last_features.shape}")

        probs = (100.0 * last_features @ self.text_features.T).softmax(dim=-1)
        anomaly_score = probs[:, 1].cpu()

        return anomaly_score

    def forward(self, images, return_details=False):
        visual_features = self.encode_image(images)
        zero_shot_score = self.calculate_zero_shot_image_score(visual_features)
        textual_anomaly_map = self.calculate_textual_anomaly_score(visual_features)

        if self.visual_gallery is not None:
            visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)
        else:
            visual_anomaly_map = textual_anomaly_map

        if self.fusion_version == 'visual':
            fused_anomaly_map = visual_anomaly_map
        elif self.fusion_version == 'textual':
            fused_anomaly_map = textual_anomaly_map
        else:
            fused_anomaly_map = 1. / (1. / textual_anomaly_map + 1. / visual_anomaly_map)

        # upsample all maps to output resolution
        textual_anomaly_map = F.interpolate(
            textual_anomaly_map,
            size=(self.out_size_h, self.out_size_w),
            mode='bilinear',
            align_corners=False
        )
        visual_anomaly_map = F.interpolate(
            visual_anomaly_map,
            size=(self.out_size_h, self.out_size_w),
            mode='bilinear',
            align_corners=False
        )
        fused_anomaly_map = F.interpolate(
            fused_anomaly_map,
            size=(self.out_size_h, self.out_size_w),
            mode='bilinear',
            align_corners=False
        )

        textual_np = textual_anomaly_map.squeeze(1).detach().cpu().numpy()
        visual_np = visual_anomaly_map.squeeze(1).detach().cpu().numpy()
        fused_np = fused_anomaly_map.squeeze(1).detach().cpu().numpy()

        textual_list = [textual_np[i] for i in range(textual_np.shape[0])]
        visual_list = [visual_np[i] for i in range(visual_np.shape[0])]
        fused_list = [fused_np[i] for i in range(fused_np.shape[0])]

        # image-level score from visual/reference map
        visual_max_score = visual_anomaly_map.flatten(1).max(dim=1)[0]

        # ascore_W(x) = 0.5 * (ascore0(f(x)) + max_ij M^W_ij)
        # in zero-shot, use ascore0 directly
        if self.visual_gallery is not None:
            fused_image_score = 0.5 * (zero_shot_score + visual_max_score)
        else:
            fused_image_score = zero_shot_score

        if not return_details:
            return fused_list

        return {
            "textual_map": textual_list,
            "visual_map": visual_list,
            "fused_map": fused_list,
            "zero_shot_score": zero_shot_score.detach().cpu().numpy(),
            "visual_max_score": visual_max_score.detach().cpu().numpy(),
            "fused_image_score": fused_image_score.detach().cpu().numpy(),
        }

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    @torch.no_grad()
    def rank_prompts_for_image(self, images, top_k=5):
        visual_features = self.encode_image(images)

        # use the last scale as a simple image-level summary
        last_features = visual_features[-1]  # shape: [N, D]
        if last_features.ndim != 2:
            raise ValueError(f"Unexpected feature shape: {last_features.shape}")

        normal_scores = (100.0 * last_features @ self.normal_text_features.T).softmax(dim=-1)
        abnormal_scores = (100.0 * last_features @ self.abnormal_text_features.T).softmax(dim=-1)

        results = []
        for i in range(last_features.shape[0]):
            n_scores = normal_scores[i]
            a_scores = abnormal_scores[i]

            top_n_vals, top_n_idx = torch.topk(n_scores, k=min(top_k, n_scores.shape[0]))
            top_a_vals, top_a_idx = torch.topk(a_scores, k=min(top_k, a_scores.shape[0]))

            results.append({
                "top_normal_prompts": [
                    (self.normal_phrases[idx], float(val))
                    for val, idx in zip(top_n_vals.cpu(), top_n_idx.cpu())
                ],
                "top_abnormal_prompts": [
                    (self.abnormal_phrases[idx], float(val))
                    for val, idx in zip(top_a_vals.cpu(), top_a_idx.cpu())
                ],
            })

        return results