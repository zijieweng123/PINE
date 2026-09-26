import os
import argparse
import numpy as np
import torch
from torchvision import transforms
from pyiqa.archs.pine_arch import UFANet
from PIL import Image
import random

random_seed = 215
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Authentic Image Quality Assessment")
    parser.add_argument('-p', '--model_path', default='ckpt', help='Path of model snapshot.', type=str)
    parser.add_argument('-f', '--trained_model_file', default='PINE_0.9006.pth', type=str)
    parser.add_argument('-m', '--model', default='UFA', type=str)
    parser.add_argument('--n_fragment', type=int, default=12)
    parser.add_argument('-i', '--image_path', required=True, type=str, help="Path to the input image")
    parser.add_argument('--gpu_ids', type=list, default=None)
    args = parser.parse_args()
    return args
    
def get_spatial_fragments(
    video,
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
    nfrags=1,
    random=False,
    random_upsample=False,
    fallback_type="upsample",
    **kwargs,
):
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    if fallback_type == "upsample" and ratio < 1:
        
        ovideo = video
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=1 / ratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)
        
    if random_upsample:
        randratio = random.random() * 0.5 + 1
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=randratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)

    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    hgrids = torch.LongTensor(
        [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    )
    wgrids = torch.LongTensor(
        [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    )
    hlength, wlength = res_h // fragments_h, res_w // fragments_w

    if random:
        if res_h > fsize_h:
            rnd_h = torch.randint(
                res_h - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if res_w > fsize_w:
            rnd_w = torch.randint(
                res_w - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    else:
        if hlength > fsize_h:
            rnd_h = torch.randint(
                hlength - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if wlength > fsize_w:
            rnd_w = torch.randint(
                wlength - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()

    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)

    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t + 1) * aligned
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                if random:
                    h_so, h_eo = rnd_h[i][j][t], rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = rnd_w[i][j][t], rnd_w[i][j][t] + fsize_w
                else:
                    h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize_w
                target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[
                    :, t_s:t_e, h_so:h_eo, w_so:w_eo
                ]
    return target_video

if __name__ == '__main__':

    args = parse_args()
    model_path = args.model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_file = args.trained_model_file

    if args.model == 'UFA':
        model = UFANet()

    model = model.to(device)
    state_dict = torch.load(os.path.join(model_path, model_file), map_location=device)

    if "params" in state_dict:
        model.load_state_dict(state_dict["params"])
    else:
        model.load_state_dict(state_dict)
    
    model.eval()
    
    transform_distortion_preprocessing = transforms.Compose([transforms.ToTensor()])

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        exit(1)

    test_image = Image.open(args.image_path)
    test_image = test_image.convert('RGB')

    w, h = test_image.size
    
    if w <= 384 or h <= 384:
        test_image_distortion = transform_distortion_preprocessing(test_image).unsqueeze(0)
    else:
        test_image_distortion = transform_distortion_preprocessing(test_image)
        test_image_distortion = test_image_distortion.unsqueeze(1)
        test_image_distortion = get_spatial_fragments(
            test_image_distortion,
            fragments_h=args.n_fragment,
            fragments_w=args.n_fragment,
            fsize_h=32,
            fsize_w=32,
            aligned=32,
            nfrags=1,
            random=False,
            random_upsample=False,
            fallback_type="upsample"
        )
        test_image_distortion = test_image_distortion.squeeze(1)
        test_image_distortion = test_image_distortion.unsqueeze(0)

    with torch.no_grad():
        test_image_distortion = test_image_distortion.to(device)
        outputs, _ = model(test_image_distortion)
        score = outputs.item()
        print(f'Quality Score: {score:.4f}')
