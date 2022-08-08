from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file, jsonify
from threading import Thread
import os
import time
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import glob
import zipfile
import time

import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os.path
from pathlib import Path
import glob
import sys

import pdb

from modules.unet import UNet
from modules.midas.dpt_depth import DPTDepthModel
from data.transforms import get_transform


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'ulorigin'
app.config['task'] = 'depth'
app.config['img_path'] = 'assets/demo/test11.png'
app.config['depth_img_path'] = 'assets/test11_depth.png'
app.config['output_path'] = 'assets/'

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def inference(imgfilename):
    img = cv2.imread(imgfilename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype('uint8')
    img = Image.fromarray(formatted)
    return img

def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img


def save_outputs(img_path, output_file_name, trans_totensor, device, trans_rgb, model, trans_topil):
    with torch.no_grad():
        save_path = os.path.join(app.config['output_path'], f'{output_file_name}_depth.png')

        print(f'Reading input {img_path} ...')
        img = Image.open(img_path)

        img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)

        rgb_path = os.path.join(app.config['output_path'], f'{output_file_name}_rgb.png')
        trans_rgb(img).save(rgb_path)

        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3,1)

        output = model(img_tensor).clamp(min=0, max=1)

        if app.config['task'] == 'depth':
            output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
            output = output.clamp(0,1)
            output = 1 - output
#             output = standardize_depth_map(output)
            plt.imsave(save_path, output.detach().cpu().squeeze(),cmap='viridis')
            
        else:
            trans_topil(output[0]).save(save_path)
            
        print(f'Writing output {save_path} ...')

def demo():
    '''
    parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')

    parser.add_argument('--task', dest='task', help="normal or depth")
    parser.set_defaults(task='depth')

    parser.add_argument('--img_path', dest='img_path', help="path to rgb image")
    parser.set_defaults(im_name='assets/demo/test11.png')

    parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored")
    parser.set_defaults(store_name='assets/')

    args = parser.parse_args()
    '''
    root_dir = 'pretrained_models/'

    print('after parser')

    trans_topil = transforms.ToPILImage()

    os.system(f"mkdir -p {app.config['output_path']}")
    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('before if')
    # get target task and model
    if app.config['task'] == 'normal':
        image_size = 384
        
        pretrained_weights_path = root_dir + 'omnidata_dpt_normal_v2.ckpt'
        model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(device)
        trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                            transforms.CenterCrop(image_size),
                                            get_transform('rgb', image_size=None)])

    elif app.config['task'] == 'depth':
        print('in depth')
        image_size = 384
        pretrained_weights_path = root_dir + 'omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
        # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
        model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(device)
        trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                            transforms.CenterCrop(image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=0.5, std=0.5)])
        print('after in depth')

    else:
        print("task should be one of the following: normal, depth")
        #sys.exit()

    trans_rgb = transforms.Compose([transforms.Resize(512, interpolation=PIL.Image.BILINEAR),
                                    transforms.CenterCrop(512)])

    img_path = Path(app.config['img_path'])
    if img_path.is_file():
        save_outputs(app.config['img_path'], os.path.splitext(os.path.basename(app.config['img_path']))[0], trans_totensor, device, trans_rgb, model, trans_topil)
    elif img_path.is_dir():
        for f in glob.glob(app.config['img_path']+'/*'):
            save_outputs(f, os.path.splitext(os.path.basename(f))[0], trans_totensor, device, trans_rgb, model, trans_topil)
    else:
        print("invalid file path!")
        #sys.exit()

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('depth.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))

@app.route('/depth')
def mypage():
    return render_template('depth.html')

@app.route('/getdepth', methods = ['POST'])
def get_depth():
    print('get_depth')
    def delayed_clear_files(value):
        import time
        time.sleep(value)

        types = ('*.jpg', '*.jpeg', '*.png') # the tuple of file types
        files_grabbed = []
        for files in types:
            files_grabbed.extend(glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], files)))
        
        # print(">>> delayed clear files: ", files_grabbed)
        for filePath in files_grabbed:
            try:
                os.remove(filePath)
            except:
                print("Error while deleting file : ", filePath)

    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp

    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message' : 'No file selected for uploading'})
        resp.status_code = 400
        return resp

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        #start = time.time()
        print('demo')
        demo()
        print('after demo')
        '''
        depth = inference(filepath)
        print("> inference elapsed: " + str(round(time.time() - start, 2)))

        depth_filename = Path(filepath).stem + "_depth.png"
        depth_path = os.path.join(app.config['UPLOAD_FOLDER'], depth_filename)
        depth.save(depth_path)
        '''
        # resp = jsonify({'message' : 'File successfully uploaded'})
        # resp.status_code = 201
        # return resp

        thread = Thread(target=delayed_clear_files, kwargs={'value': request.args.get('value', 5)})
        thread.start()

        return send_file(app.config['depth_img_path'], mimetype='image/png')
    else:
        resp = jsonify({'message' : 'Allowed file types are png, jpg, jpeg'})
        resp.status_code = 400
        return resp


if __name__ == '__main__':
   app.run()