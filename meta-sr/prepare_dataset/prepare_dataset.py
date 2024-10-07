import glob
import cv2
import numpy as np
import csv
import os
from tqdm import tqdm
import random
import shutil
#from utils import build_iqa_model, load_pretrained_state_dict, make_directory, AverageMeter, ProgressMeter, Summary
#import torch
#from imgproc import image_to_tensor, image_resize

def generate_LR_x1_x4(HR_folder):
    os.makedirs(os.path.join(HR_folder, 'LR_bicubic'), exist_ok=True)
    if os.path.exists(HR_folder):
        image_list = os.listdir(os.path.join(HR_folder, 'HR'))
        for image_ind in tqdm(range(len(image_list))):
            image_filename = image_list[image_ind]
            image_path = os.path.join(os.getcwd(),HR_folder, 'HR')
            print(image_path)
            
            image = cv2.imread(os.path.join(image_path, image_filename))
            
            print(f"scale: image H {image.shape},  ")

            #for scale in scales:
            for scale in np.arange(1.10, 4.10, 0.10):
            
                scale = round(scale, 2)
                image_h, image_w = image.shape[0:2]

                if image_h % scale != 0:
                    image_h -= 13
                if image_w % scale != 0:
                    image_w -= 13
                
                image_ = image[0:image_h, 0:image_w, :]

                scaled_image = cv2.resize(image_, None, fx = 1/scale, fy = 1/scale, interpolation=cv2.INTER_CUBIC)
                os.makedirs(os.path.join(HR_folder, 'LR_bicubic', f"X{scale:.2f}"), exist_ok=True)   
                cv2.imwrite(os.path.join(os.getcwd(), HR_folder, 'LR_bicubic', f"X{scale:.2f}", f"{image_filename}"), scaled_image) 
        
    else:
        raise FileNotFoundError(f"Image folder {HR_folder} does not exist.")
def downscale():
  val_datafolder = "/nfs/home/data/wzry_data/dataset/test"
  LR = "/nfs/home/data/wzry_data/dataset"
  if not os.path.exists(val_datafolder):
    raise FileNotFoundError(f"Image folder {val_datafolder} does not exist.")
  image_list = os.listdir(val_datafolder)
  
  for image_ind in tqdm(range(len(image_list))):
            image_filename = image_list[image_ind]
            image = cv2.imread(os.path.join(val_datafolder, image_filename))

            scaled_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            
            cv2.imwrite(os.path.join(LR, "test_lr", f"{image_filename}"), scaled_image) 
                         
def testBilinear():
    LR = "/home/usw00078/SRGAN-PyTorch/data/wzry/x2/LR"
    
    GT = "/home/usw00078/SRGAN-PyTorch/data/wzry/x2/GT"
    if not os.path.exists(LR):
        raise FileNotFoundError(f"Image folder {LR} does not exist.")
    image_list = os.listdir(LR)

    #device = torch.device("cuda", config["DEVICE_ID"])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    psnr_model, ssim_model = build_iqa_model(
        #config["SCALE"],
        2,
        #config["TEST"]["ONLY_TEST_Y_CHANNEL"],
        True, 
        device,
    )
      
    for image_ind in tqdm(range(len(image_list))):
                image_filename = image_list[image_ind]
                image = cv2.imread(os.path.join(LR, image_filename))
                #image_gt = cv2.imread(os.path.join(GT, image_filename))

                gt_image = cv2.imread(os.path.join(GT, image_filename)).astype(np.float32) / 255.
                #import pdb; pdb.set_trace()
                gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
                gt = image_to_tensor(gt_image, False, False).to(device, non_blocking=True)
                gt = gt.unsqueeze(0)

                scaled_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                            # Calculate the image sharpness evaluation index
                cv2.imwrite(os.path.join(GT, "test_bi", f"{image_filename}"), scaled_image) 

                sr_image = cv2.imread(os.path.join(GT, "test_bi", f"{image_filename}")).astype(np.float32) / 255.
                #import pdb; pdb.set_trace()
          
                sr_image = cv2.cvtColor(sr_image, cv2.COLOR_BGR2RGB)
                sr = image_to_tensor(sr_image, False, False).to(device, non_blocking=True)
                sr = sr.unsqueeze(0)



                psnr = psnr_model(sr, gt)
                ssim = ssim_model(sr, gt)
                print(f"psnr for {image_filename} is: {psnr}")
                print(f"ssim for {image_filename} is: {ssim}")
                

def split():
    DATASET = "wzry"  # "danzai"  # "wzry"
    random.seed(a=32)
    if DATASET == "wzry":
        IMAGE_FOLDER = "/nfs/home/data/wzry_data/veryhigh_lowfps"
         # ????????
        if not os.path.exists(IMAGE_FOLDER):
            raise FileNotFoundError(f"Image folder {IMAGE_FOLDER} does not exist.")
        image_list = sorted(glob.glob(os.path.join(IMAGE_FOLDER, '*', '*', 'view_ui', '*.png')))
        if not image_list:
            raise FileNotFoundError(f"No images found in {IMAGE_FOLDER} with the given pattern.")
    else:
        raise ValueError(f"{DATASET} not supported, choices are wzry / danzai")
    
    OUTPUT_FOLDER = f"/nfs/home/data/wzry_data/dataset/"
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)   
    random.seed(a=32)
    import pdb; pdb.set_trace()
    
    CSV_FILEPATH = OUTPUT_FOLDER + '.csv'
    if os.path.exists(CSV_FILEPATH):
        os.remove(CSV_FILEPATH)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "val"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "test"), exist_ok=True)
    
    with open(CSV_FILEPATH, 'w', newline='') as csvfile:
        cvswriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for image_ind in tqdm(range(len(image_list))):
            image_filename = image_list[image_ind]
            image = cv2.imread(image_filename)
            if random.random() < 0.6:
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, "train", f"{image_ind:05d}.png"), image)                
                cvswriter.writerow([f"{image_ind:05d}", image_filename, "train"])
            elif random.random() < 0.8:
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, "val", f"{image_ind:05d}.png"), image)
                cvswriter.writerow([f"{image_ind:05d}", image_filename, "val"])
            else:
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, "test", f"{image_ind:05d}.png"), image)
                cvswriter.writerow([f"{image_ind:05d}", image_filename, "test"])
def test(s):
    pass
  

if __name__ == "__main__":
  #downscale()
  #testBilinear()
  import sys
  HR_path = sys.argv[1]
  generate_LR_x1_x4(HR_path)


