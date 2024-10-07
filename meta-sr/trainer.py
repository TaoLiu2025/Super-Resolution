import os
import math
from decimal import Decimal

import utility
import pdb
import torch
from torch.autograd import Variable
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8


    ######by given the scale and the size of input image
    ######we caculate the input matrix for the weight prediction network
    ###### input matrix for weight prediction network
    def input_matrix_wpn(self,inH, inW, scale, add_scale=True):
        '''
        inH, inW: the size of the feature maps
        scale: is the upsampling times
        '''
        outH, outW = int(scale*inH), int(scale*inW)

        #### mask records which pixel is invalid, 1 valid or o invalid
        #### h_offset and w_offset caculate the offset to generate the input matrix
        scale_int = int(math.ceil(scale))
        h_offset = torch.ones(inH, scale_int, 1)
        mask_h = torch.zeros(inH,  scale_int, 1)
        w_offset = torch.ones(1, inW, scale_int)
        mask_w = torch.zeros(1, inW, scale_int)
        if add_scale:
            scale_mat = torch.zeros(1,1)
            scale_mat[0,0] = 1.0/scale
            #res_scale = scale_int - scale
            #scale_mat[0,scale_int-1]=1-res_scale
            #scale_mat[0,scale_int-2]= res_scale
            scale_mat = torch.cat([scale_mat]*(inH*inW*(scale_int**2)),0)  ###(inH*inW*scale_int**2, 4)

        ####projection  coordinate  and caculate the offset 
        h_project_coord = torch.arange(0,outH, 1).float().mul(1.0/scale)
        int_h_project_coord = torch.floor(h_project_coord)

        offset_h_coord = h_project_coord - int_h_project_coord
        int_h_project_coord = int_h_project_coord.int()

        w_project_coord = torch.arange(0, outW, 1).float().mul(1.0/scale)
        int_w_project_coord = torch.floor(w_project_coord)

        offset_w_coord = w_project_coord - int_w_project_coord
        int_w_project_coord = int_w_project_coord.int()

        ####flag for   number for current coordinate LR image
        flag = 0
        number = 0
        for i in range(outH):
            if int_h_project_coord[i] == number:
                h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], flag,  0] = 1
                flag += 1
            else:
                h_offset[int_h_project_coord[i], 0, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], 0, 0] = 1
                number += 1
                flag = 1

        flag = 0
        number = 0
        for i in range(outW):
            if int_w_project_coord[i] == number:
                w_offset[0, int_w_project_coord[i], flag] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], flag] = 1
                flag += 1
            else:
                w_offset[0, int_w_project_coord[i], 0] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], 0] = 1
                number += 1
                flag = 1

        ## the size is scale_int* inH* (scal_int*inW)
        h_offset_coord = torch.cat([h_offset] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        w_offset_coord = torch.cat([w_offset] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
        ####
        mask_h = torch.cat([mask_h] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        mask_w = torch.cat([mask_w] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)

        pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
        mask_mat = torch.sum(torch.cat((mask_h,mask_w),2),2).view(scale_int*inH,scale_int*inW)
        mask_mat = mask_mat.eq(2)
        pos_mat = pos_mat.contiguous().view(1, -1,2)
        if add_scale:
            pos_mat = torch.cat((scale_mat.view(1,-1,1), pos_mat),2)

        return pos_mat,mask_mat ##outH*outW*2 outH=scale_int*inH , outW = scale_int *inW

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        #self.loader_train.dataset.set_scale(0)
        timer_data, timer_model = utility.timer(), utility.timer()
        
        
        #for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
        for idx_scale, scale in enumerate(self.scale):   
            self.loader_train.dataset.set_scale(idx_scale)
            for batch, (lr, hr, _) in enumerate(self.loader_train):
                #print(f"index scale: {idx_scale}, scale: {scale}")

                lr, hr = self.prepare(lr, hr)
                timer_data.hold()
                timer_model.tic()
                N,C,H,W = lr.size()
                #print(f"lr size : {H}, {W}")

                _,_,outH,outW = hr.size()
                scale_coord_map, mask = self.input_matrix_wpn(H,W,self.args.scale[idx_scale])  ###  get the position matrix, mask
                #import pdb;pdb.set_trace()
                #print(f"Hr gt size : {outH}, {outW}")

                if self.args.n_GPUs>1 and not self.args.cpu:
                    scale_coord_map = torch.cat([scale_coord_map]*self.args.n_GPUs,0)
                else:
                    scale_coord_map = scale_coord_map.to(device)
                
                self.optimizer.zero_grad()
                sr = self.model(lr, idx_scale, scale_coord_map)
                re_sr = torch.masked_select(sr,mask.to(device))
                #print(f"hr size : {re_sr.shape}")
                #import pdb; pdb.set_trace()
                re_sr = re_sr.contiguous().view(N,C,outH,outW)
                #import pdb; pdb.set_trace()
                loss = self.loss(re_sr, hr)
                
                if loss.item() < self.args.skip_threshold * self.error_last:
                    loss.backward()
                    self.optimizer.step()
                else:
                    print('Skip this batch {}! (Loss: {})'.format(
                        batch + 1, loss.item()
                    ))

                timer_model.hold()

                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                        (batch + 1) * self.args.batch_size,
                        len(self.loader_train.dataset),
                        self.loss.display_loss(batch),
                        timer_model.release(),
                        timer_data.release()))

                timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        if self.args.n_GPUs == 1:
            target = self.model
        else:
            target = self.model  #.module

        torch.save(
            target.state_dict(),
            os.path.join(self.ckp.dir,'model', 'model_{}.pt'.format(epoch))
        )
        ## save models
    

    def test(self): 
        
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()
        timer_test = utility.timer()
        timer_test1 = utility.timer()
        
        warm_up = 5
        test_whole = True

        device = torch.device('cpu' if self.args.cpu else 'cuda')
    
        with torch.no_grad():           
            for idx_data, d in enumerate(self.loader_test):
                for idx_scale, scale in enumerate(self.scale):
                    d.dataset.set_scale(idx_scale)
                    eval_acc = 0
                    eval_acc_ssim = 0
                    timer_test1.reset()
                    
                    for lr_, hr_, filename in tqdm(d, ncols=80):
                        filename = filename[0]
                        # extract patches
                        N,C,H,W = lr_.size()
                        timer_test.reset()
                        save_list = []
                        if not test_whole:
                            print("===========\n patches ")
                        
                            patch_height, patch_width = self.args.patch_size, self.args.patch_size                           
                            patch_list = []
                            img_sr = torch.zeros(N,C,int(scale*H),int(scale*W))
                            img_sr = img_sr.to(device)
                            lr_ = lr_.to(device)
                            hr_ = hr_.to(device)

                            #patch_rows = -1 
                            patch_rows = 0                      
                            for i in range(0, W, patch_width):
                                patch_rows += 1
                                patch_cols = 0
                                for j in range(0, H, patch_height):
                                    patch_cols += 1
                                    right = min(i + patch_width, W)
                                    bottom = min(j + patch_height, H)
                                    patch_lr = lr_[:, :, j: bottom, i:right]
                                    patch_lr_h, patch_lr_w = int(patch_lr.shape[2] * scale), int(patch_lr.shape[3] * scale)

                                    hr_right_start = int(i * scale)
                                    hr_right_end = min(int(right * scale), hr_.shape[3])
                                    hr_bottom_start = int(j * scale)
                                    hr_bottom_end = min(int(bottom * scale), hr_.shape[2])

                                    patch_hr = hr_[:, :, hr_bottom_start:hr_bottom_end, hr_right_start:hr_right_end]
                                    #patch_hr = hr_[:, :, int(j*scale):int(bottom*scale), int(i*scale):int(right*scale)]
                                    #print(f"patch_hr : {patch_hr.shape}")

                                    h, w = patch_lr.shape[2:]
                                    #print(f"patch_lr : {patch_lr.shape}")
                                    
                                    outH,outW = int(h*scale), int(w*scale)

                                    # Super resolution for each patches              
                                    no_eval = (hr_.nelement() == 1)
                                    if not no_eval:
                                        patch_lr, patch_hr = self.prepare(patch_lr, patch_hr)
                                    else:
                                        patch_lr, = self.prepare(patch_lr)                           
                                    scale_coord_map, mask = self.input_matrix_wpn(h,w,self.args.scale[idx_scale])
                                    if self.args.n_GPUs>1 and not self.args.cpu:
                                        scale_coord_map = torch.cat([scale_coord_map]*self.args.n_GPUs,0)
                                    else:
                                        scale_coord_map = scale_coord_map.to(device)
                                    while warm_up:
                                        sr = self.model(patch_lr, idx_scale,scale_coord_map) 
                                        warm_up -= 1
                                        
                                    timer_test.tic()
                                    timer_test1.tic()
                                    #import pdb;pdb.set_trace()
                                    sr = self.model(patch_lr, idx_scale,scale_coord_map)                
                                    timer_test.hold()
                                    timer_test1.hold()
                                    #print(f"total time : {timer_test1.acc} toc: {timer_test1.toc()}")
                                    
                                    re_sr = torch.masked_select(sr,mask.to(device))
                                    #import pdb;pdb.set_trace()
                                    sr = re_sr.contiguous().view(N,C,outH,outW)

                                    #import pdb; pdb.set_trace()
                                    sr = utility.quantize(sr, self.args.rgb_range)
                                    patch_list.append(sr) 
                                   
                                    # print(f"lr: {lr_.shape}, hr: {hr_.shape}patch lr : {patch_lr.shape}, img_sr : {img_sr.shape}\n \
                                    # patch hr:{patch_hr.shape} lr*scale_h: {patch_lr.shape[2]*scale},lr*scale_w: {patch_lr.shape[3]*scale}, sr: {sr.shape}")  
                  
                            
                            w_offset = 0
                            for row in range(patch_rows):
                                h_offset = 0
                                for col in range(patch_cols):
                                    index_ = row * patch_cols + col
                                    sr_ = patch_list[index_]
                                    patch_height_, patch_width_ = sr_.shape[2:]
                                    

                                    #print(f"sr shape:{sr_.shape} h_offset: {h_offset}, w_offset {w_offset} , img_sr: {img_sr.shape}")
                                    img_sr[:, :, h_offset: h_offset + patch_height_, w_offset: w_offset + patch_width_] = sr_
                                    h_offset += patch_height_
                                w_offset += patch_width_
                                #import pdb; pdb.set_trace()
                            outH,outW = int(H*scale),int(W*scale)

                            # for index, v in enumerate(patch_list):
                            #     self.ckp.save_patches_results(filename, v, scale, self.args.patch_size, index)
                                
                
                            if not no_eval:
                                #import pdb; pdb.set_trace()
                                #print(f"sr shape: {img_sr.shape} hr_ shape:{hr_.shape}")
                                #eval_acc += utility.calc_psnr(
                                eval_acc += utility.calc_psnr(
                                    #sr, hr, scale, self.args.rgb_range,
                                    
                                    img_sr, hr_, scale, self.args.rgb_range,
                                    #benchmark=self.loader_test.dataset.benchmark
                                    benchmark=d.dataset.benchmark
                                )
                                eval_acc_ssim += utility.calc_ssim(
                                    #sr, hr, scale,
                                    img_sr, hr_, scale,
                                    benchmark=d.dataset.benchmark
                                )
                                
                                save_list.extend([img_sr, lr_, hr_])

                            if self.args.save_results:
                                a=1
                                self.ckp.save_results(filename, save_list, scale)

                            #print(f"\n Inferenct time for image: {filename}, size: {lr_.shape}, scale: {scale}: {timer_test.acc:.4f} s ")
                            #print(f"\n [{self.args.data_test}  x{scale}]\t PSNR:{eval_acc:.2f} SSIM: {eval_acc_ssim:.2f} ")

                        else:
                            print("================\n whole images")
                            no_eval = (hr_.nelement() == 1)
                            if not no_eval:
                                lr_, hr_ = self.prepare(lr_, hr_)
                            else:
                                lr_, = self.prepare(lr_)                           
                            scale_coord_map, mask = self.input_matrix_wpn(H,W,self.args.scale[idx_scale])
                            if self.args.n_GPUs>1 and not self.args.cpu:
                                scale_coord_map = torch.cat([scale_coord_map]*self.args.n_GPUs,0)
                            else:
                                scale_coord_map = scale_coord_map.to(device)

                            while warm_up:
                                sr = self.model(lr_, idx_scale,scale_coord_map) 
                                warm_up -= 1

                            timer_test.tic()
                            sr = self.model(lr_, idx_scale,scale_coord_map)
                            timer_test.hold()
                            outH, outW = int(H*scale), int(W*scale)
                            re_sr = torch.masked_select(sr,mask.to(device))
                            sr = re_sr.contiguous().view(N,C,outH,outW)
                            sr = utility.quantize(sr, self.args.rgb_range)
                            #timer_test.hold()
                            save_list = [sr]

                            if not no_eval:
                                eval_acc += utility.calc_psnr(                              
                                    sr, hr_, scale, self.args.rgb_range,
                                    #benchmark=self.loader_test.dataset.benchmark
                                    benchmark=d.dataset.benchmark
                                )
                                eval_acc_ssim += utility.calc_ssim(
                                    #sr, hr, scale,
                                    sr, hr_, scale,
                                    benchmark=d.dataset.benchmark
                                )
                                
                                save_list.extend([lr_, hr_])

                            if self.args.save_results:
                                a=1
                                self.ckp.save_results(filename, save_list, scale)



                    self.ckp.log[-1, idx_scale] = eval_acc / len(d)
                    best = self.ckp.log.max(0)            
                    print(f"{self.args.data_test} dataset average inference time: {timer_test.acc/len(d):.6f} lenth: {len(d)}")
                    
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} SSIM: {:.4f} (Best: {:.3f} @epoch {})'.format(
                            self.args.data_test,
                            scale,
                            self.ckp.log[-1, idx_scale],
                            eval_acc_ssim / len(d),
                            best[0][idx_scale],
                            best[1][idx_scale] + 1
                        )
                    )
                    self.ckp.write_log(
                        #'Total time: {:.2f}s\n'.format(timer_test1.toc()), 
                        f"{self.args.data_test} dataset average inference time: {timer_test.acc/len(d):.6f} lenth: {len(d)}",
                        refresh=True
                        )
                
        # self.ckp.write_log(
        #     'Total time: {:.2f}s\n'.format(timer_test1.toc()), refresh=True)
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch)) 


    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

