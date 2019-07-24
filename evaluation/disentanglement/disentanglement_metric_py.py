import torch
import numpy as np


# receives a tensor where 
def kl_divergence(mean_1, sigma_1, mean_2, sigma_2):
    p_q = one_side_kl_divergence(mean_1, sigma_1, mean_2, sigma_2)
    
    mean_1, mean_2 = mean_2, mean_1
    sigma_1, sigma_2 = sigma_2, sigma_1
    q_p = one_side_kl_divergence(mean_1, sigma_1, mean_2, sigma_2)

    return p_q + q_p

def one_side_kl_divergence(mean_1, sigma_1, mean_2, sigma_2):
    A = torch.log(sigma_2 / sigma_1)
    B_up = (sigma_1**2) + (mean_1 - mean_2)**2
    B_down = 2*(sigma_2**2)

    return A + (B_up/B_down) - 0.5



### INVERT T FOR B FOR THE OTHER WAY AROUND
N_ = 20 # amount of samples folders
N = 20 # amount of samples per folder
for k in range(1, N_+1):
    segs = torch.load('samples%d/segs.pt' % k, map_location='cpu')
    full_results_obj = []
    full_results_bkg = []
    for b in range(0, N):
        for t in range(0, N):
            means_obj = []
            stds_obj = []
            means_bkg = []
            stds_bkg = []
            for in s range(0, N):
                mask = segs[s].squeeze().byte()
                background_mask = (1. - mask)

                filename = 'samples%d/%d_%d_%d.pt' % (k,t,s,b)
                im = torch.load(filename, map_location='cpu').squeeze()
                im = im.mean(dim=0)

                obj = im.view(-1).masked_select(mask.view(-1))
                bkg = im.view(-1).masked_select(background_mask.view(-1))
                
                mean, std = obj.mean(), obj.std()
                means_obj.append(mean)
                stds_obj.append(std)

                mean, std = bkg.mean(), bkg.std()
                means_bkg.append(mean)
                stds_bkg.append(std)

            kls_obj = []
            kls_bkg = []
            for i in range(0, N):
                for j in range(0, N):
                    if i != j:
                        kl = one_side_kl_divergence(means_obj[i], stds_obj[i], means_obj[j], stds_obj[j]).item()
                        kls_obj.append(kl)
                        
                        kl = one_side_kl_divergence(means_bkg[i], stds_bkg[i], means_bkg[j], stds_bkg[j]).item()
                        kls_bkg.append(kl)

            result_obj = np.mean(kls_obj)
            result_bkg = np.mean(kls_bkg)
            full_results_obj.append(result_obj)
            full_results_bkg.append(result_bkg)

            # ims = torch.stack(ims)
            # change = ims.std(dim=0).mean(dim=0) # review this mean
            
            # background = change * background_mask
            # not_background = change * mask
            
            # background_mean = background.sum() / background_mask.sum()
            # std_ims_bg[b,s] = background_mean

            # not_background_mean = not_background.sum() / mask.sum()
            # std_ims_not_bg[b,s] = not_background_mean

    print('N: ', k)
    print('Change Object:     %.2f %.2f %.2f' % (np.mean(full_results_obj), np.std(full_results_obj), np.median(full_results_obj)))
    # print(full_results_obj)
    print('Change Background: %.2f %.2f %.2f' % (np.mean(full_results_bkg), np.std(full_results_bkg), np.median(full_results_bkg)))
    # print(full_results_bkg)




