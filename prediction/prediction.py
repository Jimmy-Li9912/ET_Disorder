import torch
import torch.nn as nn
import numpy as np

disp_dict = {0:'Disordered', 1: 'Ordered'}

def disordered_predicting(model, data):
    esm = data['esm'].cuda()
    conf = data['conf'].cuda()

    out = model(esm, conf)
    out = out[0].detach().cpu().numpy()
    out = np.argmax(out,axis=-1).reshape(-1)
    #print(out.shape)
    return out.tolist()
def batch_prediction(model, data):
    return
if __name__ == '__main__':
    model_path='./model/ckpt_cnn_2_0.719.pth'
    ESM_path = './demo/0_esm.pt'
    PLDDT_path = './demo/0_plddt.pt'
    out_path = './demo/0_result.txt'
    esm = torch.load(ESM_path)[0].unsqueeze(0)
    conf = torch.load(PLDDT_path)[0].unsqueeze(0).unsqueeze(dim=-1)
    #print(esm.shape)
    #print(conf.shape)

    model=torch.load(model_path).cuda()
    out = disordered_predicting(model, {'esm':esm, 'conf':conf})
    result = [disp_dict[e] for e in out]
    print(result)
    out_str = ""
    for i in result:
        out_str += i[0]
    with open(out_path, 'w+') as f:
        f.write(out_str)
        f.close()
