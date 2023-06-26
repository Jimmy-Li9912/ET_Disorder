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
    parser = argparse.ArgumentParser(description='DeepCRF Baseline Model')
    parser.add_argument('--model', default="./model/ckpt_cnn_2_0.719.pth", type=str, help='model path')
    parser.add_argument('--esm', default="./demo/0_esm.pt", type=str, help='esm file')
    parser.add_argument('--plddt', default='./demo/0_plddt.pt', type=str, help='plddt file')
    parser.add_argument('--o', default="./demo/0.pt", type=str, help='output file')
    args = parser.parse_args()

    model_path=args.model
    ESM_path = args.esm
    PLDDT_path = args.plddt
    out_path = args.o
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
