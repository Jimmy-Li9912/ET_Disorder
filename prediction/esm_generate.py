import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

from Bio import SeqIO
import esm
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse


device = torch.device("cuda:0")
device_id = [0,1]
print(device)
print(torch.cuda.device_count())
# Load ESM-2 model
#model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
#model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()

model = nn.DataParallel(model, device_ids=device_id,output_device=1)
#model = DDP(model)
model=model.to(device)

model.eval()

batch_converter = alphabet.get_batch_converter()
embedding_feature_dim = 2560
layers = 36
mean_embedding_feature_dim =40
padding_length=1632


def data_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    print(inputs)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)

def trans_data_msa(str_array):
    #print(str_array)
    batch_labels, batch_strs, batch_tokens = batch_converter(str_array)
    #batch_tokens = batch_tokens
    print(batch_tokens.shape)
    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        if batch_tokens.shape[-1] > 0:#<1024
            results = model(tokens=batch_tokens.to("cuda:1"), repr_layers=[layers], return_contacts=False)
            token_representations = results["representations"][layers].detach().cpu()
        else:
            print("long seq")
            length = batch_tokens.shape[-1]
            results1 = model(tokens=batch_tokens[:,0:1024].to("cuda:1"), repr_layers=[layers], return_contacts=False)
            repre1=results1["representations"][layers].detach().cpu()
            results2 = model(tokens=batch_tokens[:,824:].to("cuda:1"), repr_layers=[layers], return_contacts=False)
            repre2=results2["representations"][layers].detach().cpu()
            #repre1 = F.pad(repre1,(0,0,0,length-1024),'constant',0)
            #repre2 = F.pad(repre2,(0,0,824,0),'constant',0)
            #print(repre1[:,50:-50,:].shape)
            #print(repre2[:,50:-50,:].shape)
            token_representations = torch.cat([repre1[:,50:-50,:],repre2[:,50:-50,:]],dim=1)
            #print(token_representations.shape)

    #print(token_representations.shape)
    sequence_representations = []
    for i, (_, seq) in enumerate(str_array):
        temp_tensor = token_representations[i, 1: len(seq) + 1]
        #--- 1x1280
        #sequence_representations.append(temp_tensor.mean(0).detach().cpu().numpy().tolist())
        #-- n*1280
        sequence_representations.append(temp_tensor)
    #print(sequence_representations[0].shape)
    result = sequence_representations[0].numpy().tolist()
    #zero_padding = np.zeros(shape=[embedding_feature_dim])
    #while (len(result) < padding_length):
    #    result.append(zero_padding)
    result = torch.Tensor(np.array(result))
    print(result.shape)
    #--- 1 x 1280
    #result = torch.tensor(np.array(sequence_representations))
    #----
    return result.reshape(-1,embedding_feature_dim)

def trans_data_msa_in_batches(str_array, split=1, path="./embedding/train_feature.npy"):
    if(os.path.exists(path)):
        results = torch.load(path)
        print("MSA feature shape:")
        #print(type(results))
        #for item in results:
        #    print(item.shape[0])

    else:
        divide_num = int(len(str_array)/split)
        results=[]

        for i in range(1, divide_num+1):
            print("msa process batch "+str(i)+":")
            results.append(trans_data_msa(str_array[(i-1)*split:i*split]))

        if (len(str_array) % split != 0):
            print("msa process batch " + str(i+1) + ":")
            results.append(trans_data_msa(str_array[divide_num * split:len(str_array)]))

        #embedding_result = torch.cat(results).detach().cpu().numpy()
        print("MSA feature shape:")
        print(len(results))
        torch.save(results, path)
    return results

def createDatasetEmbedding(data_path, save_path):
    raw_data=[]
    fasta_data = [seq_record for seq_record in SeqIO.parse(data_path, 'fasta')]
    for item in fasta_data:
        string = str(item.seq)
        #if len(string)>1022:
            #string = string[:1022]
        raw_data.append(("protein", string))

    features = trans_data_msa_in_batches(raw_data, path=save_path)
    return features

def Round(deci=3 ,path="./embedding/train_feature.npy"):

    embedding_result = np.round(np.load(path), decimals=deci)

    print("MSA feature shape:")
    print(embedding_result.shape)
    np.save(path, embedding_result)



if __name__ == '__main__':
    #createDatasetEmbedding('../NSP3/CB513_new.fasta', "../NSP3/CB513_nopad_esm2_15B_full.pt")
    #createDatasetEmbedding('../NSP3/CASP12_new.fasta', "../NSP3/CASP12_nopad_esm2_15B_full.pt")
    #createDatasetEmbedding('../NSP3/TS115_new.fasta', "../NSP3/TS115_nopad_esm2_15B_full.pt")
    #createDatasetEmbedding('../NSP3/Train_new.fasta', "../NSP3/Train_nopad_esm2_3B_full.pt")
    #createDatasetEmbedding('../NSP3/Disprot228.fasta', "../NSP3/Disprot228_nopad_esm1b_full.pt")
    parser = argparse.ArgumentParser(description='DeepCRF Baseline Model')
    parser.add_argument('--i', default="../NSP3/demo/0.fasta", type=str, help='input file')
    parser.add_argument('--o', default="../NSP3/demo/0.pt", type=str, help='output file')
    args = parser.parse_args()
    createDatasetEmbedding(args.i, args.o)

    #createDatasetEmbedding('./test_data/data_list.txt', "./embedding/test_feature.npy")
    # Round(3,"./embedding/train_feature.npy" )
    # Round(3,"./embedding/test_feature.npy" )

