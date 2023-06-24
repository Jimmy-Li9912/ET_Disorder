import numpy as np
import torch
from Bio import SeqIO
import argparse

def createPlDD(filename = '../struct' ,fastaname="1.fasta"):
    for item in SeqIO.parse(fastaname, "fasta"):
        length=len(item.seq)
    id_confidence = {k:0 for k in range(1,length+1)}
    id_count = {}
    with open(filename, 'r') as f:
        for line in f.read().split('\n'):
            feat = line.split()
            if feat[0] != 'END' :
                if(feat[0]=='ATOM'):
                    res_id = int(feat[5])
                    pldd = float(feat[10])
                #print(line)
                    #print(res_id," ", pldd)
                    if res_id in id_count.keys():
                        id_count[res_id] += 1
                        id_confidence[res_id] += pldd
                    else:
                        id_count[res_id] = 1
                        id_confidence[res_id] = pldd

            else:
                print("end")
                break
        f.close()
        #print(id_confidence)
        #print(id_count)

        id_confidence = {k: id_confidence[k]/v for k,v in id_count.items()}
        conf_list = torch.Tensor(list(id_confidence.values()))/100

        #meanval = torch.mean(conf_list)
        #maxval = torch.max(conf_list)
        #minval = torch.min(conf_list)
        #conf_list = (conf_list-meanval)/(maxval-minval)
        #print(conf_list)
        #print(conf_list.shape)
    return  conf_list
def readidx(path):
    with open(path, 'r') as f:
        a = f.read()
        b = a.split(',')
        c = list(map(int, b))
        f.close()
    return c



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepCRF Baseline Model')
    parser.add_argument('--pdb', default="../NSP3/demo/0.pdb", type=str, help='input pdb file')
    parser.add_argument('--seq', default="../NSP3/demo/0.pdb", type=str, help='input sequence file (fasta format)')
    parser.add_argument('--dir', default=None, type=str, help='pdb directory')
    parser.add_argument('--o', default="../NSP3/demo/0_plddt.pt", type=str, help='output file')
    args = parser.parse_args()
    lists = []
    if args.dir == None:
        try:
            s=createPlDD(args.pdb, fastaname=args.seq)
        except:
            s=torch.Tensor([])
        lists.append(s)
    #else:
    #    for i in range(1):
    #        try:
    #            s=createPlDD(f"./NSP3/{fold}/{i}.pdb", fastaname=f'./NSP3/{fold}/{i}.fasta')
    #        except:
    #            s=torch.Tensor([])
    #        lists.append(s)

    print(len(lists))
    torch.save(lists, args.o)
