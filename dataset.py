import random
import pickle as pk
from torch.utils.data import Dataset
import numpy as np


class _NumpyCompatUnpickler(pk.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _compat_pickle_load(file_obj):
    return _NumpyCompatUnpickler(file_obj).load()
               
class SASV_Dataset_redimnet_sslaasist(Dataset):
    def __init__(self, args, partition):
        self.part = partition
        self.embedding_dir = args.embedding_dir
        if self.part == "trn":
            self.spk_meta_dir = args.spk_meta_dir
            self.load_meta_information()
        else:
            sasv_trial = getattr(args, 'sasv_' + self.part + '_trial')
            with open(sasv_trial, "r") as f:
                self.utt_list = f.readlines()
        self.load_embeddings()

    def load_meta_information(self):
        with open(self.spk_meta_dir + "asvspoof5_spk_meta_trn.pk", "rb") as f:
            self.spk_meta = pk.load(f)

    def load_embeddings(self):
        # load saved countermeasures(CM) related preparations
        with open(self.embedding_dir + "sslaasist_asvspoof5_cm_embd_" + self.part + ".pk", "rb") as f:
            self.cm_embd = pk.load(f)
        # load saved automatic speaker verification(ASV) related preparations
        with open(self.embedding_dir + "redimnet_asvspoof5_asv_embd_" + self.part + ".pk", "rb") as f:
            self.asv_embd = pk.load(f)
        if self.part in ["dev", "eval"]:
            # load speaker models for development and evaluation sets
            with open(self.embedding_dir + "redimnet_asvspoof5_spk_model_" + self.part + ".pk", "rb") as f:
                self.spk_model = pk.load(f)

    def __len__(self):
        if self.part == "trn":
            return len(self.cm_embd.keys())
        elif self.part in ["dev", "eval"]:
            return len(self.utt_list)

    def __getitem__(self, idx):
        return getattr(self, 'getitem_'+self.part)(idx)

    def getitem_trn(self, index):
               
        temp=np.random.uniform(0,1)
        if temp <= 0.33:
            ans_type = 1
        else:
            ans_type = 0
        if ans_type == 1:  # target
            spk = random.choice(list(self.spk_meta.keys()))
            enr, tst = random.sample(self.spk_meta[spk]["bonafide"], 2)
            nontarget_type = 0
            label_type = 'target'
            asv_label, cm_label = 1, 1
        elif ans_type == 0:  # nontarget
            nontarget_type = random.randint(1, 2)
            if nontarget_type == 1:  # zero-effort nontarget
                spk, ze_spk = random.sample(list(self.spk_meta.keys()), 2)
                enr = random.choice(self.spk_meta[spk]["bonafide"])
                tst = random.choice(self.spk_meta[ze_spk]["bonafide"])
                label_type = 'nontarget'
                asv_label, cm_label = 0, 1

            if nontarget_type == 2:  # spoof nontarget
                spk = random.choice(list(self.spk_meta.keys()))
                if len(self.spk_meta[spk]["spoof"]) == 0:
                    while True:
                        spk = random.choice(list(self.spk_meta.keys()))
                        if len(self.spk_meta[spk]["spoof"]) != 0:
                            break
                enr = random.choice(self.spk_meta[spk]["bonafide"])
                tst = random.choice(self.spk_meta[spk]["spoof"])
                label_type = 'spoof'
                asv_label, cm_label = 1, 0
        else:
            raise ValueError

        return self.asv_embd[enr], self.asv_embd[tst], \
               self.cm_embd[tst],  ans_type, label_type, asv_label, cm_label

    def getitem_dev(self, index):
        line = self.utt_list[index]
        spkmd, key, ans = line.strip().split(" ")
        ans_type = int(ans == "target")
        if ans == 'target':
            asv_label, cm_label = 1, 1
        elif ans == 'nontarget':
            asv_label, cm_label = 0, 1
        else:
            asv_label, cm_label = 1, 0

        return self.spk_model[spkmd], self.asv_embd[key], \
               self.cm_embd[key], ans_type, ans, asv_label, cm_label

    def getitem_eval(self, index):
        line = self.utt_list[index]
        spkmd, key, _, _, ans = line.strip().split(" ")
        ans_type = int(ans == "target")
        if ans == 'target':
            asv_label, cm_label = 1, 1
        elif ans == 'nontarget':
            asv_label, cm_label = 0, 1
        else:
            asv_label, cm_label = 1, 0

        return self.spk_model[spkmd], self.asv_embd[key], \
               self.cm_embd[key], ans_type, ans, asv_label, cm_label
               
