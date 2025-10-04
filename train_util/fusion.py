import numpy as np
import torch

class History(object):
    def __init__(self, n_data,device="cpu"):
        self.correctness = np.zeros((n_data))
        self.confidence = np.zeros((n_data))
        self.max_correctness = 1
        self.device=device

    #Parameter 1: Index. Parameter 2: Loss. Parameter 3: Confidence
    def correctness_update(self, data_idx, correctness, confidence):
        #probs = torch.nn.functional.softmax(output, dim=1)
        #confidence, _ = probs.max(dim=1)
        data_idx = data_idx.cpu().numpy()


        self.correctness[data_idx] =self.correctness[data_idx]+correctness.cpu().numpy()[:, np.newaxis]
        self.confidence[data_idx] = confidence.cpu().detach().numpy()[:, np.newaxis]

    # max correctness update
    def max_correctness_update(self, epoch):
        if epoch > 1:
            self.max_correctness += 1

    # correctness normalize (0 ~ 1) range
    def correctness_normalize(self, data):
        data_min = self.correctness.min()
        #data_max = float(self.max_correctness)
        data_max = float(self.correctness.max())

        return (data - data_min) / (data_max - data_min)

    def get_target_margin(self, data_idx1, data_idx2):
        data_idx1 = data_idx1.cpu().numpy()
        cum_correctness1 = self.correctness[data_idx1]
        cum_correctness2 = self.correctness[data_idx2]
        # normalize correctness values
        cum_correctness1 = self.correctness_normalize(cum_correctness1)
        cum_correctness2 = self.correctness_normalize(cum_correctness2)
        # make target pair
        n_pair = len(data_idx1)
        target1 = cum_correctness1[:n_pair]
        target2 = cum_correctness2[:n_pair]
        # calc target
        greater = np.array(target1 > target2, dtype='float')
        less = np.array(target1 < target2, dtype='float') * (-1)

        target = greater + less
        target = torch.from_numpy(target).float()
        # calc margin
        margin = abs(target1 - target2)
        margin = torch.from_numpy(margin).float()
        if self.device=="cuda":
            target = target.cuda()
            margin=margin.cuda()

        return target, margin


def rank_loss(confidence, idx, history):
    rank_input1 = confidence
    rank_input2 = torch.roll(confidence, -1)
    idx2 = torch.roll(idx, -1)

    rank_target, rank_margin = history.get_target_margin(idx, idx2)
    rank_target_nonzero = rank_target.clone()
    rank_target_nonzero[rank_target_nonzero == 0] = 1
    rank_input2 = rank_input2 + (rank_margin / rank_target_nonzero).reshape((-1,1))

    ranking_loss = torch.nn.MarginRankingLoss(margin=0.0)(rank_input1,
                                        rank_input2,
                                        -rank_target.reshape(-1,1))

    return ranking_loss
def model_forward(model,lossfn1,lossfn2,input0,input1,input2,tgt,idx,d1_history=None,d3_history=None,gram_history=None,mode="train",device="cpu"):
    input2 = input2.tolist()
    input2 = torch.tensor(input2)
    if device=="cuda":
        input0=input0.cuda()
        input1=input1.cuda()
        input2=input2.cuda()
        tgt=tgt.cuda()

    fusion_out, d1_logits, d3_logits,gram_logits, d1_conf, d3_conf,gram_conf=model(input0,input1,input2)

    d1_clf_loss = lossfn1(d1_logits, tgt)
    d3_clf_loss = lossfn1(d3_logits, tgt)
    gram_clf_loss = lossfn1(gram_logits, tgt)
    clf_loss=d1_clf_loss+d3_clf_loss+gram_clf_loss+lossfn1(fusion_out,tgt)

    if mode =="eval" or mode=="test":
        return clf_loss,fusion_out,tgt
    else:

        d1_loss = lossfn2(d1_logits, tgt).detach()
        d3_loss = lossfn2(d3_logits, tgt).detach()
        gram_loss = lossfn2(gram_logits, tgt).detach()

        # if USE_CUDA:
        #     idx = idx.cuda()
        d1_history.correctness_update(idx, d1_loss, d1_conf.squeeze())
        d3_history.correctness_update(idx, d3_loss, d3_conf.squeeze())
        gram_history.correctness_update(idx, gram_loss, gram_conf.squeeze())


        d1_rank_loss = rank_loss(d1_conf, idx, d1_history)
        d3_rank_loss = rank_loss(d3_conf, idx, d3_history)
        gram_rank_loss = rank_loss(gram_conf, idx, gram_history)
        crl_loss = d1_rank_loss + d3_rank_loss+gram_rank_loss
        loss = torch.mean(clf_loss + crl_loss)
        return loss,fusion_out,tgt
