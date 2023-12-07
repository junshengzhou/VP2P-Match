from numpy import positive
import torch
import torch.nn.functional as F
import numpy as np


def det_loss(img_score_inline,img_score_outline,pc_score_inline,pc_score_outline,dists,mask):
    #score (B,N)
    pids=torch.FloatTensor(np.arange(mask.size(-1))).to(mask.device)
    diag_mask=torch.eq(torch.unsqueeze(pids,dim=1),torch.unsqueeze(pids,dim=0)).unsqueeze(0).expand(mask.size()).float()

    loss_inline=torch.mean(1-img_score_inline)+torch.mean(1-pc_score_inline)
    loss_outline=torch.mean(img_score_outline)+torch.mean(pc_score_outline)
    return loss_inline+loss_outline

def adapt_loss(img_features,pc_features,mask,margin=0.25, gamma=32):

    pos_mask=mask
    neg_mask=1-mask
    dists=torch.sum(img_features.unsqueeze(-1)*pc_features.unsqueeze(-2),dim=1)   # dot similarity
    pos=dists * pos_mask

    neg=dists * neg_mask

    delta_p = 1 - margin
    delta_n = margin
    alpha_p = torch.clamp_min(- pos.detach() + 1 + margin, min=0.)
    alpha_n = torch.clamp_min(neg.detach() + margin, min=0.)

    pos[pos==0] = delta_p
    neg[neg==0] = delta_n

    logit_p = - alpha_p * (pos - delta_p) * gamma
    logit_n = alpha_n * (neg - delta_n) * gamma

    lse_positive_row=torch.logsumexp(logit_p,dim=-1)
    lse_positive_col=torch.logsumexp(logit_p,dim=-2)

    lse_negative_row=torch.logsumexp(logit_n,dim=-1)
    lse_negative_col=torch.logsumexp(logit_n,dim=-2)

    loss_col=F.softplus(lse_positive_row+lse_negative_row)
    loss_row=F.softplus(lse_positive_col+lse_negative_col)
    loss=loss_col+loss_row
    
    return torch.mean(loss),dists

if __name__ == '__main__':
    a = torch.rand(12,64,512)
    b = torch.rand(12,64,512)
    c = torch.sum(a.unsqueeze(-1)*b.unsqueeze(-2),dim=1)
    d = a.permute(0, 2,1) @ b

    print(adapt_loss(a,b))