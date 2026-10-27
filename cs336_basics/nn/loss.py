import torch
"""
Cross Entropy is -log(prob).sum() / batch_size, where prob is the softmax of the logits.
but prob can get small and potentially round down to 0. Instead we do the following:
CE = -log(softmax(logits)).sum() -> substitute the equation of softmax
CE = log(sum(exp(logits))) - logit[target]

Before running exp we have to subtract the max to avoid overflow
CE = log(sum(exp(logits - max(logits)))) - (logit[target] - max(logits))
"""
class CrossEntropyLoss(torch.nn.Module):
    def forward(self, x, target):

        logit_max = x.amax(-1, keepdim=True)
        log_sum_exp = torch.log(torch.sum(torch.exp(x - logit_max), dim=-1, keepdim=True))
        cross_entropy = log_sum_exp - (torch.gather(x, -1, target.unsqueeze(-1)) - logit_max)
        return cross_entropy.mean()
