#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D

class DomainAdapter(nn.Module):


    def __init__(self,
                 node: int = 50,
                 time: int = 12,
                 input_dim: int = 10,
                 domain_num: int = 20,
                 hidden_dim: int = 64,
                 external = False,
                 external_dim=-1,
                 domain_embedding_dim=32,
                 ):
        super().__init__()
        self.domain_num = domain_num
        self.domain_embedding_dim = domain_embedding_dim

        if external:
            self.external_dim = external_dim
            self.external_encoder = nn.Sequential(
                nn.Linear(external_dim, hidden_dim),
                nn.LeakyReLU(0.2, True),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.adapter = nn.Sequential(
            nn.Linear(time*input_dim,hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, domain_num)
        )

        self.domain_embedding = nn.Parameter(torch.randn(domain_num, domain_embedding_dim), requires_grad=True)
        nn.init.orthogonal_(self.domain_embedding)

    def forward(self, x, external=None):
        """
        :param x:
        :param external:
        :return: [B,N,R]  [B, N, C]
        """
        batch, node, time, input_dim = x.shape
        # [B, N, T, D] -> [B, N, TD] ——MLP——> [B, N, R]
        domain_class = self.adapter(x.reshape(batch, node, time*input_dim))

        x_domain_class = F.gumbel_softmax(domain_class, tau=1, hard=True, dim=-1)     # [B, N, R]

        x_domain_embedding= (x_domain_class.unsqueeze(-1)*self.domain_embedding.unsqueeze(0).unsqueeze(1)).sum(2)

        return x_domain_class, x_domain_embedding, domain_class

    def embedding_constrains(self):
        """
        :return:
        """
        mat = torch.abs(torch.mm(self.domain_embedding, self.domain_embedding.T))
        emb_diag = torch.diag_embed(torch.diag(mat))

        diversity_loss = (mat - emb_diag).sum()

        return diversity_loss

if __name__ == '__main__':

    x = torch.randn(32,50,12,10)
    adapter = DomainAdapter()
    embedding = adapter(x)


    l1, l2 = adapter.embedding_constrains()


    print(embedding.shape)
