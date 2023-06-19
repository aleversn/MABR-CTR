import torch
import torch.nn as nn
from torch.autograd import Variable
import ubr
import gating_network



class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """
        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class Attention(nn.Module):
    def __init__(self, input_size, embed_size, out_channels, num_negtive):
        super(Attention, self).__init__()

        self.input_size = input_size
        self.embed_size = embed_size
        self.out_channels = out_channels
        self.conv = nn.Conv1d(21, 5, 25)
        self.conv2 = nn.Conv1d(num_negtive, 5, 25)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(5, 5)),
            nn.Conv2d(1, self.out_channels, kernel_size=(1, self.embed_size - 4)),
            # nn.ReLU(),
            nn.MaxPool2d((self.input_size - 4, 1)))
        self.act = nn.ReLU()
        self.max_pool = nn.MaxPool2d((self.input_size, 1))

    def forward(self, macro, micro, prop_pref, is_neg=False):
        # review_emb, prop_scores,
        # Prop_scores reflect the properties of reviews
        # prop_scores shape: batch_num * num_property * num_reviews

        # prop_scores = torch.einsum('imj,im->imj', prop_scores, prop_pref)
        # review_emb = torch.einsum('ijk,imj->imjk', review_emb, prop_scores)
        # review_emb = torch.mean(review_emb, 1)
        # out = self.conv(review_emb)

        prop_scores = torch.einsum('imj,ik->imj', micro, prop_pref)
        macro = torch.einsum('ijk,imj->imjk', macro, prop_scores)
        macro = torch.mean(macro, 1)
        if is_neg:
            out = self.conv2(macro)
        else:
            out = self.conv(macro)

        out = self.act(out)
        out = torch.reshape(out, (out.size(0), -1))
        return out


class MB4CTR(nn.Module):

    def __init__(self, data_name, latent_dim, init_emb_wgts_bottom_items_path, total_emb_path, num_negtive,
                 device='cpu'):
        super(MB4CTR, self).__init__()
        self.input_size = 30
        self.embed_size = 1590
        self.channels = 30
        self.latent_dim = latent_dim
        self.review_out_dim = 600
        self.num_prop = 4
        self.emb_wgts_micro_items_dict = []
        self.init_emb_wgts_bottom_items_path = init_emb_wgts_bottom_items_path
        self.total_emb_path = total_emb_path
        if data_name == 'Computers_sample':
            self.num_users = 10
            self.num_items = 2255
            self.num_micro = 30
        elif data_name == 'Computers':
            self.num_users = 60242
            self.num_items = 62006
            self.num_micro = 30
        elif data_name == 'Applicances':
            self.num_users = 86549
            self.num_items = 59999
            self.num_micro = 40
        elif data_name == 'UserBehaviors':
            self.num_users = 117362
            self.num_items = 2579724
            self.num_micro = 200

        self.user_embedding = ScaledEmbedding(self.num_users + 1, self.num_micro)

        self.user_review_encode = Attention(self.input_size, self.embed_size, self.channels, num_negtive)
        self.user_latent_vector = ScaledEmbedding(self.num_users + 1, self.latent_dim + 1)
        self.user_prop_pref = ScaledEmbedding(self.num_users + 1, self.num_prop)
        self.hidden_dim = 768
        self.output_dim = 768
        self.actLayer = nn.ReLU()
        self.user_bias = nn.Embedding(self.num_users + 1, 1)
        self.item_bias = nn.Embedding(self.num_items + 1, 1)
        self.mu_bias = Variable(torch.ones(1), requires_grad=True)
        if device != 'cpu':
            self.mu_bias = self.mu_bias.to(device)

    def forward(self, user_id, macro, micro, is_neg=False):

        # user
        # user_feat = self.user_review_encode(user_review, user_review_scores, self.user_prop_pref(user_id))
        user_feat = self.user_review_encode(macro, micro, self.user_prop_pref(user_id), is_neg)
        user_feat = user_feat.view(user_feat.size(0), -1)
        user_feat = self.actLayer(user_feat)
        for i in range(len(user_id)):
            self.user_embedding.weight.data[user_id[i]].copy_(user_feat[i])

        out = torch.einsum('ij->', user_feat) + self.user_bias(user_id).squeeze() + self.mu_bias
        return out