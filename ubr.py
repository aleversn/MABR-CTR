import torch
import torch.nn as nn
import torch.nn.functional as F


class UBRBase(object):
    def __init__(self, feature_size, eb_dim, hidden_size, record_fnum, emb_initializer):
        self.record_fnum = record_fnum

        # input placeholders
        self.target_ph = None
        self.rewards = None
        self.lr = None

        # embedding
        if emb_initializer is not None:
            self.emb_mtx = nn.Parameter(torch.Tensor(emb_initializer))
        else:
            self.emb_mtx = nn.Parameter(torch.randn(feature_size, eb_dim))
            self.emb_mtx_mask = torch.cat(
                [torch.zeros(1, eb_dim), torch.ones(feature_size - 1, eb_dim)], dim=0
            )
            self.emb_mtx.data *= self.emb_mtx_mask

        self.target = None
        self.target_input = None

    def build_index_and_loss(self, probs):
        uniform = torch.rand_like(probs)
        condition = probs - uniform
        self.index = torch.where(condition >= 0, torch.ones_like(probs), torch.zeros_like(probs))
        log_probs = torch.log(torch.clamp(probs, 1e-10, 1))

        self.loss = -torch.mean(torch.sum(log_probs * self.index * self.rewards, dim=1))
        self.reward = torch.mean(self.rewards)

    def build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def train(self, batch_data, lr, rewards):
        self.optimizer.zero_grad()
        self.target_ph = torch.tensor(batch_data, dtype=torch.long)
        self.lr = lr
        self.rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)

        self.target = self.emb_mtx[self.target_ph]  # [B, F, EMB_DIM]
        self.target_input = self.target[:, 1:, :]  # exclude uid

        loss, reward = self.forward()
        loss.backward()
        self.optimizer.step()

        return loss.item(), reward.item()

    def get_distri(self, batch_data):
        self.target_ph = torch.tensor(batch_data, dtype=torch.long)
        self.target = self.emb_mtx[self.target_ph]  # [B, F, EMB_DIM]
        self.target_input = self.target[:, 1:, :]  # exclude uid

        probs = self.forward()
        return probs.detach().numpy()

    def get_index(self, batch_data):
        self.target_ph = torch.tensor(batch_data, dtype=torch.long)
        self.target = self.emb_mtx[self.target_ph]  # [B, F, EMB_DIM]
        self.target_input = self.target[:, 1:, :]  # exclude uid

        self.probs = self.build_select_probs(self.target_input)
        self.build_index_and_loss(self.probs)

        return self.index.detach().numpy()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def restore(self, path):
        self.load_state_dict(torch.load(path))
        print('model restored from {}'.format(path))


class UBR_SA(UBRBase):
    def __init__(self, feature_size, eb_dim, hidden_size, record_fnum, emb_initializer):
        super(UBR_SA, self).__init__(feature_size, eb_dim, hidden_size, record_fnum, emb_initializer)
        self.fc1 = nn.Linear(eb_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.fc3 = nn.Linear(10, 1)
        self.probs = None

    def build_select_probs(self, target_input):
        sa_target = self.multihead_attention(self.normalize(target_input), target_input)
        probs = F.relu(self.fc1(sa_target))
        probs = F.relu(self.fc2(probs))
        probs = torch.sigmoid(self.fc3(probs))
        probs = probs.view(-1, self.record_fnum - 1)
        return probs

    def multihead_attention(self, queries, keys, num_units=None, num_heads=2):
        if num_units is None:
            num_units = queries.size(-1)

        Q = self.fc(queries)
        K = self.fc(keys)
        V = self.fc(keys)

        Q_ = torch.cat(Q.split(num_units, dim=2), dim=0)
        K_ = torch.cat(K.split(num_units, dim=2), dim=0)
        V_ = torch.cat(V.split(num_units, dim=2), dim=0)

        outputs = torch.matmul(Q_, K_.transpose(1, 2))
        outputs = outputs / (K_.size(-1) ** 0.5)

        key_masks = torch.sign(torch.abs(keys.sum(dim=-1)))
        key_masks = key_masks.repeat(num_heads, 1)
        key_masks = key_masks.unsqueeze(1).repeat(1, queries.size(1), 1)

        paddings = torch.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = torch.where(key_masks.eq(0), paddings, outputs)

        outputs = F.softmax(outputs, dim=-1)

        query_masks = torch.sign(torch.abs(queries.sum(dim=-1)))
        query_masks = query_masks.repeat(num_heads, 1)
        query_masks = query_masks.unsqueeze(-1).repeat(1, 1, keys.size(1))
        outputs *= query_masks

        outputs = F.dropout(outputs, 0.8)

        outputs = torch.matmul(outputs, V_)
        outputs = torch.cat(outputs.split(outputs.size(0) // num_heads, dim=0), dim=2)
        outputs += queries

        return outputs

    def normalize(self, inputs, epsilon=1e-8):
        mean = inputs.mean(dim=-1, keepdim=True)
        variance = inputs.var(dim=-1, keepdim=True)
        normalized = (inputs - mean) / (torch.sqrt(variance + epsilon))
        outputs = self.gamma * normalized + self.beta

        return outputs
