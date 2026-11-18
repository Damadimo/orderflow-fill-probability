"""Neural survival models. Both output logits of the discrete hazards h_k for the 20 time bins."""

from __future__ import annotations

import torch
from torch import nn

N_EVENT_NUM = 7    # numeric event-token features (all columns except the type id)
N_QUEUE_NUM = 5    # numeric queue-token features (last column is the "present" flag)


class OrderFlowTransformer(nn.Module):
    """Event-level model: [context] + queue tokens + event tokens -> Transformer -> hazard head.

    * context token: MLP over the standardized hand-built features (incl. V's side),
    * queue tokens: the orders queued ahead of V (size, age, rank, volume ahead, partially filled),
    * event tokens: the last K near-touch order events, expressed relative to V.
    """

    def __init__(self, n_global: int, k_events: int = 256, d: int = 128, layers: int = 4, heads: int = 8,
                 ffn: int = 512, n_bins: int = 20, dropout: float = 0.1):
        super().__init__()
        self.type_emb = nn.Embedding(7, d)                     # 0 = padding, 1..6 = event types
        self.ev_proj = nn.Sequential(nn.Linear(N_EVENT_NUM, d), nn.GELU(), nn.Linear(d, d))
        self.q_proj = nn.Sequential(nn.Linear(N_QUEUE_NUM, d), nn.GELU(), nn.Linear(d, d))
        self.g_proj = nn.Sequential(nn.Linear(n_global, d), nn.GELU(), nn.Linear(d, d))
        self.segment = nn.Embedding(3, d)                      # context / queue / event
        self.pos_ev = nn.Parameter(torch.zeros(k_events, d))   # recency position of each event token
        nn.init.normal_(self.pos_ev, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ffn, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, n_bins))
        # masked-event pretraining
        self.mask_emb = nn.Parameter(torch.zeros(d))
        self.pre_type = nn.Linear(d, 6)
        self.pre_reg = nn.Linear(d, 3)                         # same_side, price offset, log size
        self.register_buffer("ev_mu", torch.zeros(N_EVENT_NUM))
        self.register_buffer("ev_sd", torch.ones(N_EVENT_NUM))
        self.register_buffer("q_mu", torch.zeros(N_QUEUE_NUM))
        self.register_buffer("q_sd", torch.ones(N_QUEUE_NUM))

    def set_token_stats(self, ev_mu, ev_sd, q_mu, q_sd):
        for name, v in (("ev_mu", ev_mu), ("ev_sd", ev_sd), ("q_mu", q_mu), ("q_sd", q_sd)):
            getattr(self, name).copy_(torch.as_tensor(v, dtype=torch.float32))

    def encode(self, ev, q, g, mask_ev=None):
        typ = ev[..., 0].long()
        e = self.type_emb(typ) + self.ev_proj((ev[..., 1:].float() - self.ev_mu) / self.ev_sd)
        if mask_ev is not None:
            e = torch.where(mask_ev[..., None], self.mask_emb.expand_as(e), e)
        e = e + self.pos_ev + self.segment.weight[2]
        qq = self.q_proj((q[..., :N_QUEUE_NUM].float() - self.q_mu) / self.q_sd) + self.segment.weight[1]
        ctx = self.g_proj(g)[:, None] + self.segment.weight[0]
        x = torch.cat([ctx, qq, e], 1)
        pad = torch.cat([torch.zeros_like(typ[:, :1], dtype=torch.bool), q[..., N_QUEUE_NUM] == 0, typ == 0], 1)
        return self.norm(self.encoder(x, src_key_padding_mask=pad))

    def forward(self, ev, q, g):
        return self.head(self.encode(ev, q, g)[:, 0])

    def pretrain_loss(self, ev, q, g, mask_rate: float = 0.15):
        """Masked-event modelling: hide 15% of real event tokens and predict their type, side, price, size."""
        real = ev[..., 0] > 0
        mask = (torch.rand(real.shape, device=ev.device) < mask_rate) & real
        h = self.encode(ev, q, g, mask_ev=mask)[:, 1 + q.shape[1]:]
        hm = h[mask]
        tgt_type = ev[..., 0][mask].long() - 1
        tgt_reg = ((ev[..., 1:4].float() - self.ev_mu[:3]) / self.ev_sd[:3])[mask]
        loss_t = nn.functional.cross_entropy(self.pre_type(hm), tgt_type)
        loss_r = nn.functional.mse_loss(self.pre_reg(hm), tgt_reg)
        acc = (self.pre_type(hm).argmax(-1) == tgt_type).float().mean()
        return loss_t + loss_r, {"mlm_type_loss": loss_t.item(), "mlm_reg_loss": loss_r.item(), "mlm_type_acc": acc.item()}


class SnapshotConvTransformer(nn.Module):
    """Snapshot baseline (B4, after Arroyo et al. 2024): top-10 L2 levels for the last T seconds.

    A 1-D convolution over price levels encodes each snapshot; a Transformer runs over time.
    """

    def __init__(self, n_global: int, t_hist: int = 60, levels: int = 10, d: int = 128, layers: int = 4,
                 heads: int = 8, ffn: int = 512, n_bins: int = 20, dropout: float = 0.1):
        super().__init__()
        self.levels = levels
        self.conv = nn.Sequential(nn.Conv1d(4, 32, 3, padding=1), nn.GELU(), nn.Conv1d(32, 32, 3, padding=1), nn.GELU())
        self.snap_proj = nn.Linear(32 * levels, d)
        self.g_proj = nn.Sequential(nn.Linear(n_global, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(t_hist + 1, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ffn, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, n_bins))
        self.register_buffer("s_mu", torch.zeros(40))
        self.register_buffer("s_sd", torch.ones(40))

    def set_snap_stats(self, mu, sd):
        self.s_mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
        self.s_sd.copy_(torch.as_tensor(sd, dtype=torch.float32))

    def forward(self, snaps, g):
        b, t, _ = snaps.shape
        x = (snaps.float() - self.s_mu) / self.s_sd                 # [B, T, 40] = side(2) x level(10) x (offset, log vol)
        x = x.view(b * t, 2, self.levels, 2).permute(0, 1, 3, 2).reshape(b * t, 4, self.levels)
        x = self.snap_proj(self.conv(x).reshape(b * t, -1)).view(b, t, -1)
        x = torch.cat([self.g_proj(g)[:, None], x], 1) + self.pos
        return self.head(self.norm(self.encoder(x))[:, 0])


def n_params(model: nn.Module, exclude_pretrain: bool = True) -> int:
    skip = ("mask_emb", "pre_type", "pre_reg") if exclude_pretrain else ()
    return sum(p.numel() for n, p in model.named_parameters() if not n.startswith(skip))
