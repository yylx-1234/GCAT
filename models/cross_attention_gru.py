"""Lightweight bidirectional cross-attention GRU baseline."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CrossAttentionGRUClassifier(nn.Module):
    """Use each modality summary to query the other modality's GRU tokens."""

    def __init__(
        self,
        s2_input_dim=20,
        s1_input_dim=6,
        projection_dim=128,
        hidden_dim=128,
        gru_layers=1,
        bidirectional=False,
        attention=None,
        fusion_dropout=0.2,
        num_classes=18,
    ):
        super().__init__()
        if bidirectional:
            raise ValueError("Cross-Attention baseline must use unidirectional GRUs")
        if int(gru_layers) != 1:
            raise ValueError("Cross-Attention baseline requires gru_layers=1")
        attention = dict(attention or {})
        embed_dim = int(attention.get("embed_dim", hidden_dim))
        if embed_dim != int(hidden_dim):
            raise ValueError("attention.embed_dim must equal hidden_dim")
        num_heads = int(attention.get("num_heads", 4))
        attention_dropout = float(attention.get("dropout", 0.1))

        self.s2_projection = nn.Linear(int(s2_input_dim), int(projection_dim))
        self.s2_gru = nn.GRU(
            input_size=int(projection_dim),
            hidden_size=int(hidden_dim),
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )
        # S1 ascending and descending sequences use these exact shared modules.
        self.sar_projection = nn.Linear(int(s1_input_dim), int(projection_dim))
        self.sar_gru = nn.GRU(
            input_size=int(projection_dim),
            hidden_size=int(hidden_dim),
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )
        self.s2_queries_sar = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.sar_queries_s2 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.s2_norm = nn.LayerNorm(embed_dim)
        self.sar_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(float(fusion_dropout))
        self.classifier = nn.Linear(embed_dim * 2, int(num_classes))

    @staticmethod
    def _padding_mask(lengths, maximum, device, name):
        if lengths.ndim != 1:
            raise ValueError("{}_lengths must have shape [B]".format(name))
        lengths_device = lengths.detach().to(device=device, dtype=torch.long)
        if torch.any(lengths_device <= 0) or torch.any(lengths_device > int(maximum)):
            raise ValueError("{} lengths must be within [1,T]".format(name))
        positions = torch.arange(int(maximum), device=device).unsqueeze(0)
        return positions >= lengths_device.unsqueeze(1)

    @classmethod
    def _encode(cls, sequence, lengths, projection, gru, name):
        if sequence.ndim != 3:
            raise ValueError("{} must have shape [B,T,F]".format(name))
        if lengths.ndim != 1 or lengths.shape[0] != sequence.shape[0]:
            raise ValueError("{}_lengths must have shape [B]".format(name))
        padding_mask = cls._padding_mask(
            lengths, sequence.shape[1], sequence.device, name
        )
        projected = projection(sequence)
        packed = pack_padded_sequence(
            projected,
            lengths.detach().to(device="cpu", dtype=torch.long),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, hidden = gru(packed)
        sequence_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=int(sequence.shape[1]),
        )
        return sequence_output, hidden[-1], padding_mask

    def forward_features(
        self,
        s2,
        s2_lengths,
        s1a,
        s1a_lengths,
        s1d,
        s1d_lengths,
        return_attention=False,
    ):
        h_s2_tokens, h_s2, s2_padding_mask = self._encode(
            s2, s2_lengths, self.s2_projection, self.s2_gru, "s2"
        )
        h_a_tokens, h_a, s1a_padding_mask = self._encode(
            s1a, s1a_lengths, self.sar_projection, self.sar_gru, "s1a"
        )
        h_d_tokens, h_d, s1d_padding_mask = self._encode(
            s1d, s1d_lengths, self.sar_projection, self.sar_gru, "s1d"
        )
        h_sar = (h_a + h_d) / 2.0
        h_sar_tokens = torch.cat((h_a_tokens, h_d_tokens), dim=1)
        sar_padding_mask = torch.cat(
            (s1a_padding_mask, s1d_padding_mask), dim=1
        )

        cross_sar_to_s2, attention_sar_to_s2 = self.s2_queries_sar(
            query=h_s2.unsqueeze(1),
            key=h_sar_tokens,
            value=h_sar_tokens,
            key_padding_mask=sar_padding_mask,
            need_weights=bool(return_attention),
            average_attn_weights=False,
        )
        cross_s2_to_sar, attention_s2_to_sar = self.sar_queries_s2(
            query=h_sar.unsqueeze(1),
            key=h_s2_tokens,
            value=h_s2_tokens,
            key_padding_mask=s2_padding_mask,
            need_weights=bool(return_attention),
            average_attn_weights=False,
        )
        cross_sar_to_s2 = cross_sar_to_s2.squeeze(1)
        cross_s2_to_sar = cross_s2_to_sar.squeeze(1)
        z_s2 = self.s2_norm(h_s2 + cross_sar_to_s2)
        z_sar = self.sar_norm(h_sar + cross_s2_to_sar)
        fusion = torch.cat((z_s2, z_sar), dim=1)
        return {
            "H_s2": h_s2_tokens,
            "H_sar": h_sar_tokens,
            "h_s2": h_s2,
            "h_sar": h_sar,
            "cross_sar_to_s2": cross_sar_to_s2,
            "cross_s2_to_sar": cross_s2_to_sar,
            "z_s2": z_s2,
            "z_sar": z_sar,
            "fusion": fusion,
            "s2_padding_mask": s2_padding_mask,
            "sar_padding_mask": sar_padding_mask,
            "attention_sar_to_s2": attention_sar_to_s2,
            "attention_s2_to_sar": attention_s2_to_sar,
        }

    def forward(
        self,
        s2,
        s2_lengths,
        s1a,
        s1a_lengths,
        s1d,
        s1d_lengths,
        return_attention=False,
    ):
        features = self.forward_features(
            s2,
            s2_lengths,
            s1a,
            s1a_lengths,
            s1d,
            s1d_lengths,
            return_attention=return_attention,
        )
        logits = self.classifier(self.dropout(features["fusion"]))
        if return_attention:
            return logits, {
                "sar_to_s2": features["attention_sar_to_s2"],
                "s2_to_sar": features["attention_s2_to_sar"],
            }
        return logits
