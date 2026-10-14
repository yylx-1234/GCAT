"""Frozen GCAT backbone with fixed real-DOY temporal encoding."""

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.gated_cross_attention_gru import GatedCrossAttentionGRUClassifier
from models.time_encoding import SinusoidalDOYEncoding


class GCATRPTClassifier(GatedCrossAttentionGRUClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        projection_dim = int(self.s2_projection.out_features)
        hidden_dim = int(self.s2_gru.hidden_size)
        if projection_dim != hidden_dim:
            raise ValueError("Temporal addition requires projection_dim == hidden_dim")
        self.time_encoding = SinusoidalDOYEncoding(projection_dim)
        self.uses_dates = True
        self.uses_quality = False

    def _s2_extra_embedding(self, s2_quality):
        if s2_quality is not None:
            raise ValueError("GCAT-RP-T must not receive S2 quality")
        return None

    def _encode_temporal(
        self, sequence, dates, lengths, projection, gru, name, extra_embedding=None
    ):
        if sequence.ndim != 3 or dates.ndim != 2:
            raise ValueError("{} sequence/dates shapes are invalid".format(name))
        if sequence.shape[:2] != dates.shape:
            raise ValueError("{} sequence/date dimensions differ".format(name))
        padding_mask = self._padding_mask(lengths, sequence.shape[1], sequence.device, name)
        projected = projection(sequence) + self.time_encoding(dates)
        if extra_embedding is not None:
            if extra_embedding.shape != projected.shape:
                raise ValueError("{} extra embedding shape differs".format(name))
            projected = projected + extra_embedding
        packed = pack_padded_sequence(
            projected,
            lengths.detach().to(device="cpu", dtype=torch.long),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, hidden = gru(packed)
        tokens, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=int(sequence.shape[1])
        )
        return tokens, hidden[-1], padding_mask

    def _cross_features(
        self,
        s2,
        s2_lengths,
        s2_dates,
        s1a,
        s1a_lengths,
        s1a_dates,
        s1d,
        s1d_lengths,
        s1d_dates,
        s2_quality=None,
        return_attention=False,
    ):
        extra = self._s2_extra_embedding(s2_quality)
        h_s2_tokens, h_s2, s2_padding_mask = self._encode_temporal(
            s2, s2_dates, s2_lengths, self.s2_projection, self.s2_gru, "s2", extra
        )
        h_a_tokens, h_a, s1a_padding_mask = self._encode_temporal(
            s1a, s1a_dates, s1a_lengths, self.sar_projection, self.sar_gru, "s1a"
        )
        h_d_tokens, h_d, s1d_padding_mask = self._encode_temporal(
            s1d, s1d_dates, s1d_lengths, self.sar_projection, self.sar_gru, "s1d"
        )
        h_sar = (h_a + h_d) / 2.0
        h_sar_tokens = torch.cat((h_a_tokens, h_d_tokens), dim=1)
        sar_padding_mask = torch.cat((s1a_padding_mask, s1d_padding_mask), dim=1)
        cross_sar_to_s2, attention_sar_to_s2 = self.s2_queries_sar(
            query=h_s2.unsqueeze(1), key=h_sar_tokens, value=h_sar_tokens,
            key_padding_mask=sar_padding_mask, need_weights=bool(return_attention),
            average_attn_weights=False,
        )
        cross_s2_to_sar, attention_s2_to_sar = self.sar_queries_s2(
            query=h_sar.unsqueeze(1), key=h_s2_tokens, value=h_s2_tokens,
            key_padding_mask=s2_padding_mask, need_weights=bool(return_attention),
            average_attn_weights=False,
        )
        cross_sar_to_s2 = cross_sar_to_s2.squeeze(1)
        cross_s2_to_sar = cross_s2_to_sar.squeeze(1)
        z_s2 = self.s2_norm(h_s2 + cross_sar_to_s2)
        z_sar = self.sar_norm(h_sar + cross_s2_to_sar)
        return {
            "H_s2": h_s2_tokens, "H_sar": h_sar_tokens,
            "h_s2": h_s2, "h_sar": h_sar,
            "z_s2": z_s2, "z_sar": z_sar,
            "s2_padding_mask": s2_padding_mask, "sar_padding_mask": sar_padding_mask,
            "attention_sar_to_s2": attention_sar_to_s2,
            "attention_s2_to_sar": attention_s2_to_sar,
        }

    def _apply_gate(self, features, quality_summary=None):
        if quality_summary is not None:
            raise ValueError("GCAT-RP-T gate must not receive quality summary")
        z_s2, z_sar = features["z_s2"], features["z_sar"]
        gate = torch.sigmoid(self.gate_projection(torch.cat((z_s2, z_sar), dim=1)))
        fusion = torch.cat((gate * z_s2, (1.0 - gate) * z_sar), dim=1)
        features.update({"gate": gate, "fusion": fusion})
        return features

    def forward_features(self, *args, **kwargs):
        return self._apply_gate(self._cross_features(*args, **kwargs))

    def forward(
        self, s2, s2_lengths, s2_dates, s1a, s1a_lengths, s1a_dates,
        s1d, s1d_lengths, s1d_dates, return_gate=False,
    ):
        features = self.forward_features(
            s2, s2_lengths, s2_dates, s1a, s1a_lengths, s1a_dates,
            s1d, s1d_lengths, s1d_dates,
        )
        logits = self.classifier(self.dropout(features["fusion"]))
        return (logits, features["gate"]) if return_gate else logits

