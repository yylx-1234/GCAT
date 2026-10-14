"""Cross-attention GRU with a lightweight complementary channel gate."""

import torch
from torch import nn

from models.cross_attention_gru import CrossAttentionGRUClassifier


class GatedCrossAttentionGRUClassifier(CrossAttentionGRUClassifier):
    """Add only complementary gated fusion to the frozen CA architecture."""

    def __init__(
        self,
        s2_input_dim=20,
        s1_input_dim=6,
        projection_dim=128,
        hidden_dim=128,
        gru_layers=1,
        bidirectional=False,
        attention=None,
        gate=None,
        fusion_dropout=0.2,
        num_classes=18,
    ):
        gate = dict(gate or {})
        gate_type = gate.get("type", "complementary_channel")
        gate_input_dim = int(gate.get("input_dim", hidden_dim * 2))
        gate_output_dim = int(gate.get("output_dim", hidden_dim))
        if gate_type != "complementary_channel":
            raise ValueError("GCAT requires gate.type=complementary_channel")
        if gate_input_dim != int(hidden_dim) * 2:
            raise ValueError("gate.input_dim must equal 2 * hidden_dim")
        if gate_output_dim != int(hidden_dim):
            raise ValueError("gate.output_dim must equal hidden_dim")
        super().__init__(
            s2_input_dim=s2_input_dim,
            s1_input_dim=s1_input_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            gru_layers=gru_layers,
            bidirectional=bidirectional,
            attention=attention,
            fusion_dropout=fusion_dropout,
            num_classes=num_classes,
        )
        self.gate_projection = nn.Linear(gate_input_dim, gate_output_dim)

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
        features = super().forward_features(
            s2,
            s2_lengths,
            s1a,
            s1a_lengths,
            s1d,
            s1d_lengths,
            return_attention=return_attention,
        )
        z_s2 = features["z_s2"]
        z_sar = features["z_sar"]
        gate_input = torch.cat((z_s2, z_sar), dim=1)
        gate = torch.sigmoid(self.gate_projection(gate_input))
        gated_s2 = gate * z_s2
        gated_sar = (1.0 - gate) * z_sar
        fusion = torch.cat((gated_s2, gated_sar), dim=1)
        features.update(
            {
                "gate": gate,
                "gated_s2": gated_s2,
                "gated_sar": gated_sar,
                "fusion": fusion,
            }
        )
        return features

    def forward(
        self,
        s2,
        s2_lengths,
        s1a,
        s1a_lengths,
        s1d,
        s1d_lengths,
        return_gate=False,
    ):
        features = self.forward_features(
            s2,
            s2_lengths,
            s1a,
            s1a_lengths,
            s1d,
            s1d_lengths,
            return_attention=False,
        )
        logits = self.classifier(self.dropout(features["fusion"]))
        if return_gate:
            return logits, features["gate"]
        return logits
