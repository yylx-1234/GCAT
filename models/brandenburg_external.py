"""The published GCAT backbone with Brandenburg's 4-channel SAR and 9 classes."""

from models.gcat_rp_t import GCATRPTClassifier


class BrandenburgGCATRPTClassifier(GCATRPTClassifier):
    """Input/output adaptation only; the temporal and fusion blocks are unchanged."""

    def __init__(self):
        super().__init__(
            s2_input_dim=20,
            s1_input_dim=4,
            projection_dim=128,
            hidden_dim=128,
            gru_layers=1,
            bidirectional=False,
            attention={"embed_dim": 128, "num_heads": 4, "dropout": 0.1},
            gate={"type": "complementary_channel", "input_dim": 256, "output_dim": 128},
            fusion_dropout=0.2,
            num_classes=9,
        )
