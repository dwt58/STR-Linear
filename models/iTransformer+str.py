import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


# ============================================================

# ============================================================
class STR(nn.Module):
    """
    多尺度通道精炼器：以残差方式修正初步预测中的跨通道依赖误差。
    """
    def __init__(self, num_nodes, seq_len, refiner_dim=256):
        super(STR, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.refiner_dim = refiner_dim

        self.node_encoder = nn.Linear(seq_len, refiner_dim)
        self.channel_adjust = nn.Conv1d(refiner_dim, refiner_dim, kernel_size=1)

        self.dilation_rates = [1, 2, 4, 8]
        self.dilated_convs = nn.ModuleList()
        for i, dilation in enumerate(self.dilation_rates):
            conv_block = nn.Sequential(
                nn.Conv1d(refiner_dim, refiner_dim, kernel_size=3,
                          padding=dilation, dilation=dilation,
                          padding_mode='replicate',
                          groups=min(refiner_dim, 8)),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm1d(refiner_dim),
                nn.Dropout(0.03 + 0.02 * i)
            )
            self.dilated_convs.append(conv_block)

        self.adaptive_weights = nn.Parameter(torch.ones(len(self.dilation_rates)))

        self.feature_recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(refiner_dim, refiner_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(refiner_dim // 4, refiner_dim, 1),
            nn.Sigmoid()
        )

        self.output_conv = nn.Conv1d(refiner_dim, seq_len, kernel_size=1)

    def forward(self, x):
        # x: (batch, seq_len, num_nodes)
        original_input = x

        x = x.permute(0, 2, 1)          # (B, N, L)
        x = self.node_encoder(x)        # (B, N, H)
        x = x.permute(0, 2, 1)          # (B, H, N)
        x = self.channel_adjust(x)

        temporal_features = []
        current_x = x
        for i, dilated_conv in enumerate(self.dilated_convs):
            conv_output = dilated_conv(current_x)
            weighted_output = self.adaptive_weights[i] * conv_output
            temporal_features.append(weighted_output)
            current_x = conv_output

        if len(temporal_features) > 1:
            normalized_weights = torch.softmax(self.adaptive_weights, dim=0)
            fused = sum(normalized_weights[i] * temporal_features[i]
                        for i in range(len(temporal_features)))
        else:
            fused = temporal_features[0]

        attention_weights = self.feature_recalibrate(fused)
        recalibrated = fused * attention_weights
        output = self.output_conv(recalibrated)   # (B, seq_len, N)

        return original_input + output


# ============================================================
# iTransformer 模型，直接集成 STR 
# ============================================================
class Model(nn.Module):
    """
    iTransformer with built-in MultiScaleChannelRefiner.
    The refiner is applied to the preliminary forecast before denormalization,
    providing cross-channel error correction.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.class_strategy = configs.class_strategy

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # ---- 后置多尺度通道修正模块（直接加入） ----
        self.output_refiner = STR(
            num_nodes=configs.enc_in,          # N 个变量
            seq_len=self.pred_len,             # 作用于预测输出
            refiner_dim=getattr(configs, 'refiner_dim', 256)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Projector: (B, N, d_model) -> (B, N, pred_len) -> (B, pred_len, N)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        # ---- 后置精炼：在反归一化之前直接应用 STR ----
        dec_out = self.output_refiner(dec_out)

        if self.use_norm:
            # De-Normalization
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]   # [B, pred_len, N]