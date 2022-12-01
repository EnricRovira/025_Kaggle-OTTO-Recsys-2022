import tensorflow as tf
import numpy as np

class EmbeddingTransposed(tf.keras.layers.Layer):
    def __init__(self, tied_to=None, activation=None, **kwargs):
        super(EmbeddingTransposed, self).__init__(**kwargs)
        self.tied_to = tied_to
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.custom_weights = self.tied_to.weights[0]
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.tied_to.weights[0].shape[0]

    def call(self, inputs, mask=None):
        output = tf.keras.backend.dot(inputs, tf.keras.backend.transpose(self.custom_weights))
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {'activation': tf.keras.activations.serialize(self.activation)}
        base_config = super(EmbeddingTransposed, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EncoderTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, attention_axes=None, drop_rate=0.1, att_drop_rate=0.1):
        super(EncoderTransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, attention_axes=attention_axes, dropout=att_drop_rate)
        self.ffn = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation='gelu'), 
             tf.keras.layers.Dense(embed_dim)]
        )
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, query, key, training, attention_mask=None):
        attn_output = self.att(query, key, attention_mask=attention_mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        
        out1 = self.layernorm1(query + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        return self.layernorm2(out1 + ffn_output)
      

def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = positional_encoding(length=seq_len, depth=d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        pos_encoding = self.pos_encoding[tf.newaxis, :length, :]
        return pos_encoding   


class ModelBert4Rec(tf.keras.models.Model):
    def __init__(self, num_items, model_cfg):
        super(ModelBert4Rec, self).__init__()
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        self.num_items = num_items
        self.model_cfg = model_cfg
        self.scaler = tf.math.sqrt(tf.constant(self.model_cfg.trf_dim, tf.float32))
        self.std_init = np.sqrt(1/(model_cfg.emb_dim*3)).round(6) #0.02 if model_cfg.trf_dim < 1024 else 
        self.pos_embed = PositionalEmbedding(model_cfg.trf_dim, model_cfg.seq_len)
        self.embed_items = tf.keras.layers.Embedding(
            num_items, model_cfg.emb_dim, 
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(mean=0, stddev=self.std_init)
        )
        self.embed_type = tf.keras.layers.Embedding(
            3+1, 
            model_cfg.emb_dim,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(mean=0, stddev=self.std_init)
        )
        self.mlp_proj_time_encoding = tf.keras.models.Sequential([
           tf.keras.layers.Dropout(model_cfg.drop_rate), 
           tf.keras.layers.Dense(model_cfg.trf_dim, kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0, stddev=self.std_init)),
           tf.keras.layers.LayerNormalization(epsilon=1e-6)
        ])
        # self.mlp_proj = tf.keras.models.Sequential([
        #    tf.keras.layers.Dropout(model_cfg.drop_rate), 
        #    tf.keras.layers.Dense(model_cfg.trf_dim, kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0, stddev=self.std_init)),
        #    tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # ])
        # self.mlp_proj_conts = tf.keras.models.Sequential([
        #    tf.keras.layers.Dropout(model_cfg.drop_rate), 
        #    tf.keras.layers.Dense(model_cfg.trf_dim, kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0, stddev=self.std_init)),
        #    tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # ])
        self.list_transformer_block = [EncoderTransformerBlock(model_cfg.trf_dim, model_cfg.num_heads, 
                                                               model_cfg.ff_dim, attention_axes=None, 
                                                               drop_rate=model_cfg.drop_rate, 
                                                               att_drop_rate=model_cfg.att_drop_rate) 
                                       for _ in range(model_cfg.num_layers)]
        self.pred_layer = EmbeddingTransposed(tied_to=self.embed_items, activation='linear', dtype='float32')

        
    def call(self, inputs, training=True):
        x_seq_past, x_seq_type, x_seq_encoding, x_seq_recency = inputs
        pad_mask = tf.cast(tf.where(tf.equal(x_seq_type[:, :, 0], 0), 0, 1), tf.float32)[:, tf.newaxis, tf.newaxis, :]
        if self.model_cfg.model_arch == 'sasrec':
            mask = self.create_masks(x_seq_past, pad_mask)
        else:
            mask = pad_mask
        ###########
        x_pos_embed = self.pos_embed(x_seq_past[:, :, 0])
        x_seq_past_items = self.embed_items(x_seq_past[:, :, 0])
        x_seq_past_type = self.embed_type(x_seq_type[:, :, 0])
        x_seq_time_encoding = self.mlp_proj_time_encoding(x_seq_encoding, training=training)
        # x = tf.concat([x_seq_past_items, x_seq_past_type, x_seq_encoding], axis=-1)
        # x = self.mlp_proj(x, training=training)
        # x_seq_recency = self.mlp_proj_conts(x_seq_recency, training=training)
        x_ones = tf.ones(tf.shape(x_seq_past_items))
        ########### 
        # x = x_seq_past_items * (x_ones + x_seq_past_type + x_seq_time_encoding + x_pos_embed)
        x = x_seq_past_items * (x_ones + x_seq_past_type + x_seq_time_encoding + x_pos_embed)
        for i in range(len(self.list_transformer_block)):
            x = self.list_transformer_block[i](x, x, training=training, attention_mask=mask)
        probs = self.pred_layer(x)
        return probs

    def create_masks(self, x_seq_past, pad_mask):   
        size = self.model_cfg.seq_len
        look_ahead_mask = tf.linalg.band_part(tf.ones((1, size, size), tf.float32), -1, 0) * pad_mask[:, 0, 0, :, tf.newaxis]
        return look_ahead_mask#, pad_mask
      

def build_model_bert4Rec(num_items, model_cfg):
    return ModelBert4Rec(num_items, model_cfg)