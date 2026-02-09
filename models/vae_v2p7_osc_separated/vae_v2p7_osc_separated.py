import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.layers import Layer, Input, Dense, Concatenate, Dropout, BatchNormalization, LayerNormalization, LeakyReLU, Activation, Add, Multiply, Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import register_keras_serializable
from keras.saving import serialize_keras_object, deserialize_keras_object

# ==============================================================================
# 1. SHARED LAYERS
# ==============================================================================

@register_keras_serializable(package="custom", name="FiLMLayer")
class FiLMLayer(Layer):
    def call(self, inputs):
        x, gamma, beta = inputs
        return x * gamma + beta

@register_keras_serializable(package="custom", name="StopGradient")
class StopGradient(Layer):
    """
    A permanent Keras layer to stop gradient flow.
    Safe for saving/loading because it doesn't rely on lambda serialization.
    """
    def call(self, inputs):
        return tf.stop_gradient(inputs)

@register_keras_serializable(package="custom", name="SafeStopGradientLambda")
class SafeStopGradientLambda(Layer):
    """
    A safe replacement for Lambda(lambda x: tf.stop_gradient(x)) that avoids serialization issues.
    """
    def call(self, x):
        return tf.stop_gradient(x)
        
    def get_config(self):
        return super().get_config()

@register_keras_serializable(package="custom", name="SplitLatentLayer")
class SplitLatentLayer(Layer):
    """
    A Custom Layer to split a concatenated latent vector into two parts.
    Replaces Lambda(lambda x: x[:, :split]) for better serialization.
    """
    def __init__(self, split_index, part="first", **kwargs):
        super().__init__(**kwargs)
        self.split_index = split_index
        self.part = part

    def call(self, inputs):
        if self.part == "first":
            return inputs[:, :self.split_index]
        return inputs[:, self.split_index:]

    def get_config(self):
        config = super().get_config()
        config.update({"split_index": self.split_index, "part": self.part})
        return config

# ==============================================================================
# 2. VAE MODEL (Stage 1: Compression)
# ==============================================================================

def build_encoder(embedding_dim, num_params, audio_dim, latent_dim_params, latent_dim_audio, enc_width, dropout_rate=0.0):
    """
    Builds q(z_p, z_a | params, audio, text)
    """
    text_in = Input(shape=(embedding_dim,), name="encoder_text_input")
    params_in = Input(shape=(num_params,), name="encoder_params_input")
    audio_in = Input(shape=(audio_dim,), name="encoder_audio_input")

    x = Concatenate(name="encoder_concat")([text_in, params_in, audio_in])
    x = Dense(enc_width, activation="linear")(x)
    x = LayerNormalization()(x)
    x = Activation("relu")(x)
    x = Dense(enc_width, activation="linear")(x)
    x = LayerNormalization()(x)
    x = Activation("relu")(x)

    # HEAD 1: PARAMETERS
    x_p = Dense(enc_width // 2, activation="relu", name="pre_latent_params")(x)
    zp_mu = Dense(latent_dim_params, name="zp_mean")(x_p)
    zp_log = Dense(latent_dim_params, name="zp_log_var")(x_p)

    # HEAD 2: AUDIO
    x_a = Dense(enc_width // 2, activation="relu", name="pre_latent_audio")(x)
    za_mu = Dense(latent_dim_audio, name="za_mean")(x_a)
    za_log = Dense(latent_dim_audio, name="za_log_var")(x_a)

    return Model([text_in, params_in, audio_in], [zp_mu, zp_log, za_mu, za_log], name="encoder")

def build_decoder_film(embedding_dim, latent_dim_total, latent_dim_params, dec_width,
                       unipolar_indices, bipolar_indices, bool_indices, cat_indices, categorical_num_classes, dropout_rate=0.0):
    """
    Builds p(params, audio | z, text)
    Splits Concatenated Z back into P and A using Slicing.
    """
    z_in = Input(shape=(latent_dim_total,), name="latent_input")
    text_in = Input(shape=(embedding_dim,), name="decoder_text_input")

    # FIXED: Replaced raw tf.split with robust slicing layers to avoid KerasTensor error
    # Note: Using Lambda layers here as per notebook source.
    z_p = tf.keras.layers.Lambda(lambda x: x[:, :latent_dim_params], name="slice_params")(z_in)
    z_a = tf.keras.layers.Lambda(lambda x: x[:, latent_dim_params:], name="slice_audio")(z_in)

    def film_block(x, text_in, width, block_id):
        x = LayerNormalization(name=f"dec_ln_pre_{block_id}")(x)
        gamma = Dense(width, kernel_initializer="zeros", bias_initializer="ones")(text_in)
        beta = Dense(width, kernel_initializer="zeros", bias_initializer="zeros")(text_in)
        x = FiLMLayer(name=f"film_{block_id}")([x, gamma, beta])
        x = LeakyReLU(negative_slope=0.2)(x)
        return x

    # PATH A: PARAMETERS
    x = Dense(dec_width)(z_p)
    x = film_block(x, text_in, dec_width, "params_1")
    x = Dense(dec_width)(x)
    x = film_block(x, text_in, dec_width, "params_2")
    x = LayerNormalization()(x)
    x = Activation("relu")(x)

    outs = []
    if unipolar_indices: outs.append(Dense(len(unipolar_indices), activation="sigmoid", name="unipolar_outputs")(x))
    if bipolar_indices:
        outs.append(Dense(len(bipolar_indices), activation="linear", name="bipolar_gate")(x))
        outs.append(Dense(len(bipolar_indices), activation="tanh", name="bipolar_value")(x))
    if bool_indices: outs.append(Dense(len(bool_indices), activation="sigmoid", name="boolean_outputs")(x))
    for j in sorted(cat_indices): outs.append(Dense(categorical_num_classes[int(j)], activation="softmax", name=f"cat_{j}")(x))

    # PATH B: AUDIO
    x_aud = Dense(dec_width)(z_a)
    x_aud = film_block(x_aud, text_in, dec_width, "audio_1")
    x_aud = Dense(dec_width)(x_aud)
    x_aud = film_block(x_aud, text_in, dec_width, "audio_2")
    x_aud = LayerNormalization()(x_aud)
    x_aud = Activation("relu")(x_aud)

    outs.append(Dense(512, activation="linear", name="osc_a_embed")(x_aud))
    outs.append(Dense(512, activation="linear", name="osc_b_embed")(x_aud))
    outs.append(Dense(512, activation="linear", name="osc_n_embed")(x_aud))

    return Model([z_in, text_in], outs, name="decoder_film")

@register_keras_serializable(package="custom", name="VAE_Text_to_Synth_Audio")
class VAE_Text_to_Synth_Audio(tf.keras.Model):
    def __init__(self, encoder, decoder, unipolar_indices, bipolar_indices,
                 bool_indices, cat_indices, categorical_num_classes, group_masking_map,
                 latent_dim_params, latent_dim_audio, latent_dim_total, # NEW: Explicit split dimensions
                 beta=1.0, latent_dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.unipolar_indices = [int(i) for i in unipolar_indices]
        self.bipolar_indices = [int(i) for i in bipolar_indices]
        self.bool_indices = [int(i) for i in bool_indices]
        self.cat_indices = [int(i) for i in cat_indices]
        self.categorical_num_classes = {int(k): int(v) for k, v in categorical_num_classes.items()}
        self.group_masking_map = {int(k): [int(x) for x in v] for k, v in (group_masking_map or {}).items()}
        self.param_to_enable = {int(pid): int(eid) for eid, mids in self.group_masking_map.items() for pid in mids if int(pid) != int(eid)}

        # Store dimensions explicitly
        self.latent_dim_params = int(latent_dim_params)
        self.latent_dim_audio = int(latent_dim_audio)
        self.latent_dim_total = int(latent_dim_total)

        self.beta = float(beta)
        self.latent_dropout_rate = float(latent_dropout_rate)

        # Metrics are not needed for inference-only deployment but kept for compatibility
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        
    def call(self, inputs, training=False):
        # inputs expected: [text, params, audio]
        text_embeddings, params_in, audio_in = inputs

        # 1. ENCODE (Returns 4 tensors now)
        zp_mu, zp_log, za_mu, za_log = self.encoder([text_embeddings, params_in, audio_in])

        # 2. SAMPLE PARAMS
        eps_p = tf.random.normal(shape=tf.shape(zp_mu))
        zp = zp_mu + tf.exp(0.5 * zp_log) * eps_p

        # 3. SAMPLE AUDIO
        eps_a = tf.random.normal(shape=tf.shape(za_mu))
        za = za_mu + tf.exp(0.5 * za_log) * eps_a

        # 4. CONCATENATE (Decoder handles the split)
        z_combined = tf.concat([zp, za], axis=-1)

        return self.decoder([z_combined, text_embeddings], training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "encoder": serialize_keras_object(self.encoder),
            "decoder": serialize_keras_object(self.decoder),
            "unipolar_indices": self.unipolar_indices,
            "bipolar_indices": self.bipolar_indices,
            "bool_indices": self.bool_indices,
            "cat_indices": self.cat_indices,
            "categorical_num_classes": self.categorical_num_classes,
            "group_masking_map": self.group_masking_map,
            "latent_dim_params": self.latent_dim_params,
            "latent_dim_audio": self.latent_dim_audio,
            "latent_dim_total": self.latent_dim_total,
            "beta": self.beta,
            "latent_dropout_rate": self.latent_dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        encoder = deserialize_keras_object(config.pop("encoder"))
        decoder = deserialize_keras_object(config.pop("decoder"))
        return cls(encoder, decoder, **config)


# ==============================================================================
# 3. DIFFUSION COMPONENTS (Stage 2: Denoising)
# ==============================================================================

@register_keras_serializable(package="custom", name="SinusoidalTimeEmbedding")
class SinusoidalTimeEmbedding(Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
    def call(self, time):
        # NOTE: Updated to match notebook EXACTLY to avoid rank/broadcasting issues
        half_dim = self.dim // 2
        embeddings = tf.math.log(10000.0) / (half_dim - 1)
        embeddings = tf.exp(tf.range(half_dim, dtype=tf.float32) * -embeddings)
        embeddings = tf.cast(time, tf.float32) * embeddings[None, :]
        embeddings = tf.concat([tf.sin(embeddings), tf.cos(embeddings)], axis=-1)
        return embeddings
    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config

@register_keras_serializable(package="custom", name="FiLM_Modulate")
class FiLM_Modulate(Layer):
    def call(self, inputs):
        x, gammas, betas = inputs
        return (x * (1.0 + gammas)) + betas

@register_keras_serializable(package="custom", name="ResidualBlock")
class ResidualBlock(Layer):
    def __init__(self, width, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.dropout_rate = dropout
        self.norm1 = LayerNormalization()
        self.dense1 = Dense(width, activation="swish")
        self.drop1 = Dropout(dropout)
        self.norm2 = LayerNormalization()
        self.dense2 = Dense(width, activation="swish")
        self.drop2 = Dropout(dropout)
        self.film_proj = Dense(width * 4, activation=None)
    def call(self, x, conditions):
        residual = x
        film_params = self.film_proj(conditions)
        gam1, bet1, gam2, bet2 = tf.split(film_params, num_or_size_splits=4, axis=-1)
        x = self.norm1(x)
        x = FiLM_Modulate()([x, gam1, bet1])
        x = self.dense1(x)
        x = self.drop1(x)
        x = self.norm2(x)
        x = FiLM_Modulate()([x, gam2, bet2])
        x = self.dense2(x)
        x = self.drop2(x)
        return Add()([residual, x])
    def get_config(self):
        config = super().get_config()
        config.update({"width": self.width, "dropout": self.dropout_rate})
        return config

class DiffusionScheduler:
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = tf.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas)
        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1.0 - self.alphas_cumprod)
    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alpha_prod = tf.gather(self.sqrt_alphas_cumprod, timesteps)
        sqrt_one_minus_alpha_prod = tf.gather(self.sqrt_one_minus_alphas_cumprod, timesteps)
        sqrt_alpha_prod = tf.reshape(sqrt_alpha_prod, [-1, 1])
        sqrt_one_minus_alpha_prod = tf.reshape(sqrt_one_minus_alpha_prod, [-1, 1])
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

@register_keras_serializable(package="custom", name="LatentDiffusionModel")
class LatentDiffusionModel(tf.keras.Model):
    def __init__(self, vae_encoder, vae_decoder, denoiser, timesteps=1000, **kwargs):
        super().__init__(**kwargs)
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.denoiser = denoiser
        self.timesteps = int(timesteps)
        self.scheduler = DiffusionScheduler(timesteps=self.timesteps)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs, training=False):
        return self.denoiser(inputs, training=training)

    @tf.function(jit_compile=False) # jit_compile=False to be safe with XLA issues
    def _diffusion_loop_compiled(self, z, text_embeds, timestep_indices):
        """Graph-optimized diffusion loop"""
        # We need to iterate over the TENSOR indices
        for i in timestep_indices:
            batch_size = tf.shape(text_embeds)[0]
            t = tf.ones((batch_size,), dtype=tf.int32) * i
            
            pred_noise = self.denoiser([z, t, text_embeds], training=False)
            
            alpha = tf.gather(self.scheduler.alphas, i)
            alpha_cumprod = tf.gather(self.scheduler.alphas_cumprod, i)
            beta = tf.gather(self.scheduler.betas, i)
            
            sqrt_one_minus_alpha_cumprod = tf.sqrt(1.0 - alpha_cumprod)
            model_mean = (1 / tf.sqrt(alpha)) * (z - ((1 - alpha) / (sqrt_one_minus_alpha_cumprod)) * pred_noise)
            
            # Add noise if t > 0
            # tf.cond is needed for graph mode conditional
            z = tf.cond(
                pred=i > 0,
                true_fn=lambda: model_mean + tf.sqrt(beta) * tf.random.normal(shape=tf.shape(z)),
                false_fn=lambda: model_mean
            )
            
        return z

    def generate(self, text_embeds, steps=50):
        # Ensure input is a tensor
        text_embeds = tf.convert_to_tensor(text_embeds, dtype=tf.float32)
        batch_size = tf.shape(text_embeds)[0]
        
        # Latent dim is the input shape of the denoiser (total params + audio)
        # Note: input_shape[0] usually returns (None, dim), so we take index 1
        latent_dim = self.denoiser.input_shape[0][1]
        
        z = tf.random.normal(shape=(batch_size, latent_dim))

        # Calculate strided timesteps for faster generation
        # Using tf.range to pass to compiled function
        if steps is None or steps >= self.timesteps:
            timestep_indices = tf.range(self.timesteps - 1, -1, -1, dtype=tf.int32)
        else:
            # Simple linear stride
            step_ratio = self.timesteps // steps
            timestep_indices = tf.range(self.timesteps - 1, -1, -step_ratio, dtype=tf.int32)

        # Run optimized loop
        z = self._diffusion_loop_compiled(z, text_embeds, timestep_indices)
                
        decoded = self.vae_decoder([z, text_embeds], training=False)
        return decoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "vae_encoder": serialize_keras_object(self.vae_encoder),
            "vae_decoder": serialize_keras_object(self.vae_decoder),
            "denoiser": serialize_keras_object(self.denoiser),
            "timesteps": self.timesteps
        })
        return config

    @classmethod
    def from_config(cls, config):
        vae_encoder = deserialize_keras_object(config.pop("vae_encoder"))
        vae_decoder = deserialize_keras_object(config.pop("vae_decoder"))
        denoiser = deserialize_keras_object(config.pop("denoiser"))
        return cls(vae_encoder, vae_decoder, denoiser, **config)


# ==============================================================================
# 4. UTILITIES
# ==============================================================================

def _safe_probs(raw_head):
    probs = np.array(raw_head, dtype=np.float32).reshape(-1)
    probs[probs < 0] = 0.0
    s = probs.sum()
    if not np.isfinite(s) or s <= 1e-6:
        probs = np.ones_like(probs, dtype=np.float32) / probs.size
    else:
        probs = probs / s
    return probs

class ParameterUtils:
    @staticmethod
    def get_indices_and_classes(serum_parameters):
        continuous_param_indices = []
        boolean_param_indices = []
        categorical_param_indices = []
        categorical_num_classes = {}

        for group_name, params in serum_parameters.items():
            for pinfo in params:
                idx = int(pinfo["id"])
                ptype = pinfo["type"]

                if ptype == "continuous":
                    continuous_param_indices.append(idx)
                elif ptype == "boolean":
                    boolean_param_indices.append(idx)
                elif ptype == "categorical":
                    categorical_param_indices.append(idx)
                    categorical_num_classes[idx] = int(pinfo["num_categories"])
        
        continuous_param_indices.sort()
        boolean_param_indices.sort()
        categorical_param_indices.sort()
        
        return continuous_param_indices, boolean_param_indices, categorical_param_indices, categorical_num_classes

    @staticmethod
    def reconstruct_parameters_from_heads(predicted_outputs, parameter_types, categorical_num_classes, sample_categorical=True):
        """
        Reconstructs parameter vector from split heads AND retrieves audio embeddings.
        Input parameter_types should be the cleaned dict (map of id -> param_def).
        """
        n_params = len(parameter_types)
        reconstructed = np.zeros(n_params, dtype=np.float32)

        valid_param_types = parameter_types # Assumed to be dict int->def

        # Recalculate lists based on passed types
        continuous_params = [int(i) for i, p in valid_param_types.items() if p["type"] == "continuous"]
        
        # Recalculate Mod Matrix Range based on cleaned indices (starts at 170)
        mod_matrix_ids = set(range(170, 202))

        unipolar_indices = sorted([i for i in continuous_params if i not in mod_matrix_ids])
        bipolar_indices = sorted([i for i in continuous_params if i in mod_matrix_ids])
        boolean_params = sorted([int(i) for i, p in valid_param_types.items() if p["type"] == "boolean"])
        categorical_params = sorted([int(i) for i, p in valid_param_types.items() if p["type"] == "categorical"])

        head_idx = 0

        # --- 1. RECONSTRUCT KNOBS ---

        # A. Unipolar
        if unipolar_indices:
            uni_head = np.array(predicted_outputs[head_idx], dtype=np.float32).reshape(-1)
            count = min(len(unipolar_indices), len(uni_head))
            for i in range(count):
                param_idx = unipolar_indices[i]
                if param_idx < n_params:
                    reconstructed[param_idx] = uni_head[i]
            head_idx += 1

        # B. Bipolar (Gate + Value)
        if bipolar_indices:
            gate_head = np.array(predicted_outputs[head_idx], dtype=np.float32).reshape(-1)
            val_head = np.array(predicted_outputs[head_idx+1], dtype=np.float32).reshape(-1)
            head_idx += 2
            count = min(len(bipolar_indices), len(gate_head))
            for i in range(count):
                param_idx = bipolar_indices[i]
                if param_idx < n_params:
                    gate = 1.0 / (1.0 + np.exp(-gate_head[i]))
                    val = (val_head[i] + 1.0) / 2.0
                    reconstructed[param_idx] = val if gate >= 0.25 else 0.5

        # C. Boolean
        if boolean_params:
            bool_head = np.array(predicted_outputs[head_idx], dtype=np.float32).reshape(-1)
            count = min(len(boolean_params), len(bool_head))
            for i in range(count):
                param_idx = boolean_params[i]
                if param_idx < n_params:
                    reconstructed[param_idx] = 1.0 if bool_head[i] > 0.5 else 0.0
            head_idx += 1

        # D. Categorical
        # We need to rely on the fact that audio heads are at the end (3 of them)
        # So we iterate until we hit the audio heads
        
        # Calculate how many heads are left
        total_heads = len(predicted_outputs)
        heads_remaining = total_heads - head_idx
        # We know the last 3 are audio
        categorical_heads_count = heads_remaining - 3
        
        # Iterate through categorical params
        for i in range(len(categorical_params)):
            if heads_remaining <= 3: break # Safety break if we run into audio heads
            
            param_idx = categorical_params[i]
            head = predicted_outputs[head_idx]
            if isinstance(head, (list, tuple)): head = head[0]
            probs = _safe_probs(head)
            num_c = categorical_num_classes.get(param_idx, len(probs))

            if sample_categorical: 
                choice = np.random.choice(len(probs), p=probs)
            else: 
                choice = np.argmax(probs)

            if param_idx < n_params:
                reconstructed[param_idx] = float(choice) / max(1, (num_c - 1))
            
            head_idx += 1
            heads_remaining -= 1

        # --- 2. RETRIEVE AUDIO EMBEDDINGS ---
        def extract_embed(h):
            return np.array(h, dtype=np.float32).reshape(512)

        audio_vectors = {}
        try:
            # The last 3 heads are Audio
            osc_a_vec = extract_embed(predicted_outputs[-3])
            osc_b_vec = extract_embed(predicted_outputs[-2])
            osc_n_vec = extract_embed(predicted_outputs[-1])

            audio_vectors = {
                "osc_a": osc_a_vec,
                "osc_b": osc_b_vec,
                "noise": osc_n_vec
            }
        except IndexError:
            print("Warning: Audio heads not found in output.")

        return reconstructed, audio_vectors

def numpy_to_json(parameter_array, serum_parameters):
    output_json = {}
    flat_lookup = {}
    for group, params in serum_parameters.items():
        for p in params:
            flat_lookup[int(p["id"])] = p

    for idx, val in enumerate(parameter_array):
        if idx in flat_lookup:
            param_def = flat_lookup[idx]
            entry = param_def.copy()
            entry["value"] = float(val)
            output_json[str(idx)] = entry
    return output_json


# ==============================================================================
# 5. MAIN WRAPPER CLASS
# ==============================================================================

class VAE_V2P7_OSC_SEPARATED:
    def __init__(self, model_path, timesteps=1000):
        self.model_path = model_path
        self.timesteps = timesteps
        self.model = None

    def load(self):
        print(f"Loading VAE V2.7 Oscillator Separated Model from {self.model_path}")
        
        try:
            custom_objects = {
                "VAE_Text_to_Synth_Audio": VAE_Text_to_Synth_Audio,
                "LatentDiffusionModel": LatentDiffusionModel,
                "SinusoidalTimeEmbedding": SinusoidalTimeEmbedding,
                "ResidualBlock": ResidualBlock,
                "FiLM_Modulate": FiLM_Modulate,
                "FiLMLayer": FiLMLayer,
                "SplitLatentLayer": SplitLatentLayer,
                "StopGradient": StopGradient # Standard class
            }
            
            # Use CPU loading to avoid GPU memory issues if running on CPU-only or small VRAM
            with tf.device("/cpu:0"):
                self.model = load_model(self.model_path, custom_objects=custom_objects)
            print("Model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def generate(self, text_embeds, steps=None, seed=None):
        """
        Wrapper for model.generate.
        text_embeds: (BATCH, 512) tensor/array
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
            
        if seed is not None:
             tf.random.set_seed(seed)
             np.random.seed(seed)
        
        steps_to_use = steps if steps is not None else self.timesteps
        
        # If needed, temporarily override timesteps (though the model config has it)
        # For now, just call generate
        return self.model.generate(text_embeds, steps=steps_to_use)
