
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Input, Dense, Concatenate, Dropout, BatchNormalization, LayerNormalization, LeakyReLU, Activation, Add, Multiply, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable
from keras.saving import serialize_keras_object, deserialize_keras_object

# ==============================================================================
# Helper Functions for Parameter Reconstruction
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

def numpy_to_json(parameter_array, serum_parameters):
    """
    Converts a flat parameter array back to the rich JSON structure 
    using the SERUM_PARAMETERS metadata.
    """
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

class ParameterUtils:
    """Utilities for handling Serum Parameter indices and types"""
    
    @staticmethod
    def get_indices_and_classes(serum_parameters):
        """
        Parses SERUM_PARAMETERS to return the indices used by the model
        """
        # Create a flat map of ID -> Parameter Def
        param_map = {int(p['id']): p for g in serum_parameters.values() for p in g}
        
        # Filter out the "Ignored" IDs (4, 24, 44) - Wavetable/Noise IDs usually handled separately
        # valid_param_types = {k:v for k,v in param_map.items() if int(k) not in [4, 24, 44]}
        valid_param_types = param_map
        
        continuous_params = [int(i) for i, p in valid_param_types.items() if p["type"] == "continuous"]
        # Mod Matrix is now roughly indices 170 to 201 (since we removed 3 items before it)
        mod_matrix_ids = set(range(170, 202))

        unipolar_indices = sorted([i for i in continuous_params if i not in mod_matrix_ids])
        bipolar_indices = sorted([i for i in continuous_params if i in mod_matrix_ids])
        
        boolean_params = sorted([int(i) for i, p in valid_param_types.items() if p["type"] == "boolean"])
        categorical_params = sorted([int(i) for i, p in valid_param_types.items() if p["type"] == "categorical"])
        
        # Calculate Num Classes for Categorical
        categorical_num_classes = {}
        for idx in categorical_params:
            if "num_categories" in param_map[idx]:
                 categorical_num_classes[idx] = param_map[idx]["num_categories"]
            else:
                 # Fallback default if not specified (though it should be)
                 categorical_num_classes[idx] = 10 
                 
        return {
            "param_map": param_map,
            "unipolar_indices": unipolar_indices,
            "bipolar_indices": bipolar_indices,
            "bool_indices": boolean_params,
            "cat_indices": categorical_params,
            "categorical_num_classes": categorical_num_classes,
            "n_params": len(param_map) # The total 'virtual' size including ignored ones
        }

    @staticmethod
    def reconstruct_parameters_from_heads(predicted_outputs, param_info, sample_categorical=True):
        """
        Reconstructs parameter vector from split heads AND retrieves audio embeddings.
        
        param_info: Result from get_indices_and_classes() containing indices and maps.
        """
        n_params = 202 # Fixed max size for Serum params usually
        # Or use param_info['n_params'] if we trust it covers the max ID. 
        # Safest to use strict 202 or max(param_map.keys()) + 1
        max_id = max(param_info['unipolar_indices'] + param_info['bipolar_indices'] + param_info['bool_indices'] + param_info['cat_indices'] + [201])
        n_params = max_id + 1
        
        reconstructed = np.zeros(n_params, dtype=np.float32)
        
        unipolar_indices = param_info['unipolar_indices']
        bipolar_indices = param_info['bipolar_indices']
        boolean_params = param_info['bool_indices']
        categorical_params = param_info['cat_indices']
        categorical_num_classes = param_info['categorical_num_classes']

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
        # Stop before audio heads (last 3)
        max_param_heads = len(predicted_outputs) - 3
        
        for param_idx in categorical_params:
            if head_idx >= max_param_heads: break

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

        # --- 2. RETRIEVE AUDIO EMBEDDINGS ---
        # The last 3 heads are always Osc A, Osc B, Osc N
        def extract_embed(h):
            return np.array(h, dtype=np.float32).reshape(512)

        audio_vectors = {}
        try:
            audio_vectors["osc_a"] = extract_embed(predicted_outputs[-3])
            audio_vectors["osc_b"] = extract_embed(predicted_outputs[-2])
            audio_vectors["noise"] = extract_embed(predicted_outputs[-1])
        except IndexError:
            print("Warning: Audio heads not found in output.")

        return reconstructed, audio_vectors


# --- 0. HYPERPARAMETER CONSTANTS ---
W_CONT      = 10.0   # Weight for continuous knobs
W_BOOL      = 5.0    # Weight for switches
W_CAT       = 15.0   # Weight for menus
W_MOD_GATE  = 5.0    # Weight for modulation slots
W_AUDIO     = 2.0    # Weight for CLAP embedding reconstruction

# ==============================================================================
# 1. SHARED LAYERS
# ==============================================================================

@register_keras_serializable(package="custom", name="FiLMLayer")
class FiLMLayer(Layer):
    def call(self, inputs):
        x, gamma, beta = inputs
        return x * gamma + beta

def sigmoid_focal_crossentropy(y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=True):
    if from_logits: y_pred = tf.sigmoid(y_pred)
    bce = K.binary_crossentropy(y_true, y_pred)
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    return alpha * K.pow(1.0 - p_t, gamma) * bce

@register_keras_serializable(package="custom", name="VAE_Text_to_Synth_Audio")
class VAE_Text_to_Synth_Audio(tf.keras.Model):
    def __init__(self, encoder, decoder, unipolar_indices, bipolar_indices,
                 bool_indices, cat_indices, categorical_num_classes, group_masking_map,
                 latent_dim, beta=1.0, latent_dropout_rate=0.0, **kwargs):
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
        self.latent_dim = int(latent_dim)
        self.beta = float(beta)
        self.latent_dropout_rate = float(latent_dropout_rate)

    def call(self, inputs, training=False):
        # inputs expected: [text, params, audio]
        text_embeddings, params_in, audio_in = inputs
        z_mean, z_log_var = self.encoder([text_embeddings, params_in, audio_in])
        eps = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * eps
        return self.decoder([z, text_embeddings], training=training)

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
            "latent_dim": self.latent_dim,
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
        self.alphas_cumprod_prev = tf.concat([tf.constant([1.0]), self.alphas_cumprod[:-1]], axis=0)
        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1.0 - self.alphas_cumprod)

@register_keras_serializable(package="custom", name="LatentDiffusionModel")
class LatentDiffusionModel(tf.keras.Model):
    def __init__(self, vae_encoder, vae_decoder, denoiser, timesteps=1000, embedding_model=None, **kwargs):
        super().__init__(**kwargs)
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.denoiser = denoiser
        self.timesteps = int(timesteps)
        self.scheduler = DiffusionScheduler(timesteps=self.timesteps)
        self.embedding_model = embedding_model

    def call(self, inputs, training=False):
        return self.denoiser(inputs, training=training)

    @tf.function
    def _diffusion_loop_compiled(self, z, text_embeds, timestep_indices):
        """Graph-optimized diffusion loop (No XLA to prevent potential hanging on large unrolls)"""
        # Iterate over the tensor of timestep indices
        for i in timestep_indices:
            batch_size = tf.shape(text_embeds)[0]
            t = tf.ones((batch_size,), dtype=tf.int32) * i
            
            # Predict noise
            pred_noise = self.denoiser([z, t, text_embeds], training=False)
            
            alpha = tf.gather(self.scheduler.alphas, i)
            alpha_cumprod = tf.gather(self.scheduler.alphas_cumprod, i)
            beta = tf.gather(self.scheduler.betas, i)
            
            sqrt_one_minus_alpha_cumprod = tf.sqrt(1.0 - alpha_cumprod)
            model_mean = (1 / tf.sqrt(alpha)) * (z - ((1 - alpha) / (sqrt_one_minus_alpha_cumprod)) * pred_noise)
            
            # Use tf.cond for conditional logic inside graph
            def add_noise():
                noise = tf.random.normal(shape=tf.shape(z))
                sigma = tf.sqrt(beta)
                return model_mean + sigma * noise
                
            def no_noise():
                return model_mean
                
            z = tf.cond(i > 0, add_noise, no_noise)
            
        return z

    def generate(self, text_description, diffusion_steps=50, seed=None, verbose=False):
        # 1. Encode Text
        if verbose: print("Encoding text...")
        if isinstance(text_description, str):
            text_description = [text_description]
            
        if self.embedding_model:
            # Check if embedding model is CLAP
            if hasattr(self.embedding_model, "clap_encode_text"):
                 text_embeds = self.embedding_model.clap_encode_text(text_description)
            else:
                # Assuming sentence-transformers
                 text_embeds = self.embedding_model.encode(text_description)
        else:
            raise ValueError("No embedding model loaded. Call _load_embedding_model() first.")
            
        text_embeds = tf.convert_to_tensor(text_embeds, dtype=tf.float32)
        batch_size = tf.shape(text_embeds)[0]
        latent_dim = self.vae_encoder.output_shape[0][1] # Infer from encoder output
        
        if seed is not None:
             tf.random.set_seed(seed)

        z = tf.random.normal(shape=(batch_size, latent_dim))
        
        if verbose: print(f"Sampling with {diffusion_steps} steps (JIT Compile)...")
        
        # Determine timestep indices
        # We need to construct a Tensor of indices to iterate over in the graph
        if diffusion_steps is None or diffusion_steps >= self.timesteps:
            # Full inverse range: [999, 998, ... 0]
            timestep_indices = tf.range(self.timesteps - 1, -1, -1, dtype=tf.int32)
        else:
            # Strided range
            step_ratio = self.timesteps // diffusion_steps
            # Python list construction for strided range, then convert to tensor
            # range(0, 1000, 20) -> [0, 20, 40...] -> reverse -> [980, ..., 0]
            # TF range supports negative delta directly
            timestep_indices = tf.range(self.timesteps - 1, -1, -step_ratio, dtype=tf.int32)

        # Run compiled loop
        z = self._diffusion_loop_compiled(z, text_embeds, timestep_indices)

        if verbose: print("Decoding...")
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
    
    def _load_model(self, model_path=None):
         # This method is not needed if we load weights externally, 
         # but useful for the Class wrapper pattern
         pass
         
    def _load_embedding_model(self):
        import torch
        from transformers import ClapModel, ClapProcessor
        
        class CLAPWrapper:
            def __init__(self):
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                model_name = "laion/clap-htsat-fused"
                self.model = ClapModel.from_pretrained(model_name).to(self.device)
                self.processor = ClapProcessor.from_pretrained(model_name)
                self.model.eval()

            @torch.no_grad()
            def clap_encode_text(self, texts, batch_size=128, normalize=True):
                out = []
                for i in range(0, len(texts), batch_size):
                    batch = [str(t) for t in texts[i:i+batch_size]]
                    inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    feats = self.model.get_text_features(**inputs)
                    if normalize:
                        feats = torch.nn.functional.normalize(feats, dim=-1)
                    out.append(feats.cpu())
                return torch.cat(out, dim=0).numpy()

        self.embedding_model = CLAPWrapper()


# Wrapper Class for Easy Loading
class VAE_V2P7_OSC:
    def __init__(self, model_path, timesteps=1000):
        self.model_path = model_path
        self.timesteps = timesteps
        self.model = None
    
    def load(self):
        print(f"Loading VAE V2.7 Oscillator Model from {self.model_path}")
        # Load the full model
        # Note: We need to register custom objects
        with tf.keras.utils.custom_object_scope({
            'FiLMLayer': FiLMLayer,
            'VAE_Text_to_Synth_Audio': VAE_Text_to_Synth_Audio,
            'SinusoidalTimeEmbedding': SinusoidalTimeEmbedding,
            'FiLM_Modulate': FiLM_Modulate,
            'ResidualBlock': ResidualBlock,
            'LatentDiffusionModel': LatentDiffusionModel
        }):
            self.model = tf.keras.models.load_model(self.model_path)
            
        # Load embedding model
        self.model._load_embedding_model()
        print("Model loaded successfully.")
        
    def generate(self, prompts, steps=1000, seed=None):
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        return self.model.generate(prompts, diffusion_steps=steps, seed=seed)
