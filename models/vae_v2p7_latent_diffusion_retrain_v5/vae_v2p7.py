"""
VAE V2.7: Latent Diffusion Model for VAE-based synthesizer parameter generation.

This module contains all VAE V2.7-specific components:
- Custom Keras layers (SliceLayer, FiLMLayer, SinusoidalTimeEmbedding, FiLM_Modulate, ResidualBlock)
- VAE model class with Standard Gaussian Prior
- Diffusion Scheduler (DDPM)
- Latent Diffusion Model with 1000-step denoising
- Parameter reconstruction logic
- Main VAE_V2P7 wrapper class

Key Innovation: Two-stage architecture where VAE compresses parameters to latent space,
then a diffusion model generates latents conditioned on text, enabling more flexible
generation through iterative denoising.
"""

import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Layer, Dense, LayerNormalization, Dropout, Add, Concatenate, LeakyReLU, Activation

try:
    from keras.saving import register_keras_serializable, serialize_keras_object, deserialize_keras_object
except (ImportError, AttributeError):
    from tensorflow.keras.utils import register_keras_serializable
    from tensorflow.keras.saving import serialize_keras_object, deserialize_keras_object

from encoders import encode_text, load_clap_model, get_sentence_transformer_model


# Parameter Type Definitions (VAE V2 specific)
# -------------------------------------------------------------------------

GROUPED_PARAMETER_TYPES = {
    "global": [
        {"id": "0", "serum_id": "0", "name": "MasterVol", "type": "continuous"},
        {"id": "1", "serum_id": "65", "name": "PortTime", "type": "continuous"},
        {"id": "2", "serum_id": "66", "name": "PortCurve", "type": "continuous"},
    ],

    "osc_a": [
        {"id": "3", "serum_id": "212", "name": "Osc A On", "type": "boolean"},
        {"id": "4", "serum_id": "1", "name": "A Vol", "type": "continuous"},
        {"id": "5", "serum_id": "2", "name": "A Pan", "type": "continuous"},
        {"id": "6", "serum_id": "3", "name": "A Octave", "type": "categorical", "num_categories": 9},
        {"id": "7", "serum_id": "4", "name": "A Semi", "type": "categorical", "num_categories": 25},
        {"id": "8", "serum_id": "5", "name": "A Fine", "type": "continuous"},
        {"id": "9", "serum_id": "10", "name": "A CoarsePit", "type": "continuous"},
        {"id": "10", "serum_id": "6", "name": "A Unison", "type": "continuous"},
        {"id": "11", "serum_id": "7", "name": "A UniDet", "type": "continuous"},
        {"id": "12", "serum_id": "8", "name": "A UniBlend", "type": "continuous"},
        {"id": "13", "serum_id": "172", "name": "A Uni LR", "type": "continuous"},
        {"id": "14", "serum_id": "174", "name": "A Uni Warp", "type": "continuous"},
        {"id": "15", "serum_id": "176", "name": "A Uni WTPos", "type": "continuous"},
        {"id": "16", "serum_id": "178", "name": "A Uni Stack", "type": "categorical", "num_categories": 9},
        {"id": "17", "serum_id": "11", "name": "A WTPos", "type": "continuous"},
        {"id": "18", "serum_id": "168", "name": "WarpOscA", "type": "categorical", "num_categories": 23},
        {"id": "19", "serum_id": "9", "name": "A Warp", "type": "continuous"},
        {"id": "20", "serum_id": "12", "name": "A RandPhase", "type": "continuous"},
        {"id": "21", "serum_id": "13", "name": "A Phase", "type": "continuous"},
    ],

    "osc_b": [
        {"id": "22", "serum_id": "213", "name": "Osc B On", "type": "boolean"},
        {"id": "23", "serum_id": "14", "name": "B Vol", "type": "continuous"},
        {"id": "24", "serum_id": "15", "name": "B Pan", "type": "continuous"},
        {"id": "25", "serum_id": "16", "name": "B Octave", "type": "categorical", "num_categories": 9},
        {"id": "26", "serum_id": "17", "name": "B Semi", "type": "categorical", "num_categories": 25},
        {"id": "27", "serum_id": "18", "name": "B Fine", "type": "continuous"},
        {"id": "28", "serum_id": "23", "name": "B CoarsePit", "type": "continuous"},
        {"id": "29", "serum_id": "19", "name": "B Unison", "type": "continuous"},
        {"id": "30", "serum_id": "20", "name": "B UniDet", "type": "continuous"},
        {"id": "31", "serum_id": "21", "name": "B UniBlend", "type": "continuous"},
        {"id": "32", "serum_id": "173", "name": "B Uni LR", "type": "continuous"},
        {"id": "33", "serum_id": "175", "name": "B Uni Warp", "type": "continuous"},
        {"id": "34", "serum_id": "177", "name": "B Uni WTPos", "type": "continuous"},
        {"id": "35", "serum_id": "179", "name": "B Uni Stack", "type": "categorical", "num_categories": 9},
        {"id": "36", "serum_id": "24", "name": "B WTPos", "type": "continuous"},
        {"id": "37", "serum_id": "169", "name": "WarpOscB", "type": "categorical", "num_categories": 23},
        {"id": "38", "serum_id": "22", "name": "B Warp", "type": "continuous"},
        {"id": "39", "serum_id": "25", "name": "B RandPhase", "type": "continuous"},
        {"id": "40", "serum_id": "26", "name": "B Phase", "type": "continuous"},
    ],

    "noise_osc": [
        {"id": "41", "serum_id": "214", "name": "Osc N On", "type": "boolean"},
        {"id": "42", "serum_id": "27", "name": "Noise Level", "type": "continuous"},
        {"id": "43", "serum_id": "28", "name": "Noise Pitch", "type": "continuous"},
        {"id": "44", "serum_id": "29", "name": "Noise Fine", "type": "continuous"},
        {"id": "45", "serum_id": "30", "name": "Noise Pan", "type": "continuous"},
        {"id": "46", "serum_id": "31", "name": "Noise RandPhase", "type": "continuous"},
        {"id": "47", "serum_id": "32", "name": "Noise Phase", "type": "continuous"},
    ],

    "sub_osc": [
        {"id": "48", "serum_id": "215", "name": "Osc S On", "type": "boolean"},
        {"id": "49", "serum_id": "33", "name": "Sub Osc Level", "type": "continuous"},
        {"id": "50", "serum_id": "34", "name": "Sub Osc Pan", "type": "continuous"},
        {"id": "51", "serum_id": "170", "name": "SubOscShape", "type": "categorical", "num_categories": 6},
        {"id": "52", "serum_id": "171", "name": "SubOscOctave", "type": "continuous"},
    ],

    "filter": [
        {"id": "53", "serum_id": "216", "name": "Filter On", "type": "boolean"},
        {"id": "54", "serum_id": "44", "name": "Fil Type", "type": "categorical", "num_categories": 95},
        {"id": "55", "serum_id": "40", "name": "OscA>Fil", "type": "boolean"},
        {"id": "56", "serum_id": "41", "name": "OscB>Fil", "type": "boolean"},
        {"id": "57", "serum_id": "42", "name": "OscN>Fil", "type": "boolean"},
        {"id": "58", "serum_id": "43", "name": "OscS>Fil", "type": "boolean"},
        {"id": "59", "serum_id": "45", "name": "Fil Cutoff", "type": "continuous"},
        {"id": "60", "serum_id": "46", "name": "Fil Reso", "type": "continuous"},
        {"id": "61", "serum_id": "47", "name": "Fil Driv", "type": "continuous"},
        {"id": "62", "serum_id": "48", "name": "Fil Var", "type": "continuous"},
        {"id": "63", "serum_id": "49", "name": "Fil Mix", "type": "continuous"},
        {"id": "64", "serum_id": "50", "name": "Fil Stereo", "type": "continuous"},
    ],

    "envelopes": [
        {"id": "65", "serum_id": "35", "name": "Env1 Atk", "type": "continuous"},
        {"id": "66", "serum_id": "36", "name": "Env1 Hold", "type": "continuous"},
        {"id": "67", "serum_id": "37", "name": "Env1 Dec", "type": "continuous"},
        {"id": "68", "serum_id": "38", "name": "Env1 Sus", "type": "continuous"},
        {"id": "69", "serum_id": "39", "name": "Env1 Rel", "type": "continuous"},
        {"id": "70", "serum_id": "51", "name": "Env2 Atk", "type": "continuous"},
        {"id": "71", "serum_id": "52", "name": "Env2 Hold", "type": "continuous"},
        {"id": "72", "serum_id": "53", "name": "Env2 Dec", "type": "continuous"},
        {"id": "73", "serum_id": "54", "name": "Env2 Sus", "type": "continuous"},
        {"id": "74", "serum_id": "55", "name": "Env2 Rel", "type": "continuous"},
    ],

    "lfos": [
        {"id": "75", "serum_id": "61", "name": "LFO1 Rate", "type": "categorical", "num_categories": 19},
        {"id": "76", "serum_id": "273", "name": "LFO1 Rise", "type": "categorical", "num_categories": 20},
        {"id": "77", "serum_id": "281", "name": "LFO1 Delay", "type": "categorical", "num_categories": 20},
        {"id": "78", "serum_id": "223", "name": "LFO1 Smooth", "type": "continuous"},
        {"id": "79", "serum_id": "62", "name": "LFO2 Rate", "type": "categorical", "num_categories": 19},
        {"id": "80", "serum_id": "274", "name": "LFO2 Rise", "type": "categorical", "num_categories": 20},
        {"id": "81", "serum_id": "282", "name": "LFO2 Delay", "type": "categorical", "num_categories": 20},
        {"id": "82", "serum_id": "224", "name": "LFO2 Smooth", "type": "continuous"},
    ],

    "fx_hyper_dimension": [
        {"id": "83", "serum_id": "163", "name": "Hyp Enable", "type": "boolean"},
        {"id": "84", "serum_id": "148", "name": "Hyp_Rate", "type": "continuous"},
        {"id": "85", "serum_id": "149", "name": "Hyp_Detune", "type": "continuous"},
        {"id": "86", "serum_id": "150", "name": "Hyp_Unison", "type": "continuous"},
        {"id": "87", "serum_id": "147", "name": "Hyp_Wet", "type": "continuous"},
        {"id": "88", "serum_id": "152", "name": "HypDim_Size", "type": "continuous"},
        {"id": "89", "serum_id": "153", "name": "HypDim_Mix", "type": "continuous"},
    ],

    "fx_distortion": [
        {"id": "90", "serum_id": "154", "name": "Dist Enable", "type": "boolean"},
        {"id": "91", "serum_id": "96", "name": "Dist_Wet", "type": "continuous"},
        {"id": "92", "serum_id": "97", "name": "Dist_Drv", "type": "continuous"},
        {"id": "93", "serum_id": "99", "name": "Dist_Mode", "type": "categorical", "num_categories": 16},
        {"id": "94", "serum_id": "102", "name": "Dist_PrePost", "type": "categorical", "num_categories": 3},
        {"id": "95", "serum_id": "98", "name": "Dist_L/B/H", "type": "categorical", "num_categories": 3},
        {"id": "96", "serum_id": "100", "name": "Dist_Freq", "type": "continuous"},
        {"id": "97", "serum_id": "101", "name": "Dist_BW", "type": "continuous"},
    ],

    "fx_flanger": [
        {"id": "98", "serum_id": "155", "name": "Flg Enable", "type": "boolean"},
        {"id": "99", "serum_id": "103", "name": "Flg_Wet", "type": "continuous"},
        {"id": "100", "serum_id": "104", "name": "Flg_BPM_Sync", "type": "boolean"},
        {"id": "101", "serum_id": "105", "name": "Flg_Rate", "type": "continuous"},
        {"id": "102", "serum_id": "106", "name": "Flg_Dep", "type": "continuous"},
        {"id": "103", "serum_id": "107", "name": "Flg_Feed", "type": "continuous"},
        {"id": "104", "serum_id": "108", "name": "Flg_Stereo", "type": "continuous"},
    ],

    "fx_phaser": [
        {"id": "105", "serum_id": "156", "name": "Phs Enable", "type": "boolean"},
        {"id": "106", "serum_id": "109", "name": "Phs_Wet", "type": "continuous"},
        {"id": "107", "serum_id": "110", "name": "Phs_BPM_Sync", "type": "boolean"},
        {"id": "108", "serum_id": "111", "name": "Phs_Rate", "type": "continuous"},
        {"id": "109", "serum_id": "112", "name": "Phs_Dpth", "type": "continuous"},
        {"id": "110", "serum_id": "113", "name": "Phs_Frq", "type": "continuous"},
        {"id": "111", "serum_id": "114", "name": "Phs_Feed", "type": "continuous"},
        {"id": "112", "serum_id": "115", "name": "Phs_Stereo", "type": "continuous"},
    ],

    "fx_chorus": [
        {"id": "113", "serum_id": "157", "name": "Cho Enable", "type": "boolean"},
        {"id": "114", "serum_id": "116", "name": "Cho_Wet", "type": "continuous"},
        {"id": "115", "serum_id": "117", "name": "Cho_BPM_Sync", "type": "boolean"},
        {"id": "116", "serum_id": "118", "name": "Cho_Rate", "type": "continuous"},
        {"id": "117", "serum_id": "119", "name": "Cho_Dly", "type": "continuous"},
        {"id": "118", "serum_id": "120", "name": "Cho_Dly2", "type": "continuous"},
        {"id": "119", "serum_id": "121", "name": "Cho_Dep", "type": "continuous"},
        {"id": "120", "serum_id": "122", "name": "Cho_Feed", "type": "continuous"},
        {"id": "121", "serum_id": "123", "name": "Cho_Filt", "type": "continuous"},
    ],

    "fx_delay": [
        {"id": "122", "serum_id": "158", "name": "Dly Enable", "type": "boolean"},
        {"id": "123", "serum_id": "124", "name": "Dly_Wet", "type": "continuous"},
        {"id": "124", "serum_id": "125", "name": "Dly_Freq", "type": "continuous"},
        {"id": "125", "serum_id": "126", "name": "Dly_BW", "type": "continuous"},
        {"id": "126", "serum_id": "127", "name": "Dly_BPM_Sync", "type": "boolean"},
        {"id": "127", "serum_id": "128", "name": "Dly_Link", "type": "boolean"},
        {"id": "128", "serum_id": "129", "name": "Dly_TimL", "type": "categorical", "num_categories": 12},
        {"id": "129", "serum_id": "130", "name": "Dly_TimR", "type": "categorical", "num_categories": 12},
        {"id": "130", "serum_id": "131", "name": "Dly_Mode", "type": "categorical", "num_categories": 3},
        {"id": "131", "serum_id": "132", "name": "Dly_Feed", "type": "continuous"},
        {"id": "132", "serum_id": "133", "name": "Dly_Off L", "type": "continuous"},
        {"id": "133", "serum_id": "134", "name": "Dly_Off R", "type": "continuous"},
    ],

    "fx_compressor": [
        {"id": "134", "serum_id": "159", "name": "Comp Enable", "type": "boolean"},
        {"id": "135", "serum_id": "135", "name": "Cmp_Thr", "type": "continuous"},
        {"id": "136", "serum_id": "136", "name": "Cmp_Rat", "type": "categorical", "num_categories": 21},
        {"id": "137", "serum_id": "137", "name": "Cmp_Att", "type": "continuous"},
        {"id": "138", "serum_id": "138", "name": "Cmp_Rel", "type": "continuous"},
        {"id": "139", "serum_id": "139", "name": "CmpGain", "type": "continuous"},
        {"id": "140", "serum_id": "140", "name": "CmpMBnd", "type": "categorical", "num_categories": 2},
        {"id": "141", "serum_id": "270", "name": "CompMB L", "type": "continuous"},
        {"id": "142", "serum_id": "271", "name": "CompMB M", "type": "continuous"},
        {"id": "143", "serum_id": "272", "name": "CompMB H", "type": "continuous"},
        {"id": "144", "serum_id": "269", "name": "Comp_Wet", "type": "continuous"},
    ],

    "fx_reverb": [
        {"id": "145", "serum_id": "160", "name": "Rev Enable", "type": "boolean"},
        {"id": "146", "serum_id": "81", "name": "Verb Wet", "type": "continuous"},
        {"id": "147", "serum_id": "82", "name": "VerbSize", "type": "continuous"},
        {"id": "148", "serum_id": "83", "name": "Decay", "type": "continuous"},
        {"id": "149", "serum_id": "84", "name": "VerbLoCt", "type": "continuous"},
        {"id": "150", "serum_id": "85", "name": "Spin Rate", "type": "continuous"},
        {"id": "151", "serum_id": "86", "name": "VerbHiCt", "type": "continuous"},
        {"id": "152", "serum_id": "87", "name": "Spin Depth", "type": "continuous"},
    ],

    "fx_eq": [
        {"id": "153", "serum_id": "161", "name": "EQ Enable", "type": "boolean"},
        {"id": "154", "serum_id": "88", "name": "EQ FrqL", "type": "continuous"},
        {"id": "155", "serum_id": "89", "name": "EQ FrqH", "type": "continuous"},
        {"id": "156", "serum_id": "90", "name": "EQ Q L", "type": "continuous"},
        {"id": "157", "serum_id": "91", "name": "EQ Q H", "type": "continuous"},
        {"id": "158", "serum_id": "92", "name": "EQ Vol L", "type": "continuous"},
        {"id": "159", "serum_id": "93", "name": "EQ Vol H", "type": "continuous"},
        {"id": "160", "serum_id": "94", "name": "EQ TypL", "type": "categorical", "num_categories": 3},
        {"id": "161", "serum_id": "95", "name": "EQ TypH", "type": "categorical", "num_categories": 3},
    ],

    "fx_filter": [
        {"id": "162", "serum_id": "162", "name": "FX Fil Enable", "type": "boolean"},
        {"id": "163", "serum_id": "141", "name": "FX Fil Wet", "type": "continuous"},
        {"id": "164", "serum_id": "142", "name": "FX Fil Type", "type": "categorical", "num_categories": 89},
        {"id": "165", "serum_id": "143", "name": "FX Fil Freq", "type": "continuous"},
        {"id": "166", "serum_id": "144", "name": "FX Fil Reso", "type": "continuous"},
        {"id": "167", "serum_id": "145", "name": "FX Fil Drive", "type": "continuous"},
        {"id": "168", "serum_id": "146", "name": "FX Fil Var", "type": "continuous"},
        {"id": "169", "serum_id": "268", "name": "FX Fil Pan", "type": "continuous"},
    ],

    "mod_matrix": [
        {"id": "170", "serum_id": "180", "name": "Mod 1 amt", "type": "continuous"},
        {"id": "171", "serum_id": "182", "name": "Mod 2 amt", "type": "continuous"},
        {"id": "172", "serum_id": "184", "name": "Mod 3 amt", "type": "continuous"},
        {"id": "173", "serum_id": "186", "name": "Mod 4 amt", "type": "continuous"},
        {"id": "174", "serum_id": "188", "name": "Mod 5 amt", "type": "continuous"},
        {"id": "175", "serum_id": "190", "name": "Mod 6 amt", "type": "continuous"},
        {"id": "176", "serum_id": "192", "name": "Mod 7 amt", "type": "continuous"},
        {"id": "177", "serum_id": "194", "name": "Mod 8 amt", "type": "continuous"},
        {"id": "178", "serum_id": "196", "name": "Mod 9 amt", "type": "continuous"},
        {"id": "179", "serum_id": "198", "name": "Mod10 amt", "type": "continuous"},
        {"id": "180", "serum_id": "200", "name": "Mod11 amt", "type": "continuous"},
        {"id": "181", "serum_id": "202", "name": "Mod12 amt", "type": "continuous"},
        {"id": "182", "serum_id": "204", "name": "Mod13 amt", "type": "continuous"},
        {"id": "183", "serum_id": "206", "name": "Mod14 amt", "type": "continuous"},
        {"id": "184", "serum_id": "208", "name": "Mod15 amt", "type": "continuous"},
        {"id": "185", "serum_id": "210", "name": "Mod16 amt", "type": "continuous"},
        {"id": "186", "serum_id": "228", "name": "Mod17 amt", "type": "continuous"},
        {"id": "187", "serum_id": "230", "name": "Mod18 amt", "type": "continuous"},
        {"id": "188", "serum_id": "232", "name": "Mod19 amt", "type": "continuous"},
        {"id": "189", "serum_id": "234", "name": "Mod20 amt", "type": "continuous"},
        {"id": "190", "serum_id": "236", "name": "Mod21 amt", "type": "continuous"},
        {"id": "191", "serum_id": "238", "name": "Mod22 amt", "type": "continuous"},
        {"id": "192", "serum_id": "240", "name": "Mod23 amt", "type": "continuous"},
        {"id": "193", "serum_id": "242", "name": "Mod24 amt", "type": "continuous"},
        {"id": "194", "serum_id": "244", "name": "Mod25 amt", "type": "continuous"},
        {"id": "195", "serum_id": "246", "name": "Mod26 amt", "type": "continuous"},
        {"id": "196", "serum_id": "248", "name": "Mod27 amt", "type": "continuous"},
        {"id": "197", "serum_id": "250", "name": "Mod28 amt", "type": "continuous"},
        {"id": "198", "serum_id": "252", "name": "Mod29 amt", "type": "continuous"},
        {"id": "199", "serum_id": "254", "name": "Mod30 amt", "type": "continuous"},
        {"id": "200", "serum_id": "256", "name": "Mod31 amt", "type": "continuous"},
        {"id": "201", "serum_id": "258", "name": "Mod32 amt", "type": "continuous"},
    ],
}


def flatten_grouped_parameter_types(grouped_parameter_types):
    """Flatten grouped_parameter_types to a flat dict."""
    flat = {}
    for gname, params in grouped_parameter_types.items():
        for p in params:
            q = dict(p)
            q["group"] = gname
            flat[int(p["id"])] = q
    return flat


def build_categorical_num_classes_from_types(parameter_types):
    """Build categorical num classes dict from parameter types."""
    return {
        int(i): p["num_categories"]
        for i, p in parameter_types.items()
        if p["type"] == "categorical"
    }


# Flat view that matches the grouped structure
FLAT_PARAMETER_TYPES = flatten_grouped_parameter_types(GROUPED_PARAMETER_TYPES)

# Categorical num classes
CATEGORICAL_NUM_CLASSES = build_categorical_num_classes_from_types(FLAT_PARAMETER_TYPES)


# -------------------------------------------------------------------------


# -------------------------------------------------------------------------

# Loss weights
W_CONT = 10.0
W_BOOL = 5.0
W_CAT = 15.0
W_MOD_GATE = 5.0

# Helper function for focal loss
def sigmoid_focal_crossentropy(y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=True):
    """ Focal Loss for class imbalance (Mod Matrix Gate) """
    from tensorflow.keras import backend as K
    if from_logits: y_pred = tf.sigmoid(y_pred)
    bce = K.binary_crossentropy(y_true, y_pred)
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    return alpha * K.pow(1.0 - p_t, gamma) * bce

# Custom Layers
# -------------------------------------------------------------------------

@register_keras_serializable(package="custom", name="SliceLayer")
class SliceLayer(Layer):
    """
    Utility layer to slice specific columns from a tensor.
    Used to route specific parameters to specific decoder heads.
    """
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = [int(i) for i in (indices if isinstance(indices, (list, tuple)) else [indices])]

    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"indices": self.indices})
        return config


@register_keras_serializable(package="custom", name="FiLMLayer")
class FiLMLayer(Layer):
    """
    Feature-wise Linear Modulation (FiLM) Layer.
    Equation: Output = (Input * Gamma) + Beta
    """
    def call(self, inputs):
        # inputs: [features, gamma, beta]
        x, gamma, beta = inputs
        return x * gamma + beta

# ==============================================================================
# 2. VAE MODEL (Stage 1: Compression)
# ==============================================================================


@register_keras_serializable(package="custom", name="SinusoidalTimeEmbedding")
class SinusoidalTimeEmbedding(Layer):
    """ Standard Sinusoidal Positional Embeddings """
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
    """ Used inside ResMLP Denoiser """
    def call(self, inputs):
        x, gammas, betas = inputs
        return (x * (1.0 + gammas)) + betas


@register_keras_serializable(package="custom", name="ResidualBlock")
class ResidualBlock(Layer):
    """ ResMLP Block with FiLM conditioning """
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


# -------------------------------------------------------------------------
# Diffusion Scheduler
# -------------------------------------------------------------------------

class DiffusionScheduler:
    """ Handles DDPM forward/reverse process math """
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


# -------------------------------------------------------------------------
# VAE Model (Stage 1: Compression)
# -------------------------------------------------------------------------

@register_keras_serializable(package="custom", name="VAE_Text_to_Synth_Standard")
class VAE_Text_to_Synth_Standard(tf.keras.Model):
    """ Stage 1: Standard CVAE with Gaussian Prior """
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

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def sanitize_inputs(self, params):
        """ Zeroes out parameters that are disabled by their parent switch. """
        mask_columns = []
        batch_len = tf.shape(params)[0]
        # Iterate over 202 parameters
        for i in range(params.shape[1]):
            parent_idx = self.param_to_enable.get(i)
            if parent_idx is not None:
                parent_col = params[:, parent_idx]
                gate = tf.cast(parent_col > 0.5, tf.float32)
                mask_columns.append(gate)
            else:
                mask_columns.append(tf.ones((batch_len,), dtype=tf.float32))
        mask_tensor = tf.stack(mask_columns, axis=1)
        return params * mask_tensor

    def call(self, inputs, training=False):
        text_embeddings, params_in = inputs
        z_mean, z_log_var = self.encoder([text_embeddings, params_in])
        eps = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * eps
        return self.decoder([z, text_embeddings], training=training)

    def _group_mask_matrix(self, head_indices, y_true):
        batch_size = tf.shape(y_true)[0]
        mask_cols = []
        for pid in head_indices:
            enable_id = self.param_to_enable.get(int(pid), None)
            if enable_id is None: mask_cols.append(tf.ones((batch_size, 1), dtype=tf.float32))
            else:
                enable_val = tf.gather(y_true, indices=[enable_id], axis=1)
                mask_cols.append(tf.cast(enable_val > 0.5, tf.float32))
        return tf.concat(mask_cols, axis=1) if len(mask_cols) > 1 else mask_cols[0]

    def calculate_loss(self, y_true, y_pred_list, z_mean, z_log_var):
        # --- CRITICAL FIX: SANITIZE TARGETS ---
        # We must grade the model against the CLEAN version of the targets.
        y_true_clean = self.sanitize_inputs(y_true)
        # --------------------------------------

        eps = 1e-7
        head_idx = 0
        uni_loss, mod_loss, bool_loss, cat_loss = 0.0, 0.0, 0.0, 0.0

        if self.unipolar_indices:
            y_true_uni = tf.gather(y_true_clean, self.unipolar_indices, axis=1)
            mask = self._group_mask_matrix(self.unipolar_indices, y_true_clean)
            mse = tf.square(y_true_uni - y_pred_list[head_idx]) * mask
            uni_loss = (tf.reduce_sum(mse) / (tf.reduce_sum(mask) + eps)) * W_CONT
            head_idx += 1

        if self.bipolar_indices:
            y_true_bi_raw = tf.gather(y_true_clean, self.bipolar_indices, axis=1)
            pred_gate, pred_val = y_pred_list[head_idx], y_pred_list[head_idx+1]
            head_idx += 2
            minv, maxv = tf.reduce_min(y_true_bi_raw), tf.reduce_max(y_true_bi_raw)
            y_true_bi = tf.cond(tf.logical_and(minv>=0, maxv<=1.0001), lambda: y_true_bi_raw*2-1, lambda: y_true_bi_raw)
            true_gate = tf.cast(tf.abs(y_true_bi) > 0.01, tf.float32)
            gate_loss = tf.reduce_mean(tf.reduce_sum(sigmoid_focal_crossentropy(true_gate, pred_gate), axis=-1))
            val_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_true_bi - pred_val) * true_gate, axis=-1) / (tf.reduce_sum(true_gate, axis=-1)+eps))
            mod_loss = gate_loss * W_MOD_GATE + val_loss * W_CONT

        if self.bool_indices:
            y_true_bool = tf.gather(y_true_clean, self.bool_indices, axis=1)
            mask = self._group_mask_matrix(self.bool_indices, y_true_clean)
            b_loss = K.binary_crossentropy(y_true_bool, tf.clip_by_value(y_pred_list[head_idx], eps, 1-eps)) * mask
            bool_loss = (tf.reduce_sum(b_loss) / (tf.reduce_sum(mask) + eps)) * W_BOOL
            head_idx += 1

        for j in sorted(self.cat_indices):
            y_true_cat = tf.squeeze(tf.gather(y_true_clean, [j], axis=1), -1)
            y_pred_cat = y_pred_list[head_idx]
            mask = tf.squeeze(self._group_mask_matrix([j], y_true_clean), -1)
            C = self.categorical_num_classes[int(j)]
            labels = tf.cast(tf.where(y_true_cat <= 1.0001, tf.round(y_true_cat*(C-1)), tf.round(y_true_cat)), tf.int32)
            labels = tf.clip_by_value(labels, 0, C-1)
            c_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, y_pred_cat)
            cat_loss += tf.reduce_mean(c_loss * mask) * W_CAT
            head_idx += 1

        recon_loss = uni_loss + mod_loss + bool_loss + cat_loss
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        total_loss = recon_loss + (self.beta * kl_loss)
        return total_loss, recon_loss, kl_loss

    def train_step(self, data):
        (text_embeds, params_in), y_true = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder([text_embeds, params_in])
            eps = tf.random.normal(shape=tf.shape(z_mean))
            z = z_mean + tf.exp(0.5 * z_log_var) * eps
            mask = tf.cast(tf.random.uniform((tf.shape(z)[0], 1)) >= self.latent_dropout_rate, tf.float32)
            z_dec = z * mask
            y_pred = self.decoder([z_dec, text_embeds], training=True)
            total, recon, kl = self.calculate_loss(y_true, y_pred, z_mean, z_log_var)

        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss_tracker.update_state(total)
        self.reconstruction_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        return {"loss": self.total_loss_tracker.result(), "recon": recon, "kl": kl}

    def test_step(self, data):
        (text_embeds, params_in), y_true = data
        z_mean, z_log_var = self.encoder([text_embeds, params_in])
        eps = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * eps
        y_pred = self.decoder([z, text_embeds], training=False)
        total, recon, kl = self.calculate_loss(y_true, y_pred, z_mean, z_log_var)

        self.total_loss_tracker.update_state(total)
        self.reconstruction_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        return {"loss": self.total_loss_tracker.result(), "recon": recon, "kl": kl}

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


# -------------------------------------------------------------------------
# Latent Diffusion Model (Stage 2: Generation)
# -------------------------------------------------------------------------

@register_keras_serializable(package="custom", name="LatentDiffusionModel")
class LatentDiffusionModel(tf.keras.Model):
    """ Stage 2: Latent Diffusion Model (Updated with CFG) """
    def __init__(self, vae_encoder, vae_decoder, denoiser, timesteps=1000, **kwargs):
        super().__init__(**kwargs)
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.denoiser = denoiser
        self.timesteps = int(timesteps)
        self.scheduler = DiffusionScheduler(timesteps=self.timesteps)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs, training=False):
        # Keras requirement. Expected Inputs: [z_t, t, text_embeds]
        return self.denoiser(inputs, training=training)

    def train_step(self, data):
        (text_embeds, params_in), _ = data
        batch_size = tf.shape(params_in)[0]

        # 1. Encode with Frozen VAE
        z_mean, z_log_var = self.vae_encoder([text_embeds, params_in], training=False)
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z_0 = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        # 2. Add Noise
        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int32)
        noise = tf.random.normal(shape=tf.shape(z_0))
        z_t = self.scheduler.add_noise(z_0, noise, t)

        # 3. NEW: Classifier-Free Guidance (CFG) - 10% dropout
        cfg_prob = 0.1
        mask_indices = tf.random.uniform((batch_size,)) < cfg_prob
        null_embeds = tf.zeros_like(text_embeds)
        mask_indices = tf.reshape(mask_indices, [-1, 1])
        final_text_embeds = tf.where(mask_indices, null_embeds, text_embeds)

        # 4. Train Denoiser
        with tf.GradientTape() as tape:
            pred_noise = self.denoiser([z_t, t, final_text_embeds], training=True)
            loss = tf.reduce_mean(tf.square(noise - pred_noise))

        gradients = tape.gradient(loss, self.denoiser.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.denoiser.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        (text_embeds, params_in), _ = data
        batch_size = tf.shape(params_in)[0]
        z_mean, z_log_var = self.vae_encoder([text_embeds, params_in], training=False)
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z_0 = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int32)
        noise = tf.random.normal(shape=tf.shape(z_0))
        z_t = self.scheduler.add_noise(z_0, noise, t)
        pred_noise = self.denoiser([z_t, t, text_embeds], training=False)
        loss = tf.reduce_mean(tf.square(noise - pred_noise))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @tf.function(jit_compile=True)
    def _diffusion_loop_compiled(self, z, text_embeds, timestep_indices):
        """JIT-compiled diffusion loop for GPU efficiency"""
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

            if i > 0:
                noise = tf.random.normal(shape=tf.shape(z))
                sigma = tf.sqrt(beta)
                z = model_mean + sigma * noise
            else:
                z = model_mean
        
        return z

    @tf.function(jit_compile=True)
    def _cfg_diffusion_loop_compiled(self, z, text_embeds, uncond_embeds, timestep_indices, guidance_scale):
        """JIT-compiled CFG diffusion loop for GPU efficiency"""
        for i in timestep_indices:
            batch_size = tf.shape(text_embeds)[0]
            t = tf.ones((batch_size,), dtype=tf.int32) * i

            # CFG: Batch conditional and unconditional together
            latents_in = tf.concat([z, z], axis=0)
            t_in = tf.concat([t, t], axis=0)
            text_in = tf.concat([text_embeds, uncond_embeds], axis=0)

            # Predict noise
            noise_preds = self.denoiser([latents_in, t_in, text_in], training=False)

            # Split and apply guidance
            noise_cond, noise_uncond = tf.split(noise_preds, num_or_size_splits=2, axis=0)
            pred_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            # Standard scheduler math
            alpha = tf.gather(self.scheduler.alphas, i)
            alpha_cumprod = tf.gather(self.scheduler.alphas_cumprod, i)
            beta = tf.gather(self.scheduler.betas, i)
            sqrt_one_minus_alpha_cumprod = tf.sqrt(1.0 - alpha_cumprod)

            model_mean = (1 / tf.sqrt(alpha)) * (z - ((1 - alpha) / (sqrt_one_minus_alpha_cumprod)) * pred_noise)

            if i > 0:
                noise = tf.random.normal(shape=tf.shape(z))
                sigma = tf.sqrt(beta)
                z = model_mean + sigma * noise
            else:
                z = model_mean
        
        return z

    def generate(self, text_embeds, steps=50, guidance_scale=7.5):
        """
        Generates latents using Classifier-Free Guidance (CFG) with JIT compilation.

        Args:
            text_embeds: The CLAP embeddings for the prompt.
            steps: Number of denoising steps.
            guidance_scale: Strength of text adherence.
                            1.0 = Standard Diffusion (No guidance)
                            7.5 = Stable Diffusion Standard (Sharp, accurate)
        """
        # Ensure input is a tensor
        text_embeds = tf.convert_to_tensor(text_embeds, dtype=tf.float32)
        batch_size = tf.shape(text_embeds)[0]
        latent_dim = self.vae_encoder.output_shape[0][1]

        # 1. Start with random noise
        z = tf.random.normal(shape=(batch_size, latent_dim))

        # 2. Prepare Unconditional (Null) Embeddings
        uncond_embeds = tf.zeros_like(text_embeds)

        # 3. Determine timesteps to use
        if steps is None or steps >= self.timesteps:
            timestep_indices = tf.range(self.timesteps - 1, -1, -1, dtype=tf.int32)
        else:
            step_ratio = self.timesteps // steps
            timestep_indices = tf.range(self.timesteps - 1, -1, -step_ratio, dtype=tf.int32)

        # 4. Run JIT-compiled CFG diffusion loop
        guidance_scale_tensor = tf.constant(guidance_scale, dtype=tf.float32)
        z = self._cfg_diffusion_loop_compiled(z, text_embeds, uncond_embeds, timestep_indices, guidance_scale_tensor)

        # 5. Decode the final clean latent
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

"""## Define Model Builders"""


# -------------------------------------------------------------------------
# Parameter Reconstruction
# -------------------------------------------------------------------------

def _safe_probs(raw_head):
    """Take a model categorical head, flatten it, clip negatives, and normalize."""
    probs = np.array(raw_head, dtype=np.float32).reshape(-1)
    probs[probs < 0] = 0.0
    s = probs.sum()
    if not np.isfinite(s) or s <= 1e-6:
        probs = np.ones_like(probs, dtype=np.float32) / probs.size
    else:
        probs = probs / s
    return probs


def reconstruct_parameters_from_heads(predicted_outputs, parameter_types, categorical_num_classes, sample_categorical=True, noise_level=0.0):
    """
    Reconstructs parameter vector from split heads.
    """
    n_params = len(parameter_types)
    reconstructed = np.zeros(n_params, dtype=np.float32)

    # Extract indices based on parameter definition
    continuous_params = [int(i) for i, p in parameter_types.items() if p["type"] == "continuous"]
    mod_matrix_ids = set(range(170, 202))

    unipolar_indices = sorted([i for i in continuous_params if i not in mod_matrix_ids])
    bipolar_indices = sorted([i for i in continuous_params if i in mod_matrix_ids])
    boolean_params = sorted([int(i) for i, p in parameter_types.items() if p["type"] == "boolean"])
    categorical_params = sorted([int(i) for i, p in parameter_types.items() if p["type"] == "categorical"])

    head_idx = 0

    # 1. Unipolar Continuous
    if unipolar_indices:
        # Flatten predictions
        uni_head = np.array(predicted_outputs[head_idx], dtype=np.float32).reshape(-1)
        for i, param_idx in enumerate(unipolar_indices):
            reconstructed[param_idx] = uni_head[i]
        head_idx += 1

    # 2. Bipolar (Gate + Value)
    if bipolar_indices:
        gate_head = np.array(predicted_outputs[head_idx], dtype=np.float32).reshape(-1)
        val_head = np.array(predicted_outputs[head_idx+1], dtype=np.float32).reshape(-1)
        head_idx += 2
        for i, param_idx in enumerate(bipolar_indices):
            # Sigmoid for gate
            gate = 1.0 / (1.0 + np.exp(-gate_head[i]))
            # Tanh -> 0..1 for value
            val = (val_head[i] + 1.0) / 2.0

            if gate < 0.25:
                reconstructed[param_idx] = 0.5 # Default center
            else:
                reconstructed[param_idx] = val

    # 3. Boolean
    if boolean_params:
        bool_head = np.array(predicted_outputs[head_idx], dtype=np.float32).reshape(-1)
        head_idx += 1
        for i, param_idx in enumerate(boolean_params):
            reconstructed[param_idx] = 1.0 if bool_head[i] > 0.5 else 0.0

    # 4. Categorical
    for param_idx in categorical_params:
        head = predicted_outputs[head_idx]
        if isinstance(head, (list, tuple)): head = head[0]
        probs = _safe_probs(head)
        num_c = categorical_num_classes[param_idx]

        if sample_categorical:
            choice = np.random.choice(len(probs), p=probs)
        else:
            choice = np.argmax(probs)

        if num_c > 1:
            reconstructed[param_idx] = float(choice) / (num_c - 1)
        else:
            reconstructed[param_idx] = 0.0
        head_idx += 1

    return reconstructed


# Common modulation routings - based on training data analysis
# Maps mod slot index (0-31) to destination parameter ID
COMMON_MOD_ROUTINGS = {
    # Group 1: Volume & Mix (Rhythm & Dynamics)
    0: "2",   # LFO1 -> Oscillator A.kParamVolume
    1: "22",  # LFO1 -> Oscillator B.kParamVolume
    2: "42",  # LFO1 -> Oscillator Sub.kParamVolume
    3: "43",  # LFO1 -> Oscillator Noise.kParamVolume
    4: "2",   # LFO2 -> Oscillator A.kParamVolume (duplicate dest!)
    5: "43",  # Env2 -> Oscillator Noise.kParamVolume (duplicate dest!)
    
    # Group 2: Wavetable Motion (Timbre Evolution)
    6: "5",   # LFO1 -> WTOsc A.kParamTablePos
    7: "6",   # LFO1 -> WTOsc A.kParamWarp
    8: "25",  # LFO1 -> WTOsc B.kParamTablePos
    9: "26",  # LFO1 -> WTOsc B.kParamWarp
    10: "5",  # LFO2 -> WTOsc A.kParamTablePos (duplicate dest!)
    11: "6",  # LFO2 -> WTOsc A.kParamWarp (duplicate dest!)
    12: "5",  # Env2 -> WTOsc A.kParamTablePos (duplicate dest!)
    13: "25", # LFO2 -> WTOsc B.kParamTablePos (duplicate dest!)
    14: "26", # LFO2 -> WTOsc B.kParamWarp (duplicate dest!)
    15: "6",  # Env2 -> WTOsc A.kParamWarp (duplicate dest!)
    
    # Group 3: Voice Filter (Main Filter Movement)
    16: "45", # LFO1 -> VoiceFilter.kParamFreq (#1 most common!)
    17: "45", # Env2 -> VoiceFilter.kParamFreq (#2, duplicate dest!)
    18: "45", # Env1 -> VoiceFilter.kParamFreq (duplicate dest!)
    19: "45", # LFO2 -> VoiceFilter.kParamFreq (duplicate dest!)
    20: "46", # LFO1 -> VoiceFilter.kParamVar
    21: "47", # LFO1 -> VoiceFilter.kParamReso
    
    # Group 4: FX & Post-Processing (Effects Modulation)
    22: "67", # LFO1 -> FXFilter.kParamFreq
    23: "70", # LFO1 -> FXDistortion.kParamDrive
    24: "67", # LFO2 -> FXFilter.kParamFreq (duplicate dest!)
    25: "67", # Env2 -> FXFilter.kParamFreq (duplicate dest!)
    26: "71", # LFO1 -> FXDistortion.kParamFreq
    27: "53", # Env1 -> FXReverb.kParamWet
    28: "63", # LFO1 -> FXHyperD.kParamWet
    29: "53", # LFO1 -> FXReverb.kParamWet (duplicate dest!)
    30: "72", # LFO1 -> FXDistortion.kParamWet
    31: "70", # Env2 -> FXDistortion.kParamDrive (duplicate dest!)
}


def scale_overlapping_mod_amounts(params):
    """
    Scale down modulation amounts when multiple active sources target the same destination.
    Uses the predefined COMMON_MOD_ROUTINGS mapping.
    Only scales modulations that are actually active (not at center 0.5).
    
    Modulation amounts are bipolar with 0.5 as center (no modulation).
    Formula: new_amount = 0.5 + (old_amount - 0.5) / num_active_sources_to_same_dest
    
    Args:
        params: Parameter array of shape (202,)
    
    Returns:
        Modified params array with scaled modulation amounts
    """
    # Mod matrix amounts are at indices 170-201 (32 slots)
    mod_start_idx = 170
    
    # Find active mods and group by destination
    active_mods_by_destination = {}
    
    for slot_idx in range(32):
        param_idx = mod_start_idx + slot_idx
        mod_amount = params[param_idx]
        
        # Check if this mod is active (not at center 0.5)
        # Use small epsilon for floating point comparison
        is_active = abs(mod_amount - 0.5) > 0.01
        
        if is_active and slot_idx in COMMON_MOD_ROUTINGS:
            dest_id = COMMON_MOD_ROUTINGS[slot_idx]
            
            if dest_id not in active_mods_by_destination:
                active_mods_by_destination[dest_id] = []
            active_mods_by_destination[dest_id].append({
                'slot_idx': slot_idx,
                'param_idx': param_idx,
                'amount': mod_amount
            })
    
    # Scale amounts for destinations with multiple active sources
    params_scaled = params.copy()
    for dest_id, mods in active_mods_by_destination.items():
        num_active = len(mods)
        if num_active > 1:
            # Multiple active sources targeting same destination
            for mod_info in mods:
                param_idx = mod_info['param_idx']
                old_amount = mod_info['amount']
                # Scale from center (0.5): divide the deviation by num_active
                new_amount = 0.5 + (old_amount - 0.5) / num_active
                params_scaled[param_idx] = new_amount
    
    return params_scaled






# -------------------------------------------------------------------------
# Main VAE_V2P7 Wrapper Class
# -------------------------------------------------------------------------

class VAE_V2P7:
    """
    VAE V2.7 wrapper class for generating synthesizer parameters using Latent Diffusion.
    
    Two-stage architecture:
    1. VAE compresses 202-dim parameters to 128-dim latent space
    2. Diffusion model generates latents via 1000-step denoising
    
    Args:
        model_path: Path to the trained LatentDiffusionModel file (.keras)
        embedding_model_type: Type of text embedding model ("clap" or "sentence_transformer")
        default_diffusion_steps: Default number of denoising steps (default: 1000)
        mod_gate_threshold: Threshold for modulation matrix gate activation (default 0.25)
    
    Usage:
        vae = VAE_V2P7(model_path="/path/to/ldm_final.keras")
        params = vae.generate("deep wobbling bass")
    """
    
    def __init__(self, model_path, embedding_model_type="clap", 
                 default_diffusion_steps=1000, mod_gate_threshold=0.25):
        """Initialize VAE_V2P7 model."""
        self.embedding_model_type = embedding_model_type.lower()
        self.default_diffusion_steps = default_diffusion_steps
        self.model_path = model_path
        self.mod_gate_threshold = mod_gate_threshold
        
        # Validate embedding model type
        if self.embedding_model_type not in ["clap", "sentence_transformer"]:
            raise ValueError(f"Invalid embedding_model_type: {embedding_model_type}")
        
        # Validate mod_gate_threshold
        if not 0.0 <= mod_gate_threshold <= 1.0:
            raise ValueError(f"mod_gate_threshold must be between 0.0 and 1.0, got: {mod_gate_threshold}")
        
        # Validate model path
        if not os.path.isabs(model_path):
            raise ValueError(f"model_path must be an absolute path, got: {model_path}")
        
        print(f"VAE_V2P7 initialized with model_path={model_path}, embedding_model={embedding_model_type}, diffusion_steps={default_diffusion_steps}")
        
        self._ldm_model = None
        self._embedding_loaded = False
    
    def _load_model(self):
        """Load the Latent Diffusion Model."""
        if self._ldm_model is not None:
            return self._ldm_model
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Please check that the model file exists at this location."
            )
        
        print(f"Loading Latent Diffusion Model from {self.model_path}")
        self._ldm_model = keras.models.load_model(
            self.model_path,
            custom_objects={
                "LatentDiffusionModel": LatentDiffusionModel,
                "VAE_Text_to_Synth_Standard": VAE_Text_to_Synth_Standard,
                "SliceLayer": SliceLayer,
                "FiLMLayer": FiLMLayer,
                "SinusoidalTimeEmbedding": SinusoidalTimeEmbedding,
                "FiLM_Modulate": FiLM_Modulate,
                "ResidualBlock": ResidualBlock,
            },
            compile=False,
            safe_mode=False,
        )
        print(f"Latent Diffusion Model loaded successfully")
        
        return self._ldm_model
    
    def _load_embedding_model(self):
        """Load the embedding model."""
        if self._embedding_loaded:
            return
        
        if self.embedding_model_type == "clap":
            load_clap_model()
        else:
            get_sentence_transformer_model()
        
        self._embedding_loaded = True
    
    def generate(self, text_description, diffusion_steps=None, head_noise_level=0.0,
                 sample_categorical=True, seed=None, guidance_scale=7.5, verbose=True):
        """Generate synthesizer parameters from text description using latent diffusion.
        
        Args:
            text_description: Text prompt (string) or list of text prompts describing the desired sound(s)
            diffusion_steps: Number of denoising steps (default: use model's default, typically 1000)
            head_noise_level: Noise level to add to output heads (typically 0.0)
            sample_categorical: If True, sample from categorical distributions; else argmax
            seed: Random seed for reproducibility
            guidance_scale: CFG guidance scale (1.0 = no guidance, 7.5 = standard, higher = more adherence)
            verbose: Print debug information
        
        Returns:
            np.ndarray: Parameter array of shape (202,) for single prompt, or (batch_size, 202) for list of prompts
                       Returns None on error
        """
        if seed is not None:
            np.random.seed(seed)
            import random
            random.seed(seed)
            tf.random.set_seed(seed)
            try:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except Exception:
                pass
        
        ldm_model = self._load_model()
        self._load_embedding_model()
        
        # Determine if input is a batch or single prompt
        is_batch = isinstance(text_description, (list, tuple))
        if is_batch:
            batch_size = len(text_description)
            prompts = text_description
        else:
            batch_size = 1
            prompts = [text_description]
        
        try:
            # Encode all prompts at once (batch encoding is more efficient)
            if verbose:
                print(f"Encoding {batch_size} prompt(s) with {self.embedding_model_type}...")
            
            # encode_text handles both single strings and lists
            # Returns shape: (batch_size, embedding_dim)
            emb = encode_text(prompts, self.embedding_model_type)
            
            if verbose:
                print(f"Text encoded with {self.embedding_model_type}. Shape: {emb.shape}")
        except Exception as e:
            print(f"Error encoding text: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 1:
            emb = np.expand_dims(emb, axis=0)
        
        try:
            # Run diffusion generation (1000-step denoising loop)
            if verbose:
                steps_to_use = diffusion_steps if diffusion_steps is not None else ldm_model.timesteps
                print(f"Running {steps_to_use}-step diffusion sampling for batch_size={batch_size}...")
            
            predicted_outputs = ldm_model.generate(emb, steps=diffusion_steps or ldm_model.timesteps, guidance_scale=guidance_scale)
            
            if not isinstance(predicted_outputs, (list, tuple)):
                predicted_outputs = [predicted_outputs]
            
            if verbose:
                print(f"Decoder outputs: {len(predicted_outputs)} heads")
                for i, head in enumerate(predicted_outputs):
                    print(f"  Head {i}: shape {np.array(head).shape}")
            
        except Exception as e:
            print(f"Error running diffusion model: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        try:
            # Process each sample in the batch
            all_params = []
            for batch_idx in range(batch_size):
                # Extract outputs for this batch item
                batch_outputs = []
                for head in predicted_outputs:
                    head_array = np.array(head)
                    if head_array.ndim >= 2:
                        # Extract batch item: head[batch_idx]
                        batch_outputs.append(head_array[batch_idx])
                    else:
                        # Single sample case
                        batch_outputs.append(head_array)
                
                params = reconstruct_parameters_from_heads(
                    batch_outputs, FLAT_PARAMETER_TYPES, CATEGORICAL_NUM_CLASSES,
                    sample_categorical=sample_categorical, noise_level=head_noise_level
                )
                
                # Apply automatic mod matrix scaling for overlapping destinations
                params = scale_overlapping_mod_amounts(params)
                all_params.append(params)
            
            if verbose:
                print("Applied mod matrix overlap scaling")
            
            # Return single array for single prompt, batch array for multiple prompts
            result = np.array(all_params)
            if not is_batch:
                result = result[0]
            
            if verbose:
                print(f"Reconstructed shape: {result.shape}")
                if result.ndim == 1:
                    print(f"Sample values (first 10): {result[:10]}")
                else:
                    print(f"Sample values (first sample, first 10): {result[0, :10]}")
            
            return result
        except Exception as e:
            print(f"Error reconstructing parameters: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def __repr__(self):
        return f"VAE_V2P7(model_path={self.model_path}, embedding={self.embedding_model_type}, diffusion_steps={self.default_diffusion_steps})"
