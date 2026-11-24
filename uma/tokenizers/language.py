import collections
from typing import Any, Dict, Optional, Tuple, Union, List, Literal
import torch
from torch.nn import functional as F
from transformers import AutoImageProcessor, AutoTokenizer, AutoModel

from uma.policy.input_output import UMAContextItem

# =============================================================================
# Encoder Type Definition
# =============================================================================
EncoderType = Literal["siglip2", "none"]

# =============================================================================
# SigLIP2 Constants
# =============================================================================
SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP2_IMAGE_PATCH_SIZE = 16
SIGLIP2_IMAGE_INPUT_SIZE = 224
SIGLIP2_EMBEDDING_DIM = 768

# =============================================================================
# Common Constants (the default encoder, SigLIP2)
# =============================================================================
TOKENIZER_MODEL_ID = SIGLIP2_MODEL_ID
TOKENIZER_IMAGE_PATCH_SIZE = SIGLIP2_IMAGE_PATCH_SIZE
TOKENIZER_IMAGE_INPUT_SIZE = SIGLIP2_IMAGE_INPUT_SIZE
TOKENIZER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def resolve_tokenizer_device(
    device: Optional[Union[torch.device, str]] = None,
) -> torch.device:
    """Resolve the device used for tokenizer model execution."""
    if device is not None:
        return torch.device(device)
    return torch.device(TOKENIZER_DEVICE)


def _cache_key_for_device(device: Union[torch.device, str]) -> str:
    return str(torch.device(device))

# =============================================================================
# Helper Functions for Embedding Dimensions
# =============================================================================

def get_language_embedding_dim(encoder_type: EncoderType) -> int:
    """Get the text embedding dimension for a given encoder type."""
    if encoder_type == "siglip2":
        return SIGLIP2_EMBEDDING_DIM
    elif encoder_type == "none":
        return 0
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def get_image_embedding_dim(encoder_type: EncoderType) -> int:
    """Get the image/vision embedding dimension for a given encoder type."""
    if encoder_type == "siglip2":
        return SIGLIP2_EMBEDDING_DIM
    elif encoder_type == "none":
        return 0
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def get_patch_size(encoder_type: EncoderType) -> int:
    """Get the patch size for a given encoder type."""
    if encoder_type == "siglip2":
        return SIGLIP2_IMAGE_PATCH_SIZE
    elif encoder_type == "none":
        return 14
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# =============================================================================
# Model Loading Functions
# =============================================================================

def get_siglip2_models(
    device: Optional[Union[torch.device, str]] = None,
):
    """Load and cache SigLIP2 models."""
    resolved_device = resolve_tokenizer_device(device)
    cache_key = _cache_key_for_device(resolved_device)
    if not hasattr(get_siglip2_models, "cached_models"):
        get_siglip2_models.cached_models = {}
    if cache_key in get_siglip2_models.cached_models:
        return get_siglip2_models.cached_models[cache_key]
    text_tokenizer = AutoTokenizer.from_pretrained(SIGLIP2_MODEL_ID)
    image_processor = AutoImageProcessor.from_pretrained(SIGLIP2_MODEL_ID, use_fast=True)
    model = AutoModel.from_pretrained(SIGLIP2_MODEL_ID, output_hidden_states=True)
    model = model.eval().to(resolved_device)
    get_siglip2_models.cached_models[cache_key] = (text_tokenizer, image_processor, model)
    return get_siglip2_models.cached_models[cache_key]

def get_empty_language_token(
    batch_size: int,
    device: Optional[Union[torch.device, str]] = None,
    encoder_type: EncoderType = "siglip2",
    max_language_length: int = 64,
) -> UMAContextItem:
    """Create empty language tokens with the correct embedding dimension."""
    embed_dim = get_language_embedding_dim(encoder_type)
    
    return UMAContextItem.create(
        token_data=torch.zeros((batch_size, max_language_length, embed_dim), dtype=torch.float32, device=device),
        sequence_shape=(batch_size, max_language_length),
        empty=True
    )


# =============================================================================
# Text Tokenization Functions
# =============================================================================
@torch.compiler.disable
def tokenize_languages(
    texts: List[str],
    max_language_length: int,
    device: Optional[Union[torch.device, str]] = None,
    encoder_type: EncoderType = "siglip2"
) -> UMAContextItem:
    """
    Tokenize text using the specified encoder.
    
    Args:
        texts: List of text strings to tokenize
        max_language_length: Maximum sequence length
        device: Device to place tensors on
        encoder_type: Which encoder to use ("siglip2", or "none" for no features)
    
    Returns:
        UMAContextItem with text features
        - siglip2: (B, max_language_length, 768)
    """
    if all(len(t.strip()) == 0 for t in texts) or encoder_type == "none":
        return get_empty_language_token(
            batch_size=len(texts),
            device=device,
            encoder_type=encoder_type,
            max_language_length=max_language_length
        )

    if encoder_type == "siglip2":
        return _tokenize_languages_siglip2(texts, max_language_length, device)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


#: Strings whose SigLIP2 text features stay cached (per device and length); ~0.2 MB each.
SIGLIP2_TEXT_CACHE_SIZE = 2048
_SIGLIP2_TEXT_CACHE: "collections.OrderedDict[Any, Tuple[torch.Tensor, torch.Tensor]]" = collections.OrderedDict()


def _tokenize_languages_siglip2(
    texts: List[str],
    max_language_length: int,
    device: Optional[Union[torch.device, str]] = None
) -> UMAContextItem:
    """Tokenize text using SigLIP2. Output: (B, max_language_length, 768)

    The text encoder is frozen (eval, no grad), so each string's features are
    cached per device and length (LRU, :data:`SIGLIP2_TEXT_CACHE_SIZE`
    entries): a training step only tokenizes and encodes strings it has not
    seen, instead of the whole batch every step.
    """
    compute_device = resolve_tokenizer_device(device)
    text_tokenizer, image_processor, model = get_siglip2_models(compute_device)

    cache = _SIGLIP2_TEXT_CACHE
    missing = list(dict.fromkeys(t for t in texts if (compute_device, max_language_length, t) not in cache))
    if missing:
        text_tokenized = text_tokenizer(
            missing,
            padding='max_length',
            truncation=True,
            max_length=max_language_length,
            return_tensors='pt',
            return_attention_mask=True
        )
        text_tokenized = {k: v.pin_memory().to(compute_device, non_blocking=True) for k, v in text_tokenized.items()}
        with torch.no_grad():
            new_features = model.text_model(
                **text_tokenized,
                return_dict=True
            ).last_hidden_state  # [n_missing, seq_len, 768]
        new_masks = text_tokenized['attention_mask'].to(torch.bool)
        for i, text in enumerate(missing):
            cache[(compute_device, max_language_length, text)] = (new_features[i], new_masks[i])
    keys = [(compute_device, max_language_length, t) for t in texts]
    for key in keys:
        cache.move_to_end(key)
    text_features = torch.stack([cache[key][0] for key in keys], dim=0)  # [batch_size, seq_len, 768]
    text_mask = torch.stack([cache[key][1] for key in keys], dim=0)  # [batch_size, seq_len]
    while len(cache) > SIGLIP2_TEXT_CACHE_SIZE:
        cache.popitem(last=False)

    if device is not None:
        text_features = text_features.to(device)
        text_mask = text_mask.to(device)

    assert text_features.shape[-1] == SIGLIP2_EMBEDDING_DIM, \
        f"Text features dimension {text_features.shape[-1]} does not match SIGLIP2_EMBEDDING_DIM={SIGLIP2_EMBEDDING_DIM}"
    assert text_features.shape[:-1] == text_mask.shape
    assert text_features.shape[1] == max_language_length, "Text features length does not match max_language_length"
    
    return UMAContextItem.create(
        token_data=text_features,
        valid_mask=text_mask,
        sequence_shape=text_mask.shape
    )


@torch.compiler.disable
def tokenize_patch_features(
    images: torch.Tensor, 
    device: Optional[Union[torch.device, str]] = None,
    upscale: bool = False,
    encoder_type: EncoderType = "siglip2"
) -> torch.Tensor:
    """
    Extract patch features from images using the specified encoder.
    
    Args:
        images: RGB images of shape (B, 3, H, W) with uint8 values [0, 255] or with float values [0, 1]
        device: Device to place tensors on
        upscale: Whether to upsample features to original image resolution
        encoder_type: Which encoder to use ("siglip2", or "none" for no features)
        
    Returns:
        Patch features tensor:
        - siglip2: (B, H', W', 768) where H'=14, W'=14 (or original res if upscale)
    """
    if encoder_type == "siglip2":
        return _tokenize_patch_features_siglip2(images, device, upscale)
    elif encoder_type == "none":
        batch_size, _, H, W = images.shape
        return torch.empty((batch_size, H, W, 0), dtype=torch.float32, device=device) if upscale else torch.empty((batch_size, 14, 14, 0), dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def _siglip2_pixel_values(images: torch.Tensor, image_processor: Any) -> torch.Tensor:
    """The SigLIP2 image processor's ``pixel_values`` for ``(B, 3, H, W)`` images, on their device.

    Same steps as the (fast) processor: resize to its size (only when the images
    differ from it; bilinear, antialiased), rescale ``uint8`` by 1/255 (float
    images are taken as already in ``[0, 1]``) and normalize with its mean/std.
    Calling the processor instead costs per-call validation, grouping and host
    syncs on every training step.
    """
    from torchvision.transforms.v2 import functional as TF

    height, width = image_processor.size["height"], image_processor.size["width"]
    x = images
    if tuple(x.shape[-2:]) != (height, width):
        x = TF.resize(x, [height, width], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    x = x.float() / 255.0 if images.dtype == torch.uint8 else x.float()
    key = (x.device, id(image_processor))
    stats = _SIGLIP2_NORM_STATS.get(key)
    if stats is None:
        stats = _SIGLIP2_NORM_STATS[key] = (
            torch.tensor(image_processor.image_mean, dtype=torch.float32, device=x.device).view(1, -1, 1, 1),
            torch.tensor(image_processor.image_std, dtype=torch.float32, device=x.device).view(1, -1, 1, 1),
        )
    mean, std = stats
    return (x - mean) / std


_SIGLIP2_NORM_STATS: Dict[Any, Tuple[torch.Tensor, torch.Tensor]] = {}


def _tokenize_patch_features_siglip2(
    images: torch.Tensor,
    device: Optional[Union[torch.device, str]] = None,
    upscale: bool = False
) -> torch.Tensor:
    """Extract patch features using SigLIP2 vision encoder. Output: (B, H', W', 768)"""
    compute_device = resolve_tokenizer_device(device)
    text_tokenizer, image_processor, model = get_siglip2_models(compute_device)
    
    batch_size = images.shape[0]
    original_height = images.shape[-2]
    original_width = images.shape[-1]
    
    # Resize to 224x224 and normalize. HF SigLIP/SigLIP2 processors expect
    # [0, 255] inputs when do_rescale=True, but the datasets usually pass float
    # images already in [0, 1], so those are not rescaled by 1/255 again.
    pixel_values = _siglip2_pixel_values(images.to(compute_device), image_processor)

    # Extract patch features from vision model
    with torch.no_grad():
        vision_outputs = model.vision_model(
            pixel_values=pixel_values,
            return_dict=True
        )
    
    # Get all patch embeddings (no CLS token in SigLIP2 vision model)
    patch_features = vision_outputs.last_hidden_state  # (B, num_patches, 768)
    hidden_dim = patch_features.shape[-1]
    
    assert hidden_dim == SIGLIP2_EMBEDDING_DIM, \
        f"Patch features dimension {hidden_dim} does not match SIGLIP2_EMBEDDING_DIM={SIGLIP2_EMBEDDING_DIM}"

    if device is not None:
        patch_features = patch_features.to(device)

    # Reshape from (B, num_patches, hidden_dim) to (B, H_patches, W_patches, hidden_dim)
    patches_per_side = SIGLIP2_IMAGE_INPUT_SIZE // SIGLIP2_IMAGE_PATCH_SIZE  # 14
    assert patch_features.shape[1] == patches_per_side ** 2, \
        f"Expected {patches_per_side**2} patches but got {patch_features.shape[1]}"
    
    patch_features_2d = patch_features.view(batch_size, patches_per_side, patches_per_side, hidden_dim)

    if upscale:
        patch_features = upsample_patch_features(patch_features_2d, original_height, original_width)
    else:
        patch_features = patch_features_2d

    return patch_features

def upsample_patch_features(
    features: torch.Tensor,
    target_height: int,
    target_width: int,
    upsampling_mode: str = 'bilinear'
) -> torch.Tensor:
    """Upsample patch features to the target height and width using bilinear interpolation.
    
    Args:
        features: (B, H_patches, W_patches, C) - Patch features in 2D grid format
        target_height: Target height in pixels
        target_width: Target width in pixels
        upsampling_mode: Interpolation mode ('bilinear', 'nearest', etc.)
        
    Returns:
        upsampled_features: (B, target_height, target_width, C) - Upsampled features
    """
    upsampled_features = F.interpolate(
        features.permute(0, 3, 1, 2),  # (B, C, H_patches, W_patches)
        size=(target_height, target_width),
        mode=upsampling_mode,
        align_corners=False if upsampling_mode == 'bilinear' else None
    )
    upsampled_features = upsampled_features.permute(0, 2, 3, 1)  # (B, target_height, target_width, C)
    return upsampled_features
