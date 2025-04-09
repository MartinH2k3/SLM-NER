import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def prepare_inputs(messages, image_paths, tokenizer, device='cuda', dtype=torch.bfloat16):
    """
    Prepares multi-modal input components (supports multiple images per prompt).

    Args:
        messages: List of input messages/prompts (strings or dicts with 'role' and 'content')
        image_paths: List where each element is either None (for text-only) or a list of image paths
        tokenizer: The tokenizer to use for applying chat templates
        device: Device to place tensors on ('cuda', 'cpu', etc.)
        dtype: Data type for image tensors (default: torch.bfloat16)

    Returns:
        dict: Contains 'prompts', 'pixel_values_list', and 'num_patches_list' ready for the model
    """
    # Make sure image_paths list is at least as long as messages
    if len(image_paths) < len(messages):
        # Pad with None for text-only messages
        image_paths = image_paths + [None] * (len(messages) - len(image_paths))

    # Process images and collect patch information
    loaded_images = []
    num_patches_list = []
    for paths in image_paths:
        if paths and isinstance(paths, list) and len(paths) > 0:
            # Load each image in this prompt
            prompt_images = []
            prompt_patches = []

            for path in paths:
                # Load the image
                img = load_image(path).to(dtype=dtype, device=device)

                # Ensure img has correct shape [patches, C, H, W]
                if len(img.shape) == 3:  # [C, H, W] -> [1, C, H, W]
                    img = img.unsqueeze(0)

                prompt_images.append(img)
                # Record the number of patches for this image
                prompt_patches.append(img.shape[0])

            loaded_images.append(prompt_images)
            num_patches_list.append(prompt_patches)
        else:
            # Text-only prompt
            loaded_images.append(None)
            num_patches_list.append([])

    # Create the concatenated pixel_values_list
    pixel_values_list = []
    for prompt_images in loaded_images:
        if prompt_images:
            # Concatenate all images for this prompt
            pixel_values_list.append(torch.cat(prompt_images, dim=0))
        else:
            # Text-only prompt
            pixel_values_list.append(None)

    # Format messages for the model
    if all(isinstance(m, str) for m in messages):
        # Simple string messages: convert to chat format
        batch_messages = [
            [{"role": "user", "content": message}]
            for message in messages
        ]
    else:
        # Assume messages are already in the right format
        batch_messages = messages

    # Apply chat template
    prompts = tokenizer.apply_chat_template(
        batch_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return {
        'prompts': prompts,
        'pixel_values_list': pixel_values_list,
        'num_patches_list': num_patches_list
    }


def construct_message(text, template, examples=None):
    """
    Construct the individual NuExtract message texts, prior to chat template formatting.
    """
    # add few-shot examples if needed
    if examples is not None and len(examples) > 0:
        icl = "# Examples:\n"
        for row in examples:
            icl += f"## Input:\n{row['input']}\n## Output:\n{row['output']}\n"
    else:
        icl = ""

    return f"""# Template:\n{template}\n{icl}# Context:\n{text}"""


IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'


def nuextract_generate(model, tokenizer, prompts, generation_config, pixel_values_list=None, num_patches_list=None):
    """
    Generate responses for a batch of NuExtract inputs.
    Support for multiple and varying numbers of images per prompt.

    Args:
        model: The vision-language model
        tokenizer: The tokenizer for the model
        pixel_values_list: List of tensor batches, one per prompt
                          Each batch has shape [num_images, channels, height, width] or None for text-only prompts
        prompts: List of text prompts
        generation_config: Configuration for text generation
        num_patches_list: List of lists, each containing patch counts for images in a prompt

    Returns:
        List of generated responses
    """
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    # Replace all image placeholders with appropriate tokens
    modified_prompts = []
    total_image_files = 0
    total_patches = 0
    image_containing_prompts = []
    for idx, prompt in enumerate(prompts):
        # check if this prompt has images
        has_images = (pixel_values_list and
                      idx < len(pixel_values_list) and
                      pixel_values_list[idx] is not None and
                      isinstance(pixel_values_list[idx], torch.Tensor) and
                      pixel_values_list[idx].shape[0] > 0)

        if has_images:
            # prompt with image placeholders
            image_containing_prompts.append(idx)
            modified_prompt = prompt

            patches = num_patches_list[idx] if (num_patches_list and idx < len(num_patches_list)) else []
            num_images = len(patches)
            total_image_files += num_images
            total_patches += sum(patches)

            # replace each <image> placeholder with image tokens
            for i, num_patches in enumerate(patches):
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
                modified_prompt = modified_prompt.replace('<image>', image_tokens, 1)
        else:
            # text-only prompt
            modified_prompt = prompt

        modified_prompts.append(modified_prompt)

    # process all prompts in a single batch
    tokenizer.padding_side = 'left'
    model_inputs = tokenizer(modified_prompts, return_tensors='pt', padding=True)
    input_ids = model_inputs['input_ids'].to(model.device)
    attention_mask = model_inputs['attention_mask'].to(model.device)

    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>\n".strip())
    generation_config['eos_token_id'] = eos_token_id

    # prepare pixel values
    flattened_pixel_values = None
    if image_containing_prompts:
        # collect and concatenate all image tensors
        all_pixel_values = []
        for idx in image_containing_prompts:
            all_pixel_values.append(pixel_values_list[idx])

        flattened_pixel_values = torch.cat(all_pixel_values, dim=0)
        print(
            f"Processing batch with {len(prompts)} prompts, {total_image_files} actual images, and {total_patches} total patches")
    else:
        print(f"Processing text-only batch with {len(prompts)} prompts")

    # generate outputs
    outputs = model.generate(
        pixel_values=flattened_pixel_values,  # will be None for text-only prompts
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_config
    )

    # Decode responses
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return responses
