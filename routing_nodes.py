"""Routing nodes that replace groups of Any Switch / primitive nodes in the Modular workflow (V6).

Every "pick the first source that has something" rule works exactly like the rgthree Any Switch it replaces:
the first input that is connected AND not empty (None) wins. Muting a source group simply leaves that input empty.

  Image Sources    main image + image 2, with the per-prompt image first
  Encoder Images   the three images the Qwen text encoder gets (with the optional shot-type swap)
  Save Names       the save folder and file name: scheduler's names when it is on, else the typed ones
  Prompt Builder   prompt from the scheduler / a prompt file / the typed box, then the lighting line
"""
try:
    from .lighting_node import PRESETS, lighting_line, parse_by_shot, line_for_value, CUSTOM_DEFAULT
except ImportError:  # imported as a plain module (tests)
    from lighting_node import PRESETS, lighting_line, parse_by_shot, line_for_value, CUSTOM_DEFAULT


def first(*values):
    for v in values:
        if v is not None:
            return v
    return None


class ImageSources:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {}, 'optional': {
            'per_prompt_1': ('IMAGE', {'tooltip': 'Scheduler: the per-prompt image for slot 1 (empty when none).'}),
            'batch_image': ('IMAGE', {'tooltip': 'Load Image by Index (Source: Batch Folder).'}),
            'main_image': ('IMAGE', {'tooltip': 'The fixed main Load Image.'}),
            'per_prompt_2': ('IMAGE', {'tooltip': 'Scheduler: the per-prompt image for slot 2 (empty when none).'}),
            'ref_2': ('IMAGE', {'tooltip': 'The Ref 2 Load Image.'}),
        }}

    RETURN_TYPES = ('IMAGE', 'IMAGE')
    RETURN_NAMES = ('main_image', 'image_2')
    FUNCTION = 'run'
    CATEGORY = 'Claude/batch'

    def run(self, per_prompt_1=None, batch_image=None, main_image=None, per_prompt_2=None, ref_2=None):
        return (first(per_prompt_1, batch_image, main_image), first(per_prompt_2, ref_2))


class EncoderImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'swap_images_for': ('STRING', {'default': '', 'tooltip': 'A shot type (sub-folder name), for example faces. Prompts of that shot type get image 2 as image 1 and image 1 as image 2. Empty = never swap. Needs the scheduler\'s category.'}),
        }, 'optional': {
            'scaled_main': ('IMAGE', {'tooltip': 'The main image after the scale step.'}),
            'image_2': ('IMAGE', {'tooltip': 'Image 2 from Image Sources.'}),
            'per_prompt_3': ('IMAGE', {'tooltip': 'Scheduler: the per-prompt image for slot 3 (empty when none).'}),
            'ref_3': ('IMAGE', {'tooltip': 'The Ref 3 Load Image.'}),
            'category': ('STRING', {'forceInput': True, 'tooltip': 'The scheduler\'s category output (the shot type).'}),
        }}

    RETURN_TYPES = ('IMAGE', 'IMAGE', 'IMAGE')
    RETURN_NAMES = ('image_1', 'image_2', 'image_3')
    FUNCTION = 'run'
    CATEGORY = 'Claude/batch'

    def run(self, swap_images_for='', scaled_main=None, image_2=None, per_prompt_3=None, ref_3=None, category=None):
        swap = bool((swap_images_for or '').strip()) and (category or '').strip().lower() == swap_images_for.strip().lower()
        if swap and scaled_main is not None and image_2 is not None:
            scaled_main, image_2 = image_2, scaled_main
        return (scaled_main, image_2, first(per_prompt_3, ref_3))


class SaveNames:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'typed_folder': ('STRING', {'default': 'Qwen Edits', 'tooltip': 'Used when the scheduler is off.'}),
            'typed_file': ('STRING', {'default': 'Qwen Edit', 'tooltip': 'Used when the scheduler is off.'}),
        }, 'optional': {
            'folder': ('STRING', {'forceInput': True, 'tooltip': 'The scheduler\'s folder_prefix.'}),
            'file': ('STRING', {'forceInput': True, 'tooltip': 'The scheduler\'s filename_prefix.'}),
        }}

    RETURN_TYPES = ('STRING', 'STRING')
    RETURN_NAMES = ('folder', 'file')
    FUNCTION = 'run'
    CATEGORY = 'Claude/batch'

    def run(self, typed_folder, typed_file, folder=None, file=None):
        return (folder if folder else typed_folder, file if file else typed_file)


class PromptBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'lighting': (['none'] + list(PRESETS) + ['custom'], {'default': 'none', 'tooltip': 'Added to the end of every prompt that is not listed in lighting_by_shot. none = prompt untouched.'}),
            'custom_lighting': ('STRING', {'multiline': True, 'default': CUSTOM_DEFAULT}),
            'lighting_by_shot': ('STRING', {'multiline': True, 'default': '',
                                            'tooltip': 'Optional, with the scheduler: one per line, a shot type or prompt number = a preset name, none, or your own wording. For example: faces=soft studio   09=hard side light'}),
        }, 'optional': {
            'scheduler_prompt': ('STRING', {'forceInput': True, 'tooltip': 'The scheduler\'s prompt (first choice).'}),
            'index_prompt': ('STRING', {'forceInput': True, 'tooltip': 'Load Text by Index (second choice).'}),
            'typed_prompt': ('STRING', {'forceInput': True, 'tooltip': 'The typed Edit Prompt (last choice).'}),
            'category': ('STRING', {'forceInput': True, 'tooltip': 'The scheduler\'s category output.'}),
            'prompt_id': ('STRING', {'forceInput': True, 'tooltip': 'The scheduler\'s prompt_id output.'}),
        }}

    RETURN_TYPES = ('STRING', 'STRING')
    RETURN_NAMES = ('prompt', 'lighting_line')
    FUNCTION = 'run'
    CATEGORY = 'Claude/batch'

    def run(self, lighting, custom_lighting, lighting_by_shot='', scheduler_prompt=None, index_prompt=None, typed_prompt=None, category=None, prompt_id=None):
        base = first(scheduler_prompt, index_prompt, typed_prompt) or ''
        table = parse_by_shot(lighting_by_shot)
        key = next((k for k in ((prompt_id or '').strip().lower(), (category or '').strip().lower()) if k and k in table), None)
        line = line_for_value(table[key]) if key is not None else lighting_line(lighting, custom_lighting)
        text = base.rstrip()
        return ((text + ' ' + line).strip() if line else text, line)


NODE_CLASS_MAPPINGS = {'ImageSources': ImageSources, 'EncoderImages': EncoderImages, 'SaveNames': SaveNames, 'PromptBuilder': PromptBuilder}
NODE_DISPLAY_NAME_MAPPINGS = {'ImageSources': 'Image Sources', 'EncoderImages': 'Encoder Images', 'SaveNames': 'Save Names', 'PromptBuilder': 'Prompt Builder'}
