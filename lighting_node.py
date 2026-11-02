"""Prompt Lighting: adds a lighting line to the end of a prompt (works with any prompt source, typed or from a file).

Pick a preset, or "custom" and write your own line. "none" passes the prompt through untouched.
Works together with the Prompt Batch Scheduler: wire its category and prompt_id outputs in, and "lighting_by_shot" can pick a
different lighting for each shot type or prompt (one per line: faces=soft studio, 09=hard side light, full_body=custom text...).
Anything not listed uses the lighting chosen in the box.
"""

PRESETS = {
    'soft studio': 'Change the lighting to soft, even studio lighting with gentle shadows. Keep everything else exactly the same.',
    'warm golden hour': 'Change the lighting to warm golden-hour sunlight from low in the sky, with long soft shadows. Keep everything else exactly the same.',
    'cool overcast': 'Change the lighting to cool, soft overcast daylight with no hard shadows. Keep everything else exactly the same.',
    'hard side light': 'Change the lighting to a single hard light from one side, with strong shadows and clear highlights on the lit side. Keep everything else exactly the same.',
    'dramatic low-key': 'Change the lighting to dramatic low-key lighting: deep shadows, one soft key light and dark surroundings. Keep everything else exactly the same.',
    'backlit rim': 'Change the lighting to a backlight that traces a bright rim of light around the subject, with the front softly lit. Keep everything else exactly the same.',
    'candle or firelight': 'Change the lighting to warm flickering candle or firelight, with soft orange light and deep shadows. Keep everything else exactly the same.',
}
CUSTOM_DEFAULT = 'Change the lighting to: describe the lighting here. Keep everything else exactly the same.'


def lighting_line(lighting, custom_lighting=''):
    if lighting == 'custom':
        return (custom_lighting or '').strip()
    return PRESETS.get(lighting, '')


def parse_by_shot(text):
    out = {}
    for raw in (text or '').splitlines():
        if '=' in raw:
            k, v = raw.split('=', 1)
            if k.strip() and v.strip():
                out[k.strip().lower()] = v.strip()
    return out


def line_for_value(value):
    """a lighting_by_shot value is a preset name, 'none', or your own wording"""
    v = value.strip()
    low = v.lower()
    if low == 'none':
        return ''
    if low in PRESETS:
        return PRESETS[low]
    if low.startswith('custom:'):
        return v[len('custom:'):].strip()
    return v


class PromptLighting:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'prompt': ('STRING', {'forceInput': True}),
                'lighting': (['none'] + list(PRESETS) + ['custom'], {'default': 'none',
                                                                    'tooltip': 'The lighting for every prompt that is not listed in lighting_by_shot. none = the prompt passes through untouched.'}),
                'custom_lighting': ('STRING', {'multiline': True, 'default': CUSTOM_DEFAULT}),
                'lighting_by_shot': ('STRING', {'multiline': True, 'default': '',
                                                'tooltip': 'Optional, with the Prompt Batch Scheduler: one per line, a shot type or prompt number = a preset name, none, or your own wording. For example: faces=soft studio   09=hard side light   full_body=warm golden hour'}),
            },
            'optional': {
                'category': ('STRING', {'forceInput': True, 'tooltip': 'Wire the scheduler\'s category output here (the shot type).'}),
                'prompt_id': ('STRING', {'forceInput': True, 'tooltip': 'Wire the scheduler\'s prompt_id output here.'}),
            },
        }

    RETURN_TYPES = ('STRING', 'STRING')
    RETURN_NAMES = ('prompt', 'lighting_line')
    FUNCTION = 'run'
    CATEGORY = 'Claude/batch'

    def run(self, prompt, lighting, custom_lighting, lighting_by_shot='', category=None, prompt_id=None):
        table = parse_by_shot(lighting_by_shot)
        key = next((k for k in ((prompt_id or '').strip().lower(), (category or '').strip().lower()) if k and k in table), None)
        line = line_for_value(table[key]) if key is not None else lighting_line(lighting, custom_lighting)
        text = (prompt or '').rstrip()
        return ((text + ' ' + line).strip() if line else text, line)


NODE_CLASS_MAPPINGS = {'PromptLighting': PromptLighting}
NODE_DISPLAY_NAME_MAPPINGS = {'PromptLighting': 'Prompt Lighting'}
