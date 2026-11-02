"""Prompt Batch Scheduler: works through a whole folder of prompt .txt files, one render per run.

Each run it picks the next prompt that is still under its render target (counting the images already
saved for it), and outputs the prompt text, a random seed and the folder / file name for the save node.
When every prompt has reached its target it blocks everything downstream, so the run ends quietly.

Optional: set images_folder and it also loads the image that belongs to the current prompt (matched by the
prompt id in the file name), so one workflow can use a different input image for each prompt. Or set
image_map to a CSV (prompt, image) that says which image each prompt uses, so several prompts can share one image.

The same node can also (all optional, off by default): build the save folder from a template ({base}, {id}, {category}),
add a studio / custom background line to every prompt, swap image 1 and 2 for one shot type (e.g. faces), route the per-prompt
image to image slot 1, 2 or 3, keep contact sheets (review sheets) up to date for every prompt folder, make the seed repeatable
(same prompt + same render number = same seed), keep a CSV log of every run, and pick the picture size (megapixels) per shot type.

Also in this file: "Contact Sheet Builder", which makes a review sheet (thumbnails with file names) for a prompt folder.
"""
import csv
import hashlib
import json
import math
import os
import random
import re
import time

import folder_paths
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except Exception:  # older ComfyUI
    ExecutionBlocker = None

IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.webp')
HARD_MAX = 100
STATE_NAME = '_batch_scheduler_state.json'
LOG_NAME = '_batch_scheduler_log.csv'
STUDIO_DEFAULT = 'Change the background to a plain, neutral studio backdrop with soft, even studio lighting. Keep everything else exactly the same.'
CUSTOM_DEFAULT = 'Change the background to: describe the background here. Keep everything else exactly the same.'


def _prompt_id(stem):
    # text after the last underscore, so z3ph3er4_18 -> '18' and z3ph3er4_18c -> '18c' (no clash)
    return stem.rsplit('_', 1)[-1] if '_' in stem else stem


def _clean(text, strip_heading):
    lines = text.replace('\r', '').split('\n')
    if strip_heading:
        # heading lines like "zv4ne_09." or "C01"
        lines = [l for l in lines if not re.match(r'^\s*([A-Za-z0-9]+_\d+|[A-Za-z]{1,3}\d{1,3})\.?\s*$', l)]
    return ' '.join(l.strip() for l in lines if l.strip())


def _parse_targets(text):
    out = {}
    for line in (text or '').splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            try:
                out[k.strip().lower()] = int(v.strip())
            except ValueError:
                pass
    return out


def _parse_sizes(text):
    out = {}
    for line in (text or '').splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            try:
                out[k.strip().lower()] = max(0.1, min(8.0, float(v.strip())))
            except ValueError:
                pass
    return out


def scan_prompts(folder):
    """[(category, id, path)] for every .txt under folder (not the state file), sorted."""
    found = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        cat = os.path.relpath(root, folder)
        cat = '' if cat == '.' else cat.replace('\\', '/')
        for f in sorted(files):
            if f.lower().endswith('.txt'):
                found.append((cat, _prompt_id(os.path.splitext(f)[0]), os.path.join(root, f)))
    return found


def find_image(folder, pid, cat):
    """Path of the image in `folder` (searched recursively) that belongs to prompt `pid`, or None.
    The file name must be the id (09.png) or end with it after an underscore (anything_09.png).
    Numeric ids match with or without leading zeros (9 = 09). An image in a sub-folder named like the
    prompt's shot type is preferred."""
    if not folder or not os.path.isdir(folder):
        return None
    want = pid.lower()
    num = (want.lstrip('0') or '0') if want.isdigit() else None
    hits = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        rel = os.path.relpath(root, folder)
        rel = '' if rel == '.' else rel.replace('\\', '/')
        for f in sorted(files):
            if not f.lower().endswith(IMAGE_EXT):
                continue
            sid = _prompt_id(os.path.splitext(f)[0]).lower()
            if sid == want or (num is not None and sid.isdigit() and (sid.lstrip('0') or '0') == num):
                hits.append((0 if rel.lower() == cat.lower() else 1, os.path.join(root, f)))
    return min(hits)[1] if hits else None


def load_image_map(csv_path):
    """{prompt key (lower case): image name} from a CSV with a header row.
    Prompt column: 'prompt' or 'id' (else the first column); keys may be the full prompt file name
    (z3phv4n3_17) or just the id (17). Image column: 'image', 'image3', 'image_path', 'file' or
    'filename' (else the last column)."""
    out = {}
    try:
        with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
            rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    except OSError:
        print(f'[Prompt Batch Scheduler] image_map not found: {csv_path}')
        return out
    if not rows:
        return out
    head = [c.strip().lower() for c in rows[0]]
    pcol = next((head.index(n) for n in ('prompt', 'id') if n in head), 0)
    icol = next((head.index(n) for n in ('image', 'image3', 'image_path', 'file', 'filename') if n in head), len(head) - 1)
    for r in rows[1:]:
        if len(r) > max(pcol, icol) and r[pcol].strip() and r[icol].strip():
            key = os.path.splitext(r[pcol].strip())[0].lower()
            out[key] = r[icol].strip()
    return out


def resolve_image(name, images_folder, csv_dir):
    """Full path for a name from the map: absolute, or under images_folder, or next to the CSV,
    or found by file name anywhere under images_folder. None if it does not exist."""
    cands = [name] if os.path.isabs(name) else []
    if not os.path.isabs(name):
        if images_folder:
            cands.append(os.path.join(images_folder, name))
        cands.append(os.path.join(csv_dir, name))
    for c in cands:
        if os.path.isfile(c):
            return c
    if images_folder and os.path.isdir(images_folder):
        base = os.path.basename(name).lower()
        for root, dirs, files in os.walk(images_folder):
            dirs.sort()
            for f in sorted(files):
                if f.lower() == base:
                    return os.path.join(root, f)
    return None


def load_image_tensor(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None, ...]


def count_images(path):
    try:
        return sum(1 for f in os.listdir(path) if f.lower().endswith(IMAGE_EXT))
    except OSError:
        return 0


def pick_next(items, targets, state, chunk, cap, in_order=False):
    """items: [(cat, id, path)]; returns the item to render next, or None.
    in_order: finish every render of a prompt before the next one (list order).
    otherwise: rounds of `chunk`, smaller targets first."""
    def target(it):
        t = targets.get(it[1].lower(), targets.get(it[0].lower(), cap))
        return max(0, min(t, cap, HARD_MAX))
    if in_order:
        for it in items:
            if state.get(it[1], 0) < target(it):
                return it
        return None
    order = sorted(items, key=lambda it: (target(it), it[0], it[1]))
    top = max((target(it) for it in order), default=0)
    level = chunk
    while True:
        r = min(level, top)
        for it in order:
            if state.get(it[1], 0) < min(r, target(it)):
                return it
        if r >= top:
            return None
        level += chunk


class PromptBatchScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'prompts_folder': ('STRING', {'default': 'F:/LoRAs - Images/Vane/Prompts/comfy_batch'}),
                'output_base': ('STRING', {'default': 'Vane/_Batch Render',
                                           'tooltip': 'Folder under the ComfyUI output folder. Each prompt gets its own sub-folder "_<id>".'}),
                'renders_per_prompt': ('INT', {'default': 100, 'min': 1, 'max': HARD_MAX,
                                               'tooltip': 'How many renders each prompt gets (the whole set). Never more than 100.'}),
                'order': (['in order (all renders of a prompt, then the next)', 'rounds (a few of each prompt, then repeat)'],),
                'chunk': ('INT', {'default': 10, 'min': 1, 'max': HARD_MAX,
                                  'tooltip': 'Only for rounds: every prompt gets this many, then this many more, and so on.'}),
                'filename_template': ('STRING', {'default': 'Slot {id}'}),
                'targets': ('STRING', {'multiline': True, 'default': '',
                                       'tooltip': 'Optional different counts, one per line: "faces=12" (shot type = sub-folder name) or "09=5" (one prompt id). Never above 100.'}),
                'strip_heading': ('BOOLEAN', {'default': True,
                                              'tooltip': 'Drop a heading line like "zv4ne_09." from the prompt text.'}),
                'recount_from_disk': ('BOOLEAN', {'default': False,
                                                  'tooltip': 'On: forget the saved counts and use what is on disk now (deleted frames get re-rendered). Off: deleted frames stay deleted.'}),
            },
            'optional': {
                'images_folder': ('STRING', {'default': '',
                                             'tooltip': 'Optional. A folder of input images named by prompt number (09.png or anything_09.png; sub-folders are searched, one named like the shot type wins). The matching image comes out of "image"; has_image says whether one was found. Leave empty to turn off.'}),
                'image_map': ('STRING', {'default': '',
                                         'tooltip': 'Optional. A CSV file with a header row, one line per prompt: prompt (full name like zv4ne_17, or just 17) and image (a file name, looked up in images_folder or next to the CSV, or a full path). Several prompts may share one image. A prompt listed here uses its mapped image; the others fall back to the images_folder name matching.'}),
                'folder_template': ('STRING', {'default': '{base}/_{id}',
                                               'tooltip': 'Where each prompt saves. {base} = output_base, {id} = prompt number, {category} = shot type. Default {base}/_{id}. The counts follow whatever folder this makes.'}),
                'background': (['none', 'studio', 'custom'], {'default': 'none',
                                                              'tooltip': 'Adds a background line to the end of every prompt from this node. none = prompt untouched.'}),
                'studio_background': ('STRING', {'multiline': True, 'default': STUDIO_DEFAULT}),
                'custom_background': ('STRING', {'multiline': True, 'default': CUSTOM_DEFAULT}),
                'swap_images_for': ('STRING', {'default': '',
                                               'tooltip': 'A shot type (sub-folder name), for example faces. Prompts of that shot type get ref_image as image 1 and main_image as image 2 (swapped_image_1 / _2; both are empty for every other prompt). Empty = never swap.'}),
                'image_slot': (['off', 'image 1', 'image 2', 'image 3'], {'default': 'off',
                                                                           'tooltip': 'Which image slot gets the per-prompt image (from images_folder / image_map). The matching output (per_prompt_1 / 2 / 3) carries it; the others are empty.'}),
                'contact_sheets': ('BOOLEAN', {'default': False,
                                               'tooltip': 'Keeps review sheets (numbered thumbnails with file names) up to date in a "_Contact Sheets" folder next to the prompt folders. They catch up at the start of each run, so they are one render behind.'}),
                'seed_mode': (['random', 'repeatable'], {'default': 'random',
                                                         'tooltip': 'repeatable: the seed comes from seed_base + prompt number + render number, so any render can be redone exactly (it is also in the log).'}),
                'seed_base': ('INT', {'default': 0, 'min': 0, 'max': 2 ** 31 - 1}),
                'log_runs': ('BOOLEAN', {'default': True,
                                         'tooltip': 'Adds a line per run (time, prompt, render number, seed, folder, size) to _batch_scheduler_log.csv in the prompts folder.'}),
                'megapixels_default': ('FLOAT', {'default': 1.3, 'min': 0.1, 'max': 8.0, 'step': 0.1,
                                                 'tooltip': 'Picture size for the megapixels output. Wire it into the Scale node; when the scheduler is on, set the size here.'}),
                'size_by_shot': ('STRING', {'multiline': True, 'default': '',
                                            'tooltip': 'Optional different sizes, one per line: "faces=1.0" (shot type = sub-folder name) or "09=1.7" (one prompt id). Others use megapixels_default.'}),
                'main_image': ('IMAGE', {'tooltip': 'Optional. Wire the main image here to use swap_images_for.'}),
                'ref_image': ('IMAGE', {'tooltip': 'Optional. Wire the second image here to use swap_images_for.'}),
            },
        }

    RETURN_TYPES = ('STRING', 'INT', 'STRING', 'STRING', 'STRING', 'STRING', 'INT', 'INT', 'IMAGE', 'BOOLEAN', 'IMAGE', 'IMAGE', 'IMAGE', 'IMAGE', 'IMAGE', 'FLOAT', 'STRING')
    RETURN_NAMES = ('prompt', 'seed', 'folder_prefix', 'filename_prefix', 'prompt_id', 'category', 'render_number', 'remaining', 'image', 'has_image',
                    'swapped_image_1', 'swapped_image_2', 'per_prompt_1', 'per_prompt_2', 'per_prompt_3', 'megapixels', 'status')
    FUNCTION = 'run'
    CATEGORY = 'Claude/batch'

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')  # always run

    def run(self, prompts_folder, output_base, renders_per_prompt, order, chunk, filename_template, targets, strip_heading, recount_from_disk, images_folder='', image_map='',
            folder_template='{base}/_{id}', background='none', studio_background=STUDIO_DEFAULT, custom_background=CUSTOM_DEFAULT,
            swap_images_for='', image_slot='off', contact_sheets=False, main_image=None, ref_image=None,
            seed_mode='random', seed_base=0, log_runs=True, megapixels_default=1.3, size_by_shot=''):
        items = scan_prompts(prompts_folder)
        if not items:
            raise ValueError(f'No .txt prompts found in {prompts_folder}')
        out_dir = folder_paths.get_output_directory()

        def folder_for(pid, cat):
            pref = ((folder_template or '{base}/_{id}').replace('{base}', output_base.strip('\\/')).replace('{id}', pid)
                    .replace('{category}', cat).replace('\\', '/'))
            return pref, (pref if os.path.isabs(pref) else os.path.join(out_dir, *re.split(r'[\\/]+', pref.strip('/'))))
        state_path = os.path.join(prompts_folder, STATE_NAME)
        key = output_base
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                all_state = json.load(f)
        except (OSError, ValueError):
            all_state = {}
        state = {} if recount_from_disk else dict(all_state.get(key, {}))
        for cat_, pid, _ in items:  # first time a prompt is seen: start from what is on disk
            if pid not in state:
                state[pid] = count_images(folder_for(pid, cat_)[1])
        inflight = state.pop('__inflight__', None)
        if isinstance(inflight, dict) and inflight.get('pid') in state and state[inflight['pid']] > 0:
            if count_images(inflight.get('folder', '')) <= inflight.get('disk', 0):
                state[inflight['pid']] -= 1
                print(f"[Prompt Batch Scheduler] the last run for prompt {inflight['pid']} saved no picture (interrupted?): giving that render back")
        if contact_sheets:
            self._sheets(items, folder_for)
        tmap = _parse_targets(targets)
        it = pick_next(items, tmap, state, chunk, renders_per_prompt, order.startswith('in order'))
        if it is None:
            all_state[key] = state
            self._save(state_path, all_state)
            if contact_sheets:
                self._sheets(items, folder_for)
            print('[Prompt Batch Scheduler] every prompt has reached its target.')
            if ExecutionBlocker is None:
                raise RuntimeError('Prompt Batch Scheduler: all prompts are done.')
            b = ExecutionBlocker(None)
            return (b,) * len(self.RETURN_TYPES)
        cat, pid, path = it
        with open(path, 'r', encoding='utf-8-sig') as f:
            text = _clean(f.read(), strip_heading)
        bg = studio_background if background == 'studio' else custom_background if background == 'custom' else ''
        if bg and bg.strip():
            text = (text + ' ' + bg.strip()).strip()
        state[pid] += 1
        full_folder = folder_for(pid, cat)[1]
        state['__inflight__'] = {'pid': pid, 'folder': full_folder, 'disk': count_images(full_folder)}
        all_state[key] = state
        self._save(state_path, all_state)
        t = lambda i: max(0, min(tmap.get(i[1].lower(), tmap.get(i[0].lower(), renders_per_prompt)), renders_per_prompt, HARD_MAX))
        remaining = sum(max(0, t(i) - state.get(i[1], 0)) for i in items)
        folder_prefix = folder_for(pid, cat)[0]
        folder = (images_folder or '').strip()
        map_path = (image_map or '').strip()
        img_path, mapped = None, False
        if map_path:
            stem = os.path.splitext(os.path.basename(path))[0].lower()
            cmap = load_image_map(map_path)
            name = cmap.get(stem) or cmap.get(pid.lower())
            if name:
                mapped = True
                img_path = resolve_image(name, folder, os.path.dirname(os.path.abspath(map_path)))
                if not img_path:
                    print(f'[Prompt Batch Scheduler] image_map lists "{name}" for prompt {pid}, but that file was not found')
        if not mapped:
            img_path = find_image(folder, pid, cat)
        if img_path:
            image, has_image = load_image_tensor(img_path), True
        else:
            image, has_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32), False  # placeholder, check has_image
            if (folder or map_path) and not mapped:
                print(f'[Prompt Batch Scheduler] no image for prompt {pid}; has_image = False')
        swap = bool((swap_images_for or '').strip()) and cat.lower() == swap_images_for.strip().lower()
        out1, out2 = (ref_image, main_image) if swap else (None, None)   # empty unless this prompt is swapped, so the encoder falls back to its normal images
        slot = lambda k: image if (image_slot == f'image {k}' and has_image) else None
        if seed_mode == 'repeatable':
            seed = int(hashlib.sha256(f'{seed_base}|{pid}|{state[pid]}'.encode('utf-8')).hexdigest()[:12], 16) % (10 ** 15)
        else:
            seed = random.randint(0, 10 ** 15)
        status = f"Prompt {pid} ({cat or 'no shot type'}) - render {state[pid]} of {t(it)}, {remaining} renders left in the batch"
        sizes = _parse_sizes(size_by_shot)
        mp = float(sizes.get(pid.lower(), sizes.get(cat.lower(), megapixels_default)))
        if log_runs:
            self._log(prompts_folder, [time.strftime('%Y-%m-%d %H:%M:%S'), pid, cat, state[pid], seed, folder_prefix, mp])
        return (text, seed, folder_prefix, filename_template.replace('{id}', pid),
                pid, cat, state[pid], remaining, image, has_image, out1, out2, slot(1), slot(2), slot(3), mp, status)

    @staticmethod
    def _log(prompts_folder, row):
        path = os.path.join(prompts_folder, LOG_NAME)
        try:
            new = not os.path.exists(path)
            with open(path, 'a', encoding='utf-8', newline='') as f:
                w = csv.writer(f)
                if new:
                    w.writerow(['time', 'prompt_id', 'category', 'render_number', 'seed', 'folder', 'megapixels'])
                w.writerow(row)
        except OSError as e:
            print(f'[Prompt Batch Scheduler] could not write the run log: {e}')

    @staticmethod
    def _sheets(items, folder_for):
        seen = set()
        for cat_, pid_, _ in items:
            full = folder_for(pid_, cat_)[1]
            if full in seen or not os.path.isdir(full):
                continue
            seen.add(full)
            try:
                build_contact_sheet(full, os.path.join(os.path.dirname(full), SHEETS_DIR))
            except Exception as e:
                print(f'[Prompt Batch Scheduler] contact sheet for {full} failed: {e}')

    @staticmethod
    def _save(path, data):
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------
# Contact Sheet Builder
# ----------------------------------------------------------------------------------------------
SHEETS_DIR = '_Contact Sheets'


def _cs_font(size):
    for fp in (r'C:\Windows\Fonts\segoeui.ttf', r'C:\Windows\Fonts\arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def _cs_images(folder):
    try:
        names = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXT) and not f.lower().startswith('_contact')]
    except OSError:
        return []
    return sorted(names, key=lambda f: [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', f)])


def _cs_thumb(path, tile_w, tile_h, cache_dir):
    st = os.stat(path)
    key = hashlib.sha1(f'{path}|{st.st_mtime_ns}|{st.st_size}|{tile_w}x{tile_h}'.encode('utf-8')).hexdigest()
    cp = os.path.join(cache_dir, key + '.jpg')
    if os.path.exists(cp):
        return Image.open(cp).convert('RGB')
    im = Image.open(path)
    im.draft('RGB', (tile_w * 2, tile_h * 2))
    im = ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'), (tile_w, tile_h), Image.LANCZOS)
    os.makedirs(cache_dir, exist_ok=True)
    im.save(cp, quality=90)
    return im


def build_contact_sheet(folder, out_dir, cols=4, rows=4, tile=360, force=False):
    """One contact sheet (several pages if needed) for the images in `folder`; returns a status string or None."""
    names = _cs_images(folder)
    if not names:
        return None
    label = os.path.basename(folder.rstrip('\\/'))
    per_page = cols * rows
    pages = math.ceil(len(names) / per_page)
    outs = [os.path.join(out_dir, f'{label}.jpg' if pages == 1 else f'{label} (page {k + 1} of {pages}).jpg') for k in range(pages)]
    newest = max(os.path.getmtime(os.path.join(folder, n)) for n in names)
    if not force and all(os.path.exists(o) and os.path.getmtime(o) >= newest for o in outs):
        return f'{label}: unchanged'
    tw, th = tile, int(tile * 4 / 3)
    pad, head, foot = 12, 54, 26
    cw, ch = tw + pad, th + foot + pad
    f_head, f_small, f_big = _cs_font(24), _cs_font(15), _cs_font(20)
    cache = os.path.join(out_dir, '.thumbs')
    os.makedirs(out_dir, exist_ok=True)
    for pg in range(pages):
        chunk = names[pg * per_page:(pg + 1) * per_page]
        nrows = math.ceil(len(chunk) / cols)
        sheet = Image.new('RGB', (cols * cw + pad, head + nrows * ch + pad), (24, 24, 28))
        d = ImageDraw.Draw(sheet)
        d.text((pad, 12), f'{label}   -   {len(names)} renders' + (f'   -   page {pg + 1} of {pages}' if pages > 1 else ''), font=f_head, fill=(235, 235, 238))
        for k, n in enumerate(chunk):
            cx, cy = pad + (k % cols) * cw, head + (k // cols) * ch
            try:
                im = _cs_thumb(os.path.join(folder, n), tw, th, cache)
            except Exception:
                im = Image.new('RGB', (tw, th), (60, 20, 20))
                ImageDraw.Draw(im).text((10, 10), 'cannot read', font=f_small, fill=(235, 235, 238))
            sheet.paste(im, (cx + (tw - im.width) // 2, cy + (th - im.height) // 2))
            d.rectangle((cx, cy, cx + 34, cy + 24), fill=(0, 0, 0))
            d.text((cx + 5, cy + 1), str(pg * per_page + k + 1), font=f_big, fill=(235, 235, 238))
            stem = os.path.splitext(n)[0]
            while d.textlength(stem, font=f_small) > tw and len(stem) > 8:
                stem = stem[:-2]
            d.text((cx, cy + th + 4), stem, font=f_small, fill=(150, 150, 158))
        sheet.save(outs[pg], quality=88)
    return f'{label}: {len(names)} renders -> {pages} sheet(s)'


class ContactSheetBuilder:
    """Makes a review sheet (thumbnails with file names) for a prompt folder after each render, or for a whole batch folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'folder': ('STRING', {'default': '', 'forceInput': False,
                                      'tooltip': 'A prompt folder (wire the scheduler\'s folder_prefix here to refresh just that prompt after each render), or a whole batch folder to rebuild every prompt. Relative to the ComfyUI output folder, or a full path.'}),
                'columns': ('INT', {'default': 4, 'min': 1, 'max': 10}),
                'rows': ('INT', {'default': 4, 'min': 1, 'max': 10, 'tooltip': 'Renders per sheet = columns x rows; more renders make more pages.'}),
                'tile_width': ('INT', {'default': 360, 'min': 120, 'max': 800}),
            },
            'optional': {
                'after': ('STRING', {'forceInput': True, 'tooltip': 'Wire the save node\'s output here so the sheet is built after the image is saved.'}),
            },
        }

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('status',)
    FUNCTION = 'run'
    OUTPUT_NODE = True
    CATEGORY = 'Claude/batch'

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def run(self, folder, columns, rows, tile_width, after=None):
        target = (folder or '').strip()
        if not target:
            return ('no folder set',)
        full = target if os.path.isabs(target) else os.path.join(folder_paths.get_output_directory(), *re.split(r'[\\/]+', target.strip('\\/')))
        if not os.path.isdir(full):
            print(f'[Contact Sheet Builder] folder not found: {full}')
            return (f'folder not found: {full}',)
        msgs = []
        if _cs_images(full):          # a single prompt folder: sheets go next to it
            targets, out_dir = [full], os.path.join(os.path.dirname(full), SHEETS_DIR)
        else:                          # a batch folder: one sheet per prompt folder
            targets = [os.path.join(full, d) for d in sorted(os.listdir(full)) if os.path.isdir(os.path.join(full, d)) and d != SHEETS_DIR and not d.startswith('.')]
            out_dir = os.path.join(full, SHEETS_DIR)
        for tf in targets:
            try:
                r = build_contact_sheet(tf, out_dir, columns, rows, tile_width)
            except Exception as e:
                r = f'{os.path.basename(tf)}: failed ({e})'
            if r:
                msgs.append(r)
        status = '; '.join(m for m in msgs if 'unchanged' not in m) or 'sheets up to date'
        print(f'[Contact Sheet Builder] {status}')
        return (status,)


NODE_CLASS_MAPPINGS = {'PromptBatchScheduler': PromptBatchScheduler, 'ContactSheetBuilder': ContactSheetBuilder}
NODE_DISPLAY_NAME_MAPPINGS = {'PromptBatchScheduler': 'Prompt Batch Scheduler (whole list)', 'ContactSheetBuilder': 'Contact Sheet Builder'}
