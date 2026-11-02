# ComfyUI LoRA Dataset Batch

Batch-render a LoRA training set from ComfyUI: work through a whole folder of prompt files, one render per run, with per-prompt images, lighting, background, shot-type swaps, contact sheets and sensible saving, all inside one node. Plus four small routing nodes and a ready-made image-edit workflow ("Modular v6") built around them.

**Install:** copy this folder into `ComfyUI/custom_nodes/` (any folder name works, e.g. `ComfyUI-LoRA-Dataset-Batch`) and restart ComfyUI. No extra Python packages.
**Workflow:** `workflows/Qwen 2.1 Edit - Modular v6.json` (drag it onto ComfyUI). It also uses other node packs (rgthree, Pixaroma, the Qwen 2.1 edit nodes); ComfyUI Manager will offer to install anything missing. Set the paths first: `prompts_folder`, `output_base`, the image loaders.

All nodes are in the `Claude/batch` category.

## Prompt Batch Scheduler (whole list)

Reads every `.txt` under `prompts_folder` (sub-folders allowed; the sub-folder name is the shot type; the number after the last underscore in a file name is the prompt id). Each run picks the next render that is still due. Set the ComfyUI **Run count** to prompts x renders; when everything is rendered further runs do nothing. Counts are remembered in `_batch_scheduler_state.json` inside the prompts folder, an interrupted render is refunded on the next run, and every pick is logged to `_batch_scheduler_log.csv` (`log_runs`).

| Setting | What it does |
| --- | --- |
| `prompts_folder`, `output_base` | Where the prompts are, and the folder inside the ComfyUI output folder (or an absolute path) that gets `<output_base>/_<id>`. |
| `renders_per_prompt`, `order`, `chunk`, `targets` | How many renders per prompt (1-100), in order or in rounds, and optional per-shot-type (`faces=12`) or per-prompt (`09=5`) counts. |
| `strip_heading`, `recount_from_disk` | Drop a heading line such as `name_09.`; or recount the images on disk instead of using the saved counts. |
| `filename_template`, `folder_template` | File name (`Slot {id}`) and folder (`{base}/_{id}`, `{category}` also works) patterns. |
| `images_folder`, `image_map`, `image_slot` | Optional per-prompt image: files named by prompt number (`09.png`, `anything_09.png`), or a CSV (`prompt`, `image`). `image_slot` sends it to image 1, 2 or 3. |
| `background`, `studio_background`, `custom_background` | Optionally append a "change the background" line to every prompt. |
| `swap_images_for` | A shot type whose prompts swap image 1 and 2 (needs the `main_image` / `ref_image` sockets). |
| `seed_mode`, `seed_base` | Random seeds, or reproducible ones from a base seed. |
| `megapixels_default`, `size_by_shot` | Output size, with per-shot-type overrides (`faces=1.0`). |
| `contact_sheets` | Builds numbered thumbnail sheets per prompt folder (`_Contact Sheets`); they lag one render behind, so queue one extra run at the end. |

Outputs: `prompt`, `seed`, `folder_prefix`, `filename_prefix`, `prompt_id`, `category`, `render_number`, `remaining`, `image`, `has_image`, `swapped_image_1/2`, `per_prompt_1/2/3`, `megapixels`, `status` (a readable line such as "Prompt 12 (faces) - render 3 of 5, 40 renders left in the batch").

## The other nodes

- **Prompt Lighting** adds a lighting line to a prompt (presets, custom text, or per shot type).
- **Image Sources** picks the main image and image 2: per-prompt image first, then the batch folder / fixed image.
- **Encoder Images** gives the text encoder its three images, with the shot-type swap built in.
- **Save Names** uses the scheduler's folder and file name when it is on, otherwise typed boxes.
- **Prompt Builder** chooses the prompt (scheduler > prompt file by index > typed) and adds the lighting line.
- **Contact Sheet Builder** the contact-sheet maker on its own.

Everything is optional and off by default, so older workflows keep working.
