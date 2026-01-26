# ComfyUI VibeVoice ASR

A ComfyUI custom node for [Microsoft VibeVoice ASR](https://github.com/microsoft/VibeVoice).
This node allows you to perform high-quality, long-form speech recognition, speaker diarization, and timestamping directly within ComfyUI.

## features

*   **Long-form ASR**: Supports processing up to 60 minutes of audio in a single pass.
*   **SRT Export**: Automatically generates standard SRT subtitles with timestamps.
*   **Speaker Diarization**: Identifies different speakers (Speaker 0, Speaker 1, etc.).
*   **Context/Hotwords**: Improve accuracy for specific terms or names by providing context.
*   **Rich Output**: Provides both SRT strings and raw JSON data for advanced processing.

## Installation

1.  **Clone the repository** into your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes/
    git clone https://github.com/kana112233/ComfyUI-kaola-VibeVoice-ASR.git
    cd ComfyUI-kaola-VibeVoice-ASR
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    > **Note**: This node requires `transformers>=4.51.3`. Please ensure your environment is updated.

3.  **Restart ComfyUI**.

## Usage

You can find the nodes under the `VibeVoice` category.

### Nodes

1.  **VibeVoice Model Loader**
    *   Loads the model.
    *   **model_name**: 
        *   **Auto Download**: Keep default `microsoft/VibeVoice-ASR`. It will download automatically to your HuggingFace cache folder on first run.
        *   **Manual**: You can download the model from [HuggingFace](https://huggingface.co/microsoft/VibeVoice-ASR), place it in a local folder (e.g., `ComfyUI/models/vibevoice`), and paste the **absolute path** to that folder here.
    *   **precision**: `bf16` (recommended for modern GPUs), `fp16`, or `fp32`.
    *   **device**: `auto` (recommended), `cuda`, `mps` (Mac), `cpu`.

## Model Download regarding
By default, the node uses `microsoft/VibeVoice-ASR` from HuggingFace.
If you have internet access, it will download automatically (~7B model) to your `~/.cache/huggingface` directory.

### Manual Download (Offline)
1. Download the model files from: [microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR)
2. Place them in a folder, for example: `ComfyUI/models/vibevoice/VibeVoice-ASR`
3. In the `VibeVoiceLoader` node, replace `microsoft/VibeVoice-ASR` with the **absolute path** to your folder (e.g., `/content/ComfyUI/models/vibevoice/VibeVoice-ASR` or `D:\ComfyUI\models\vibevoice\VibeVoice-ASR`).

2.  **VibeVoice Transcribe**
    *   Performs the transcription.
    *   **audio**: Connect a standard AUDIO output (e.g., from `LoadAudio`).
    *   **context_info**: (Optional) Enter names, technical terms, or context to help the model recognize them correctly.
    *   **Outputs**:
        *   `srt_content`: Subtitle text.
        *   `json_content`: Detailed JSON data.

### Example Workflow
An example workflow is included in `examples/vibevoice_workflow_example.json`. Drag and drop it into ComfyUI to get started.

## Updating

This node contains the original VibeVoice repository as a submodule in `VibeVoice_src`. To update the core VibeVoice library:

```bash
cd custom_nodes/ComfyUI-kaola-VibeVoice-ASR/VibeVoice_src
git pull
```

## Credits

Based on [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice).
