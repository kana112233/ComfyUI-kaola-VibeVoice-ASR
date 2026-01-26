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
    git clone https://github.com/your-username/ComfyUI-kaola-VibeVoice-ASR.git
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
    *   **model_name**: Default is `microsoft/VibeVoice-ASR`. It will auto-download from Hugging Face on first use.
    *   **precision**: `bf16` (recommended for modern GPUs), `fp16`, or `fp32`.
    *   **device**: `auto` (recommended), `cuda`, `mps` (Mac), `cpu`.

2.  **VibeVoice Transcribe**
    *   Performs the transcription.
    *   **audio**: Connect a standard AUDIO output (e.g., from `LoadAudio`).
    *   **context_info**: (Optional) Enter names, technical terms, or context to help the model recognize them correctly.
    *   **Outputs**:
        *   `srt_content`: Subtitle text.
        *   `json_content`: Detailed JSON data.

### Example Workflow
An example workflow is included: `vibevoice_workflow_example.json`. Drag and drop it into ComfyUI to get started.

## Updating

This node contains the original VibeVoice repository as a submodule in `VibeVoice_src`. To update the core VibeVoice library:

```bash
cd custom_nodes/ComfyUI-kaola-VibeVoice-ASR/VibeVoice_src
git pull
```

## Credits

Based on [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice).
