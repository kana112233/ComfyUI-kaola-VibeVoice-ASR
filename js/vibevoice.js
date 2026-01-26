import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "VibeVoice.ShowString",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "VibeVoiceShowText") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                // message.text is the list of strings returned by python {"ui": {"text": [...]}}
                if (message && message.text) {
                    const text = message.text.join("");

                    // Try to find existing widget to update
                    let w = this.widgets?.find((w) => w.name === "text_output");

                    if (!w) {
                        // Create new widget if not exists
                        // Note: ComfyWidgets["STRING"] creates a wrapper, accessing .widget gets the actual widget object
                        // Parameters: node, inputName, [type, options], app
                        const widgetWrapper = ComfyWidgets["STRING"](this, "text_output", ["STRING", { multiline: true }], app);
                        w = widgetWrapper.widget;
                        w.inputEl.readOnly = true;
                        w.inputEl.style.opacity = 0.8;
                    }

                    if (w) {
                        w.value = text;
                        this.onResize?.(this.size); // Trigger resize to fit text
                    }
                }
            };
        }
    },
});
