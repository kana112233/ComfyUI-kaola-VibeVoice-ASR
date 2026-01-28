import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "VibeVoice.ShowString",
    async setup() {
        console.log("%c VibeVoice ShowString Extension Loaded", "color: green; font-weight: bold;");
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "VibeVoiceShowText") {

            // 1. Ensure widget exists when node is created
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Check if widget already exists (from serialization)
                if (!this.widgets || !this.widgets.find(w => w.name === "text")) {
                    // Create a multiline string widget
                    const w = ComfyWidgets["STRING"](this, "text", ["STRING", { multiline: true }], app).widget;
                    w.inputEl.readOnly = true;
                    w.inputEl.style.opacity = 0.6;
                }
                return r;
            };

            // 2. Update widget when execution completes
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (message && message.text) {
                    const text = message.text.join("");
                    const w = this.widgets?.find((w) => w.name === "text");
                    if (w) {
                        w.value = text;
                        this.onResize?.(this.size);
                    }
                }
            };
        }

        if (nodeData.name === "VibeVoiceSaveFile") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (message && message.file_info && message.file_info.length > 0) {
                    const file_info = message.file_info[0];
                    const filename = file_info.filename;
                    const subfolder = file_info.subfolder || "";
                    const type = file_info.type || "output";

                    const url = `/view?filename=${encodeURIComponent(filename)}&type=${type}&subfolder=${encodeURIComponent(subfolder)}`;

                    // Add or update button
                    const buttonName = "Open File";
                    let widget = this.widgets?.find((w) => w.name === buttonName || w.name.startsWith("Open "));

                    if (!widget) {
                        widget = this.addWidget("button", buttonName, null, () => { });
                    }

                    widget.name = "Open " + filename;
                    widget.callback = () => {
                        window.open(url, '_blank');
                    };

                    this.setDirtyCanvas(true, true);
                }
            };
        }
    },
});
