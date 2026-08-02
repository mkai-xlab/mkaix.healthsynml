"use strict";

const elements = {
  inputView: document.getElementById("inputView"),
  resultView: document.getElementById("resultView"),
  responseInput: document.getElementById("responseInput"),
  validationMessage: document.getElementById("validationMessage"),
  renderButton: document.getElementById("renderButton"),
  pasteButton: document.getElementById("pasteButton"),
  exampleButton: document.getElementById("exampleButton"),
  editButton: document.getElementById("editButton"),
  clearButton: document.getElementById("clearButton"),
  headerStatus: document.getElementById("headerStatus"),
  filename: document.getElementById("filename"),
  summaryStats: document.getElementById("summaryStats"),
  annotatedSection: document.getElementById("annotatedSection"),
  annotatedImage: document.getElementById("annotatedImage"),
  annotatedError: document.getElementById("annotatedError"),
  annotatedOpen: document.getElementById("annotatedOpen"),
  annotatedDownload: document.getElementById("annotatedDownload"),
  predictionCount: document.getElementById("predictionCount"),
  predictionList: document.getElementById("predictionList"),
  additionalSection: document.getElementById("additionalSection"),
  additionalFields: document.getElementById("additionalFields"),
  imageDialog: document.getElementById("imageDialog"),
  dialogTitle: document.getElementById("dialogTitle"),
  dialogImage: document.getElementById("dialogImage"),
  dialogClose: document.getElementById("dialogClose"),
};

const gradeColors = ["#2f775e", "#789344", "#b77a12", "#bd5b35", "#963c3c"];
const knownTopLevelFields = new Set(["filename", "predictions", "annotated_image"]);
const knownPredictionFields = new Set([
  "predicted_class",
  "predicted_grade",
  "confidence",
  "description",
  "details",
  "box",
  "yolo_confidence",
  "knee_side",
  "roi_image",
  "gradcam_image",
]);

function parseResponse(raw) {
  const text = raw.trim().replace(/^\uFEFF/, "");
  if (!text) {
    throw new Error("Paste a prediction response first.");
  }

  let value;
  try {
    value = JSON.parse(text);
  } catch (error) {
    const position = /position (\d+)/i.exec(error.message);
    let location = "";
    if (position) {
      const offset = Number(position[1]);
      const before = text.slice(0, offset);
      const line = before.split("\n").length;
      const column = offset - before.lastIndexOf("\n");
      location = ` at line ${line}, column ${column}`;
    }
    throw new Error(`Invalid JSON${location}.`);
  }

  for (let depth = 0; depth < 3 && typeof value === "string"; depth += 1) {
    value = JSON.parse(value);
  }
  if (value && typeof value.body === "string") {
    value = JSON.parse(value.body);
  }
  if (value && value.data && typeof value.data === "object" && !value.predictions) {
    value = value.data;
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("The response must be a JSON object.");
  }
  if (value.detail && !value.predictions) {
    throw new Error(`API error: ${formatValue(value.detail)}`);
  }
  if (!Array.isArray(value.predictions)) {
    throw new Error('Missing required array field "predictions".');
  }
  return value;
}

function detectMime(base64Value) {
  const prefix = base64Value.slice(0, 16);
  if (prefix.startsWith("iVBORw0KGgo")) return "image/png";
  if (prefix.startsWith("/9j/")) return "image/jpeg";
  if (prefix.startsWith("R0lGOD")) return "image/gif";
  if (prefix.startsWith("UklGR")) return "image/webp";
  if (prefix.startsWith("PHN2Z") || prefix.startsWith("PD94bW")) return "image/svg+xml";
  return "image/jpeg";
}

function imageSource(value) {
  if (typeof value !== "string" || !value.trim()) return null;
  const trimmed = value.trim();
  if (/^(data:image\/|blob:|https?:\/\/|\/)/i.test(trimmed)) return trimmed;
  const compact = trimmed.replace(/\s+/g, "");
  if (!/^[A-Za-z0-9+/]+={0,2}$/.test(compact)) return null;
  return `data:${detectMime(compact)};base64,${compact}`;
}

function normalizedProbability(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.max(0, Math.min(1, number > 1 ? number / 100 : number));
}

function percentage(value, digits = 1) {
  return `${(normalizedProbability(value) * 100).toFixed(digits)}%`;
}

function gradeNumber(prediction) {
  const direct = Number(prediction.predicted_class);
  if (Number.isInteger(direct)) return Math.max(0, Math.min(4, direct));
  const parsed = Number.parseInt(String(prediction.predicted_grade || ""), 10);
  return Number.isInteger(parsed) ? Math.max(0, Math.min(4, parsed)) : 0;
}

function gradeName(prediction, grade) {
  const raw = String(prediction.predicted_grade || `Grade ${grade}`);
  return raw.replace(/^\d+\s*/, "") || `Grade ${grade}`;
}

function confidenceStyle(value) {
  const confidence = normalizedProbability(value);
  if (confidence < 0.4) {
    return { label: "Low confidence", ink: "#7f2d27", background: "#fbeae7" };
  }
  if (confidence < 0.7) {
    return { label: "Moderate confidence", ink: "#7b4d00", background: "#fff2d9" };
  }
  return { label: "High confidence", ink: "#10543e", background: "#e5f2ed" };
}

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function formatValue(value) {
  if (value === null) return "null";
  if (value === undefined) return "Not provided";
  if (typeof value === "string") {
    if (value.length > 160 && imageSource(value)) return "[image rendered separately]";
    return value;
  }
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(6);
  if (typeof value === "boolean") return value ? "true" : "false";
  return JSON.stringify(value);
}

function configureImage(image, errorElement, value) {
  const source = imageSource(value);
  image.hidden = !source;
  errorElement.hidden = Boolean(source);
  image.removeAttribute("src");
  if (!source) return null;
  image.onload = () => {
    image.hidden = false;
    errorElement.hidden = true;
  };
  image.onerror = () => {
    image.hidden = true;
    errorElement.hidden = false;
  };
  image.src = source;
  return source;
}

function openImage(source, title) {
  if (!source) return;
  elements.dialogTitle.textContent = title;
  elements.dialogImage.src = source;
  elements.dialogImage.alt = title;
  elements.imageDialog.showModal();
}

function createSummaryStat(value, label) {
  const item = createElement("div", "summary-stat");
  item.append(createElement("span", "stat-value", value));
  item.append(createElement("span", "stat-label", label));
  return item;
}

function sortedProbabilities(details) {
  if (!details || typeof details !== "object" || Array.isArray(details)) return [];
  return Object.entries(details).sort(([left], [right]) => {
    const a = Number.parseInt(left, 10);
    const b = Number.parseInt(right, 10);
    if (Number.isFinite(a) && Number.isFinite(b)) return a - b;
    return left.localeCompare(right);
  });
}

function createProbabilityList(prediction) {
  const wrapper = createElement("div");
  const probabilities = sortedProbabilities(prediction.details);
  const heading = createElement("div", "probability-heading");
  heading.append(createElement("strong", null, "Grade probabilities"));
  const total = probabilities.reduce((sum, [, value]) => sum + normalizedProbability(value), 0);
  const totalLabel = probabilities.length ? `Total ${total.toFixed(3)}` : "Not provided";
  heading.append(createElement("span", "probability-total", totalLabel));
  wrapper.append(heading);

  const list = createElement("div", "probability-list");
  if (!probabilities.length) {
    list.append(createElement("div", "empty-state", "No probability details"));
  }
  probabilities.forEach(([label, value], index) => {
    const probability = normalizedProbability(value);
    const parsedGrade = Number.parseInt(label, 10);
    const colorIndex = Number.isInteger(parsedGrade) ? parsedGrade : index;
    const row = createElement("div", "probability-row");
    row.append(createElement("span", "probability-label", label));
    const track = createElement("div", "probability-track");
    const fill = createElement("div", "probability-fill");
    fill.style.setProperty("--probability", `${probability * 100}%`);
    fill.style.setProperty("--bar-color", gradeColors[colorIndex % gradeColors.length]);
    track.append(fill);
    row.append(track);
    row.append(createElement("span", "probability-value", percentage(value, 2)));
    list.append(row);
  });
  wrapper.append(list);
  return wrapper;
}

function createMetricPair(prediction) {
  const pair = createElement("div", "metric-pair");
  const yolo = createElement("div", "metric-cell");
  yolo.append(createElement("strong", null, percentage(prediction.yolo_confidence, 1)));
  yolo.append(createElement("span", null, "YOLO confidence"));
  const box = createElement("div", "metric-cell");
  box.append(createElement("strong", null, Array.isArray(prediction.box) ? prediction.box.join(", ") : "None"));
  box.append(createElement("span", null, "Box x1, y1, x2, y2"));
  pair.append(yolo, box);
  return pair;
}

function createPredictionImage(value, label, filename) {
  const button = createElement("button", "image-button prediction-image");
  button.type = "button";
  button.setAttribute("aria-label", `Expand ${label}`);
  const imageLabel = createElement("span", "image-label", label);
  const image = document.createElement("img");
  image.alt = label;
  const error = createElement("span", "image-error", "Image unavailable");
  const source = configureImage(image, error, value);
  button.disabled = !source;
  button.addEventListener("click", () => openImage(source, `${filename} - ${label}`));
  button.append(imageLabel, image, error);
  return button;
}

function appendExtraFields(container, object, excluded) {
  Object.entries(object).forEach(([key, value]) => {
    if (excluded.has(key)) return;
    container.append(createElement("dt", null, key));
    container.append(createElement("dd", null, formatValue(value)));
  });
}

function createPredictionCard(prediction, index, filename) {
  const grade = gradeNumber(prediction);
  const confidence = confidenceStyle(prediction.confidence);
  const article = createElement("article", "prediction-card");

  const header = createElement("header", "prediction-header");
  const gradeBlock = createElement("div", "grade-block");
  gradeBlock.style.setProperty("--grade-color", gradeColors[grade]);
  gradeBlock.append(createElement("strong", null, String(grade)));
  gradeBlock.append(createElement("span", null, "Grade"));

  const title = createElement("div", "prediction-title");
  title.append(createElement("h3", null, gradeName(prediction, grade)));
  title.append(createElement("p", null, `Knee ${index + 1} | Class index ${formatValue(prediction.predicted_class)}`));

  const tags = createElement("div", "prediction-tags");
  tags.append(createElement("span", "side-label", String(prediction.knee_side || "unknown")));
  const confidenceTag = createElement(
    "span",
    "confidence-label",
    `${percentage(prediction.confidence, 1)} | ${confidence.label}`,
  );
  confidenceTag.style.setProperty("--confidence-ink", confidence.ink);
  confidenceTag.style.setProperty("--confidence-bg", confidence.background);
  tags.append(confidenceTag);
  header.append(gradeBlock, title, tags);

  const body = createElement("div", "prediction-body");
  const metrics = createElement("div", "metrics-column");
  metrics.append(createElement("p", "description", String(prediction.description || "No description provided.")));
  metrics.append(createMetricPair(prediction));
  metrics.append(createProbabilityList(prediction));

  const extra = createElement("dl", "metadata-list");
  appendExtraFields(extra, prediction, knownPredictionFields);
  if (extra.children.length) {
    const extraHeading = createElement("div", "probability-heading");
    extraHeading.append(createElement("strong", null, "Additional prediction fields"));
    metrics.append(extraHeading, extra);
  }

  const images = createElement("div", "image-column");
  images.append(
    createPredictionImage(prediction.roi_image, "ROI image", filename),
    createPredictionImage(prediction.gradcam_image, "Class activation heatmap", filename),
  );
  body.append(metrics, images);
  article.append(header, body);
  return article;
}

function renderResponse(response) {
  const filename = String(response.filename || "Unnamed image");
  const predictions = response.predictions;
  const confidences = predictions.map((item) => normalizedProbability(item.confidence));
  const meanConfidence = confidences.length
    ? confidences.reduce((sum, value) => sum + value, 0) / confidences.length
    : 0;
  const sides = new Set(predictions.map((item) => String(item.knee_side || "unknown").toLowerCase()));

  elements.filename.textContent = filename;
  elements.headerStatus.textContent = filename;
  elements.summaryStats.replaceChildren(
    createSummaryStat(String(predictions.length), predictions.length === 1 ? "Knee" : "Knees"),
    createSummaryStat(percentage(meanConfidence, 1), "Mean confidence"),
    createSummaryStat(String(sides.size), sides.size === 1 ? "Side label" : "Side labels"),
  );

  const annotatedSource = configureImage(
    elements.annotatedImage,
    elements.annotatedError,
    response.annotated_image,
  );
  elements.annotatedSection.hidden = !annotatedSource;
  elements.annotatedOpen.onclick = () => openImage(annotatedSource, `${filename} - Annotated radiograph`);
  if (annotatedSource) {
    elements.annotatedDownload.href = annotatedSource;
    elements.annotatedDownload.download = `${filename.replace(/\.[^.]+$/, "")}-annotated.jpg`;
  }

  elements.predictionCount.textContent = `${predictions.length} detected`;
  elements.predictionList.replaceChildren();
  if (!predictions.length) {
    elements.predictionList.append(createElement("div", "empty-state", "No knee predictions"));
  } else {
    predictions.forEach((prediction, index) => {
      if (!prediction || typeof prediction !== "object") {
        const error = createElement("div", "empty-state", `Prediction ${index + 1} is not an object`);
        elements.predictionList.append(error);
        return;
      }
      elements.predictionList.append(createPredictionCard(prediction, index, filename));
    });
  }

  elements.additionalFields.replaceChildren();
  appendExtraFields(elements.additionalFields, response, knownTopLevelFields);
  elements.additionalSection.hidden = elements.additionalFields.children.length === 0;

  elements.inputView.hidden = true;
  elements.resultView.hidden = false;
  elements.editButton.hidden = false;
  elements.validationMessage.hidden = true;
  window.scrollTo({ top: 0, behavior: "instant" });
}

function showInput() {
  elements.resultView.hidden = true;
  elements.inputView.hidden = false;
  elements.editButton.hidden = true;
  requestAnimationFrame(() => elements.responseInput.focus());
}

function showError(message) {
  elements.validationMessage.textContent = message;
  elements.validationMessage.hidden = false;
  elements.responseInput.setAttribute("aria-invalid", "true");
}

function clearAll() {
  elements.responseInput.value = "";
  elements.responseInput.removeAttribute("aria-invalid");
  elements.validationMessage.hidden = true;
  elements.headerStatus.textContent = "No response loaded";
  elements.predictionList.replaceChildren();
  elements.additionalFields.replaceChildren();
  showInput();
}

function renderFromInput() {
  try {
    const response = parseResponse(elements.responseInput.value);
    elements.responseInput.removeAttribute("aria-invalid");
    renderResponse(response);
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
  }
}

function sampleImage(kind, label) {
  const canvas = document.createElement("canvas");
  canvas.width = kind === "annotated" ? 900 : 520;
  canvas.height = kind === "annotated" ? 520 : 520;
  const context = canvas.getContext("2d");
  context.fillStyle = "#202725";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#c9cecb";
  context.fillRect(canvas.width * 0.14, canvas.height * 0.12, canvas.width * 0.30, canvas.height * 0.31);
  context.fillRect(canvas.width * 0.56, canvas.height * 0.12, canvas.width * 0.30, canvas.height * 0.31);
  context.fillStyle = "#aeb5b1";
  context.fillRect(canvas.width * 0.13, canvas.height * 0.56, canvas.width * 0.31, canvas.height * 0.30);
  context.fillRect(canvas.width * 0.56, canvas.height * 0.56, canvas.width * 0.31, canvas.height * 0.30);
  context.fillStyle = "#111714";
  context.fillRect(canvas.width * 0.10, canvas.height * 0.46, canvas.width * 0.36, canvas.height * 0.055);
  context.fillRect(canvas.width * 0.54, canvas.height * 0.46, canvas.width * 0.36, canvas.height * 0.055);
  if (kind === "heatmap") {
    context.fillStyle = "rgba(238, 189, 28, 0.60)";
    context.beginPath();
    context.ellipse(canvas.width * 0.66, canvas.height * 0.49, 95, 54, 0, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = "rgba(207, 53, 43, 0.78)";
    context.beginPath();
    context.ellipse(canvas.width * 0.69, canvas.height * 0.49, 43, 31, 0, 0, Math.PI * 2);
    context.fill();
  }
  if (kind === "annotated") {
    context.strokeStyle = "#58d6a6";
    context.lineWidth = 5;
    context.strokeRect(canvas.width * 0.08, canvas.height * 0.06, canvas.width * 0.39, canvas.height * 0.84);
    context.strokeRect(canvas.width * 0.53, canvas.height * 0.06, canvas.width * 0.39, canvas.height * 0.84);
  }
  context.fillStyle = "#f5f8f6";
  context.font = "700 18px sans-serif";
  context.fillText(label, 18, 30);
  return canvas.toDataURL("image/png");
}

function loadExample() {
  const roi = sampleImage("roi", "Example knee ROI");
  const heatmap = sampleImage("heatmap", "Example native CAM");
  const example = {
    filename: "example-bilateral-knee.png",
    predictions: [
      {
        predicted_class: 2,
        predicted_grade: "2Mild",
        confidence: 0.6842,
        description: "Grade 2: Definite osteophytes and possible joint space narrowing.",
        details: {
          "0Normal": 0.0712,
          "1Doubtful": 0.1558,
          "2Mild": 0.6842,
          "3Moderate": 0.0801,
          "4Severe": 0.0087,
        },
        box: [48, 61, 438, 486],
        yolo_confidence: 0.947,
        knee_side: "right",
        roi_image: roi.split(",")[1],
        gradcam_image: heatmap,
      },
      {
        predicted_class: 3,
        predicted_grade: "3Moderate",
        confidence: 0.7315,
        description: "Grade 3: Multiple osteophytes, definite joint space narrowing, and some sclerosis.",
        details: {
          "0Normal": 0.0121,
          "1Doubtful": 0.0438,
          "2Mild": 0.1887,
          "3Moderate": 0.7315,
          "4Severe": 0.0239,
        },
        box: [462, 58, 853, 488],
        yolo_confidence: 0.932,
        knee_side: "left",
        roi_image: roi,
        gradcam_image: heatmap.split(",")[1],
      },
    ],
    annotated_image: sampleImage("annotated", "Example annotated radiograph"),
  };
  elements.responseInput.value = JSON.stringify(example, null, 2);
  renderFromInput();
}

elements.renderButton.addEventListener("click", renderFromInput);
elements.exampleButton.addEventListener("click", loadExample);
elements.editButton.addEventListener("click", showInput);
elements.clearButton.addEventListener("click", clearAll);
elements.pasteButton.addEventListener("click", async () => {
  try {
    elements.responseInput.value = await navigator.clipboard.readText();
    elements.validationMessage.hidden = true;
    elements.responseInput.focus();
  } catch (error) {
    showError("Clipboard access was blocked. Paste into the JSON field directly.");
  }
});
elements.responseInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    renderFromInput();
  }
});
elements.dialogClose.addEventListener("click", () => elements.imageDialog.close());
elements.imageDialog.addEventListener("click", (event) => {
  if (event.target === elements.imageDialog) elements.imageDialog.close();
});

elements.responseInput.focus();
