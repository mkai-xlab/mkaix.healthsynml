import assert from "node:assert/strict";
import { createRequire } from "node:module";
import fs from "node:fs";
import vm from "node:vm";


const require = createRequire(import.meta.url);
const { parseHTML } = require("linkedom");


const html = fs.readFileSync("/viewer/index.html", "utf8");
const javascript = fs.readFileSync("/viewer/app.js", "utf8");
const { document, window } = parseHTML(html);

window.scrollTo = () => {};
globalThis.document = document;
globalThis.window = window;
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: { clipboard: { readText: async () => "" } },
});
globalThis.requestAnimationFrame = (callback) => callback();

vm.runInThisContext(javascript, { filename: "app.js" });

const input = document.getElementById("responseInput");
const renderButton = document.getElementById("renderButton");
const validation = document.getElementById("validationMessage");

input.value = "not-json";
renderButton.click();
assert.equal(validation.hidden, false);
assert.match(validation.textContent, /Invalid JSON/);

const onePixelGif = "R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==";
input.value = JSON.stringify({
  filename: "smoke-knee.png",
  predictions: [
    {
      predicted_class: 2,
      predicted_grade: "2Mild",
      confidence: 0.62,
      description: "Grade 2",
      details: {
        "0Normal": 0.05,
        "1Doubtful": 0.15,
        "2Mild": 0.62,
        "3Moderate": 0.16,
        "4Severe": 0.02,
      },
      box: [10, 20, 300, 380],
      yolo_confidence: 0.94,
      knee_side: "right",
      roi_image: onePixelGif,
      gradcam_image: `data:image/gif;base64,${onePixelGif}`,
    },
  ],
  annotated_image: onePixelGif,
});
renderButton.click();

assert.equal(document.getElementById("inputView").hidden, true);
assert.equal(document.getElementById("resultView").hidden, false);
assert.equal(document.getElementById("filename").textContent, "smoke-knee.png");
assert.equal(document.querySelectorAll(".prediction-card").length, 1);
assert.equal(document.querySelectorAll(".probability-row").length, 5);
assert.equal(document.querySelector(".grade-block strong").textContent, "2");
assert.match(document.getElementById("annotatedImage").src, /^data:image\/gif;base64,/);
assert.match(document.querySelector(".prediction-image img").src, /^data:image\/gif;base64,/);
assert.doesNotMatch(document.getElementById("resultView").textContent, /R0lGOD/);

console.log("KL response viewer DOM smoke test passed");
