/* Minimal valid PNG (1×1). Replace with real art before store release. */
const fs = require("fs");
const path = require("path");

const b64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

const assets = path.join(__dirname, "..", "assets");
fs.mkdirSync(assets, { recursive: true });
const buf = Buffer.from(b64, "base64");
fs.writeFileSync(path.join(assets, "icon.png"), buf);
fs.writeFileSync(path.join(assets, "splash.png"), buf);
