import { copyFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const siteRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const projectRoot = resolve(siteRoot, "..");
const publicRoot = resolve(siteRoot, "public");
const modelRoot = resolve(projectRoot, "models", "cadencevad-v0.1");
const benchmarkRoot = resolve(projectRoot, "benchmarks", "cadencevad-v0.1");

await mkdir(resolve(publicRoot, "models"), { recursive: true });
await mkdir(resolve(publicRoot, "benchmarks"), { recursive: true });
await Promise.all([
  copyFile(
    resolve(modelRoot, "cadencevad-stream.onnx"),
    resolve(publicRoot, "models", "cadencevad-stream.onnx"),
  ),
  copyFile(
    resolve(modelRoot, "cadencevad-stream.json"),
    resolve(publicRoot, "models", "cadencevad-stream.json"),
  ),
  copyFile(
    resolve(modelRoot, "MODEL_LICENSE.md"),
    resolve(publicRoot, "models", "MODEL_LICENSE.md"),
  ),
  copyFile(resolve(projectRoot, "NOTICE"), resolve(publicRoot, "NOTICE.txt")),
  copyFile(
    resolve(benchmarkRoot, "external-runtime-m4-pro.json"),
    resolve(publicRoot, "benchmarks", "external-runtime-m4-pro.json"),
  ),
  copyFile(
    resolve(benchmarkRoot, "onnx-provider-colab-t4.json"),
    resolve(publicRoot, "benchmarks", "onnx-provider-colab-t4.json"),
  ),
]);
