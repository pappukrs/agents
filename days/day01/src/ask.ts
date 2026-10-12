import { ask } from "./llm.js";

// Usage: npm run ask -- "your prompt" [--system "..."] [--temp 0.2] [--model gemini-2.5-pro]
const argv = process.argv.slice(2);
const flag = (name: string) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : undefined;
};
const prompt = argv.filter((a, i) => !a.startsWith("--") && !argv[i - 1]?.startsWith("--")).join(" ");
if (!prompt) {
  console.error('Usage: npm run ask -- "prompt" [--system "..."] [--temp 0.7] [--model name]');
  process.exit(1);
}

const result = await ask(prompt, {
  system: flag("system"),
  temperature: flag("temp") ? Number(flag("temp")) : undefined,
  model: flag("model"),
});

console.log(result.text.trim());
console.log("\n---");
console.log(`model=${result.model} latency=${result.latencyMs}ms tokens in/out/total=${result.inputTokens}/${result.outputTokens}/${result.totalTokens}`);
