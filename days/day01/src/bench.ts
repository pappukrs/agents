import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { ask } from "./llm.js";

/**
 * Block 3: run every system prompt in prompts/ against every input in cases.json,
 * record latency + tokens, and write a markdown table to results/.
 * Quality is scored by you (1-5) after reading results — fill the column in by hand.
 */
interface Case { id: string; input: string }
const cases: Case[] = JSON.parse(readFileSync("cases.json", "utf8"));
const promptFiles = readdirSync("prompts").filter((f) => f.endsWith(".txt"));
const temperature = Number(process.env.TEMP ?? 0.3);

const rows: string[] = ["| prompt | case | latency ms | in | out | quality (1-5) | output (first 120 chars) |", "|---|---|---|---|---|---|---|"];

for (const file of promptFiles) {
  const system = readFileSync(`prompts/${file}`, "utf8");
  for (const c of cases) {
    const r = await ask(c.input, { system, temperature });
    const snippet = r.text.replace(/\s+/g, " ").slice(0, 120);
    rows.push(`| ${file} | ${c.id} | ${r.latencyMs} | ${r.inputTokens} | ${r.outputTokens} |  | ${snippet} |`);
    console.log(`${file} / ${c.id}: ${r.latencyMs}ms, ${r.totalTokens} tokens`);
  }
}

const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const out = `results/bench-${stamp}.md`;
writeFileSync(out, `# Bench ${stamp}\n\ntemperature=${temperature}\n\n${rows.join("\n")}\n`);
console.log(`\nWrote ${out}`);
