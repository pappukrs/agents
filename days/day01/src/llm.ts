import "dotenv/config";
import { GoogleGenAI } from "@google/genai";

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) throw new Error("GEMINI_API_KEY missing. Copy .env.example to .env and add your key.");

export const ai = new GoogleGenAI({ apiKey });
export const DEFAULT_MODEL = process.env.GEMINI_MODEL ?? "gemini-2.5-flash";

export interface AskOptions {
  system?: string;
  model?: string;
  temperature?: number;
  maxOutputTokens?: number;
}

export interface AskResult {
  text: string;
  latencyMs: number;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  model: string;
}

/** One LLM call with configurable system prompt, model, and temperature. */
export async function ask(prompt: string, opts: AskOptions = {}): Promise<AskResult> {
  const model = opts.model ?? DEFAULT_MODEL;
  const started = performance.now();

  const res = await ai.models.generateContent({
    model,
    contents: prompt,
    config: {
      systemInstruction: opts.system,
      temperature: opts.temperature ?? 0.7,
      maxOutputTokens: opts.maxOutputTokens ?? 1024,
    },
  });

  const usage = res.usageMetadata;
  return {
    text: res.text ?? "",
    latencyMs: Math.round(performance.now() - started),
    inputTokens: usage?.promptTokenCount ?? 0,
    outputTokens: usage?.candidatesTokenCount ?? 0,
    totalTokens: usage?.totalTokenCount ?? 0,
    model,
  };
}
