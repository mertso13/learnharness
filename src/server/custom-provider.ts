/**
 * The custom OpenAI-compatible endpoint: turns the user's non-secret config
 * (base URL + discovered models) into a pi-ai provider registered in
 * `piModels()`, so its models fill the same slots as any built-in. Keyless
 * local servers work via a placeholder key resolved at request time.
 */

import { createProvider, type ApiKeyAuth, type Api, type Model, type Provider } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { piModels } from "../agent/models.ts";
import { CUSTOM_PROVIDER } from "../core/auth-keys.ts";
import {
	CUSTOM_MODEL_DEFAULT_CONTEXT_WINDOW,
	CUSTOM_MODEL_DEFAULT_MAX_TOKENS,
	customProviderUsable,
	normalizeBaseUrl,
	type CustomModel,
	type CustomModelOverride,
	type CustomProviderSettings,
} from "../core/settings.ts";
import { keyValue } from "./keys.ts";

/** Placeholder bearer for keyless endpoints (the OpenAI client needs *a* key). */
const KEYLESS_PLACEHOLDER = "not-needed";

/** Narrow an unknown to a plain object (not null, not an array). */
const isRecord = (value: unknown): value is Record<string, unknown> =>
	typeof value === "object" && value !== null && !Array.isArray(value);

/** Auth: the reserved key via the keys seam, else the keyless placeholder. */
const customEndpointAuth: ApiKeyAuth = {
	name: `${CUSTOM_PROVIDER.label} API key`,
	resolve: async () => {
		const key = keyValue(CUSTOM_PROVIDER.envVar)?.trim();
		return { auth: { apiKey: key || KEYLESS_PLACEHOLDER }, source: key ? CUSTOM_PROVIDER.envVar : "keyless" };
	},
};

/** Map a discovered model to a pi-ai entry: image→input, reasoning→thinking
 *  levels, context/maxTokens→limits. Resolution ladder (owner, 2026-10-27):
 *  the user's pin wins, then the endpoint's own metadata, then the compiled
 *  default — a local server that says nothing (or says 8k) can be corrected.
 *  baseUrl is stamped per-model. */
function toPiModel(
	model: CustomModel,
	baseUrl: string,
	overrides: Record<string, CustomModelOverride>,
): Model<Api> {
	const pin = overrides[model.id];
	return {
		id: model.id,
		name: model.id,
		api: "openai-completions",
		provider: CUSTOM_PROVIDER.id,
		baseUrl,
		// Pin wins; else detected from the endpoint's metadata (see
		// detectReasoning). true makes pi-ai offer the standard levels and send
		// reasoning_effort on the wire; no thinkingLevelMap → the default
		// off/minimal/low/medium/high set.
		reasoning: pin?.reasoning ?? Boolean(model.reasoning),
		input: model.image ? ["text", "image"] : ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: pin?.context ?? model.context ?? CUSTOM_MODEL_DEFAULT_CONTEXT_WINDOW,
		maxTokens: pin?.maxTokens ?? model.maxTokens ?? CUSTOM_MODEL_DEFAULT_MAX_TOKENS,
		type: "chat",
	};
}

/** Build an OpenAI-compatible provider from parts. baseUrl must be normalized. */
export function buildOpenAiCompatibleProvider(input: {
	baseUrl: string;
	models: readonly CustomModel[];
	/** User pins by model id (see CustomModelOverride) — absent = none. */
	overrides?: Record<string, CustomModelOverride>;
}): Provider<Api> {
	const overrides = input.overrides ?? {};
	return createProvider<Api>({
		id: CUSTOM_PROVIDER.id,
		name: CUSTOM_PROVIDER.label,
		baseUrl: input.baseUrl,
		auth: { apiKey: customEndpointAuth },
		models: input.models.map((m) => toPiModel(m, input.baseUrl, overrides)),
		api: openAICompletionsApi(),
	});
}

/** Re-sync the provider into piModels() (idempotent): usable → rebuild;
 *  not usable → remove. Never throws. */
export function syncCustomProvider(cp: CustomProviderSettings | undefined): void {
	const models = piModels();
	if (!customProviderUsable(cp)) {
		models.deleteProvider(CUSTOM_PROVIDER.id);
		return;
	}
	models.setProvider(
		buildOpenAiCompatibleProvider({
			baseUrl: normalizeBaseUrl(cp.baseUrl),
			models: cp.models ?? [],
			overrides: cp.overrides ?? {},
		}),
	);
}

/** Outcome of a model-discovery probe. `status` is the endpoint's HTTP status
 *  when it answered at all, so callers can tell a rejected key from a missing
 *  model list. */
export type DiscoverResult =
	| { ok: true; models: CustomModel[] }
	| { ok: false; error: string; status?: number };

/** Probe GET {baseUrl}/models for model ids. Scheme validated (http/https
 *  only); keyless endpoints answer without a bearer. Non-2xx or unparseable
 *  body → honest error, never a silent empty list. */
export async function discoverModels(baseUrl: string, apiKey?: string): Promise<DiscoverResult> {
	const base = normalizeBaseUrl(baseUrl);
	if (!base) return { ok: false, error: "Base URL must be a valid http(s) URL" };
	const key = apiKey?.trim() || keyValue(CUSTOM_PROVIDER.envVar)?.trim();
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), 15_000);
	try {
		const res = await fetch(`${base}/models`, {
			headers: key ? { authorization: `Bearer ${key}` } : {},
			signal: controller.signal,
		});
		if (!res.ok) return { ok: false, error: `Endpoint returned ${res.status} ${res.statusText}`, status: res.status };
		const body = (await res.json().catch(() => null)) as unknown;
		const list = extractModels(body);
		if (list === null) return { ok: false, error: "Endpoint response was not a model list" };
		return { ok: true, models: list };
	} catch (err) {
		const msg = err instanceof Error ? (err.name === "AbortError" ? "Endpoint timed out" : err.message) : "Request failed";
		return { ok: false, error: msg };
	} finally {
		clearTimeout(timer);
	}
}

/** The connect's model list: what the endpoint advertises, plus every typed id
 *  it does not advertise (at default limits, so the pins can correct them).
 *  Discovered metadata always wins over a typed id of the same name. */
export function mergeManualModels(discovered: readonly CustomModel[], manualIds: readonly string[]): CustomModel[] {
	const byId = new Map(discovered.map((m) => [m.id, m]));
	for (const id of manualIds) if (!byId.has(id)) byId.set(id, { id });
	return [...byId.values()].sort((a, b) => a.id.localeCompare(b.id));
}

/** Model rows from a list response ({data:[…]} | {models:[…]} | bare array),
 *  or null if unrecognized. */
function modelRows(body: unknown): unknown[] | null {
	if (Array.isArray(body)) return body;
	if (isRecord(body)) {
		if (Array.isArray(body.data)) return body.data;
		if (Array.isArray(body.models)) return body.models;
	}
	return null;
}

/** Does a /models row advertise image input? Scans the common modality/
 *  capability fields; absent or unrecognized → false (text-only). */
function detectImage(row: Record<string, unknown>): boolean {
	const architecture = isRecord(row.architecture) ? row.architecture : undefined;
	const capabilities = isRecord(row.capabilities) ? row.capabilities : undefined;
	if (capabilities?.vision === true || capabilities?.image === true || capabilities?.image_input === true) return true;
	const candidates = [
		architecture?.input_modalities,
		row.input_modalities,
		row.modalities,
		row.supported_modalities,
		row.capabilities,
	];
	for (const c of candidates) {
		if (Array.isArray(c) && c.some((v) => typeof v === "string" && /image|vision/i.test(v))) return true;
	}
	return false;
}

/** Does a /models row advertise reasoning? Metadata only — no name guessing,
 *  because a false positive sends reasoning_effort an endpoint will reject.
 *  Trusted signals: an explicit boolean (reasoning / supports_reasoning /
 *  thinking) or a reasoning token in the capability/parameter lists servers
 *  publish (vLLM + OpenRouter's supported_parameters, capabilities, tags).
 *  Absent or unrecognized → false (thinking stays "off"). */
function detectReasoning(row: Record<string, unknown>): boolean {
	for (const key of ["reasoning", "supports_reasoning", "thinking"] as const) {
		if (row[key] === true) return true;
	}
	const architecture = isRecord(row.architecture) ? row.architecture : undefined;
	const capabilities = isRecord(row.capabilities) ? row.capabilities : undefined;
	if (capabilities?.reasoning === true || capabilities?.thinking === true) return true;
	const candidates = [row.supported_parameters, row.capabilities, row.tags, architecture?.output_modalities];
	return candidates.some(
		(list) =>
			Array.isArray(list) &&
			list.some((v) => typeof v === "string" && (/reasoning/i.test(v) || v.toLowerCase() === "thinking")),
	);
}

/** First positive integer among `keys` on a row (endpoints name limits
 *  inconsistently). */
function firstPositive(row: Record<string, unknown>, keys: readonly string[]): number | undefined {
	for (const key of keys) {
		const value = row[key];
		if (typeof value === "number" && Number.isInteger(value) && value > 0) return value;
	}
	return undefined;
}

// Field names seen in the wild for each limit (first present wins).
const CONTEXT_KEYS = ["max_input_tokens", "context_length", "context_window", "max_context", "n_ctx", "max_seq_len", "max_position_embeddings"] as const;
const MAX_TOKEN_KEYS = ["max_output_tokens"] as const;

/** Pull models from a list response (objects with `id` or bare strings), each
 *  with detected image + token limits. Null if the shape is unrecognized. */
function extractModels(body: unknown): CustomModel[] | null {
	const rows = modelRows(body);
	if (rows === null) return null;
	const byId = new Map<string, CustomModel>();
	for (const row of rows) {
		if (typeof row === "string") {
			if (row.trim()) byId.set(row, { id: row });
			continue;
		}
		if (isRecord(row) && typeof row.id === "string" && row.id.trim()) {
			const model: CustomModel = { id: row.id };
			if (detectImage(row)) model.image = true;
			if (detectReasoning(row)) model.reasoning = true;
			const context = firstPositive(row, CONTEXT_KEYS);
			if (context) model.context = context;
			const maxTokens = firstPositive(row, MAX_TOKEN_KEYS);
			if (maxTokens) model.maxTokens = maxTokens;
			byId.set(row.id, model);
		}
	}
	return [...byId.values()].sort((a, b) => a.id.localeCompare(b.id));
}
