export type StepName = "understand" | "retrieve" | "decide" | "draft" | "verify" | "refuse";

export interface AcceptedEvent { request_id: string; queue_position: number; }
export interface StepStartEvent { step: StepName; lane: number; }
export interface StepEndEvent { step: StepName; lane: number; model_ms: number | null; tokens_in: number | null; tokens_out: number | null; }
export interface LimiterWaitEvent { ms: number; reason: "rpm" | "tpm" | "concurrency" | "daily"; }
export interface RetryEvent { attempt: number; backoff_ms: number; reason: string; }
export interface DegradeEvent { rung: number; reason: string; }
export interface FirstTokenEvent { }
export interface TokenEvent { text: string; }
export interface FinalEvent {
  status: "ok" | "refused" | "error";
  answer: string;
  citations: { paper: string; page: number; chunk_id: string }[];
  refusal_reason: string | null;
  grade_hint: string;
  total_ms: number | null;
}

export type SSEEvent =
  | { type: "accepted"; data: AcceptedEvent }
  | { type: "step_start"; data: StepStartEvent }
  | { type: "step_end"; data: StepEndEvent }
  | { type: "limiter_wait"; data: LimiterWaitEvent }
  | { type: "retry"; data: RetryEvent }
  | { type: "degrade"; data: DegradeEvent }
  | { type: "first_token"; data: FirstTokenEvent }
  | { type: "token"; data: TokenEvent }
  | { type: "final"; data: FinalEvent };

export interface LimitsResponse {
  rpm: { used: number; limit: number; remaining: number };
  tpm: { used: number; limit: number; remaining: number };
  rpd: { used: number; limit: number; remaining: number };
  in_flight: number;
  queue_length: number;
  resets_at: string | null;
}

export interface SummaryResponse {
  scenarios: Record<string, Record<string, number>>;
  settings: Record<string, Record<string, number>>;
}

export interface TraceResponse {
  question: string;
  total_ms: number;
  refused: boolean;
  refusal_reason: string | null;
  final_answer: string | null;
  retrieved_chunks: string[];
  retries: number;
  decisions: { decision: string; [k: string]: unknown }[];
  steps: {
    name: string;
    duration_ms: number;
    queue_wait_ms: number;
    model_time_ms: number;
    retry_time_ms: number;
    first_token_ms: number | null;
    prompt_tokens: number | null;
    completion_tokens: number | null;
    status: string;
    extra: Record<string, unknown>;
  }[];
}
