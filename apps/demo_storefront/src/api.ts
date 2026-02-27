export interface ClickPayload {
  year: number;
  month: number;
  day: number;
  country: number;
  page_1_main_category: number;
  page_2_clothing_model: string;
  colour: number;
  location: number;
  model_photography: number;
  price: number;
  price_2: number;
  page: number;
}

export interface Prediction {
  label: "low-intent" | "high-intent";
  probability: number;
}

export interface SessionState {
  session_id: number;
  click_count: number;
  status: "collecting" | "predicted";
  prediction: Prediction | null;
}

export interface ClickResponse {
  session_id: number;
  click_count: number;
  triggered: boolean;
  prediction: Prediction | null;
  show_ad: boolean;
  raw_row: Record<string, string | number>;
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function createSession(): Promise<SessionState> {
  const response = await fetch("/api/v1/sessions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  });
  return parseResponse<SessionState>(response);
}

export async function postClick(
  sessionId: number,
  payload: ClickPayload,
): Promise<ClickResponse> {
  const response = await fetch(`/api/v1/sessions/${sessionId}/clicks`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return parseResponse<ClickResponse>(response);
}

export interface DemoApi {
  createSession: typeof createSession;
  postClick: typeof postClick;
}

export const defaultDemoApi: DemoApi = {
  createSession,
  postClick,
};
