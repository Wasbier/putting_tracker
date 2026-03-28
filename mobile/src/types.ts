export type PuttOutcome = "made" | "miss";

/** Snapshot from putting_ws_server (Tapo stream on PC). */
export interface LiveState {
  attempts: number;
  made: number;
  missed: number;
  putt_sequence: string[];
}

export interface Session {
  id: string;
  startedAt: string;
  endedAt: string | null;
  putts: PuttOutcome[];
}

export interface SessionSummary {
  attempts: number;
  made: number;
  missed: number;
}

export function summarizeSession(s: Session): SessionSummary {
  const attempts = s.putts.length;
  const made = s.putts.filter((p) => p === "made").length;
  return { attempts, made, missed: attempts - made };
}
