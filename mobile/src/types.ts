export type PuttOutcome = "made" | "miss";

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
