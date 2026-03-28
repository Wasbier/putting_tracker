import AsyncStorage from "@react-native-async-storage/async-storage";
import type { PuttOutcome, Session } from "./types";

const SESSIONS_KEY = "@putting_tracker/sessions_v1";
const DRAFT_KEY = "@putting_tracker/draft_session_v1";

function parseSessions(raw: string | null): Session[] {
  if (!raw) return [];
  try {
    const data = JSON.parse(raw) as unknown;
    if (!Array.isArray(data)) return [];
    return data.filter(isSession);
  } catch {
    return [];
  }
}

function isSession(x: unknown): x is Session {
  if (x === null || typeof x !== "object") return false;
  const o = x as Record<string, unknown>;
  return (
    typeof o.id === "string" &&
    typeof o.startedAt === "string" &&
    (o.endedAt === null || typeof o.endedAt === "string") &&
    Array.isArray(o.putts) &&
    o.putts.every((p) => p === "made" || p === "miss")
  );
}

export async function loadCompletedSessions(): Promise<Session[]> {
  const raw = await AsyncStorage.getItem(SESSIONS_KEY);
  return parseSessions(raw).filter((s) => s.endedAt !== null);
}

export async function saveCompletedSessions(sessions: Session[]): Promise<void> {
  const completed = sessions.filter((s) => s.endedAt !== null);
  await AsyncStorage.setItem(SESSIONS_KEY, JSON.stringify(completed));
}

export async function loadDraftSession(): Promise<Session | null> {
  const raw = await AsyncStorage.getItem(DRAFT_KEY);
  if (!raw) return null;
  try {
    const data = JSON.parse(raw) as unknown;
    return isSession(data) && data.endedAt === null ? data : null;
  } catch {
    return null;
  }
}

export async function saveDraftSession(session: Session | null): Promise<void> {
  if (session === null) {
    await AsyncStorage.removeItem(DRAFT_KEY);
    return;
  }
  if (session.endedAt !== null) {
    await AsyncStorage.removeItem(DRAFT_KEY);
    return;
  }
  await AsyncStorage.setItem(DRAFT_KEY, JSON.stringify(session));
}

export async function appendCompletedSession(session: Session): Promise<void> {
  if (session.endedAt === null) return;
  const all = await loadCompletedSessions();
  all.unshift(session);
  await saveCompletedSessions(all);
}

export function newSession(): Session {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
    startedAt: new Date().toISOString(),
    endedAt: null,
    putts: [],
  };
}

export function withPutt(session: Session, outcome: PuttOutcome): Session {
  return { ...session, putts: [...session.putts, outcome] };
}

export function undoLastPutt(session: Session): Session {
  if (session.putts.length === 0) return session;
  return { ...session, putts: session.putts.slice(0, -1) };
}

export function endSession(session: Session): Session {
  return { ...session, endedAt: new Date().toISOString() };
}
