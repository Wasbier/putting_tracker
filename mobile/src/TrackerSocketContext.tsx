import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  DEFAULT_WS_URL,
  loadTrackerWsUrl,
  saveTrackerWsUrl,
} from "./trackerSettings";
import type { LiveState } from "./types";

type TrackerSocketContextValue = {
  urlInput: string;
  setUrlInput: (v: string) => void;
  settingsLoaded: boolean;
  connecting: boolean;
  connected: boolean;
  live: LiveState | null;
  statusLine: string;
  connect: () => void;
  disconnect: () => void;
  sendReset: () => void;
};

const TrackerSocketContext = createContext<TrackerSocketContextValue | null>(
  null,
);

export function TrackerSocketProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [urlInput, setUrlInput] = useState(DEFAULT_WS_URL);
  const [settingsLoaded, setSettingsLoaded] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [connected, setConnected] = useState(false);
  const [live, setLive] = useState<LiveState | null>(null);
  const [statusLine, setStatusLine] = useState("");
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      const u = await loadTrackerWsUrl();
      if (alive) {
        setUrlInput(u);
        setSettingsLoaded(true);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const disconnect = useCallback(() => {
    const w = wsRef.current;
    wsRef.current = null;
    if (w) {
      w.onclose = null;
      w.onerror = null;
      w.onmessage = null;
      w.onopen = null;
      try {
        w.close();
      } catch {
        /* ignore */
      }
    }
    setConnected(false);
    setConnecting(false);
    setLive(null);
  }, []);

  useEffect(() => () => disconnect(), [disconnect]);

  const connect = useCallback(() => {
    disconnect();
    const url = urlInput.trim();
    if (!url.startsWith("ws://") && !url.startsWith("wss://")) {
      setStatusLine("URL must start with ws:// or wss://");
      return;
    }
    setStatusLine("");
    setConnecting(true);
    saveTrackerWsUrl(url).catch(() => {});
    const ws = new WebSocket(url);
    wsRef.current = ws;
    ws.onopen = () => {
      setConnecting(false);
      setConnected(true);
      setStatusLine("Connected — live counts from PC (Tapo stream)");
    };
    ws.onmessage = (ev) => {
      try {
        const d = JSON.parse(String(ev.data)) as Record<string, unknown>;
        if (d.type !== "state") return;
        setLive({
          attempts: Number(d.attempts) || 0,
          made: Number(d.made) || 0,
          missed: Number(d.missed) || 0,
          putt_sequence: Array.isArray(d.putt_sequence)
            ? (d.putt_sequence as string[]).map(String)
            : [],
        });
      } catch {
        /* ignore */
      }
    };
    ws.onerror = () => {
      setStatusLine(
        "Connection error — check URL, Wi‑Fi, and putting_ws_server on PC",
      );
    };
    ws.onclose = () => {
      setConnecting(false);
      setConnected(false);
      setLive(null);
      if (wsRef.current === ws) wsRef.current = null;
    };
  }, [disconnect, urlInput]);

  const sendReset = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "reset_session" }));
  }, []);

  const value = useMemo(
    () => ({
      urlInput,
      setUrlInput,
      settingsLoaded,
      connecting,
      connected,
      live,
      statusLine,
      connect,
      disconnect,
      sendReset,
    }),
    [
      urlInput,
      settingsLoaded,
      connecting,
      connected,
      live,
      statusLine,
      connect,
      disconnect,
      sendReset,
    ],
  );

  return (
    <TrackerSocketContext.Provider value={value}>
      {children}
    </TrackerSocketContext.Provider>
  );
}

export function useTrackerSocket(): TrackerSocketContextValue {
  const ctx = useContext(TrackerSocketContext);
  if (!ctx) {
    throw new Error("useTrackerSocket must be used within TrackerSocketProvider");
  }
  return ctx;
}
