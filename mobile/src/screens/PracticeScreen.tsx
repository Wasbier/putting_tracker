import React, { useCallback, useEffect, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useTrackerSocket } from "../TrackerSocketContext";
import { appendCompletedSession } from "../storage";
import { DEFAULT_WS_URL } from "../trackerSettings";
import type { LiveState, PuttOutcome, Session } from "../types";

const green = "#0d4f2b";
const cream = "#f4f1e8";
const missRed = "#c44c4c";
const madeGold = "#d4a84b";

function buildSessionFromLive(live: LiveState, startedAt: string): Session {
  const putts = [...live.putt_sequence] as PuttOutcome[];
  while (putts.length < live.attempts) {
    putts.push("miss");
  }
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
    startedAt,
    endedAt: new Date().toISOString(),
    putts,
  };
}

export default function PracticeScreen() {
  const {
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
  } = useTrackerSocket();

  const [sessionActive, setSessionActive] = useState(false);
  const [sessionStartedAt, setSessionStartedAt] = useState<string | null>(null);
  const [practiceMessage, setPracticeMessage] = useState("");

  useEffect(() => {
    if (connected) setPracticeMessage("");
  }, [connected]);

  const startSession = useCallback(() => {
    if (!connected) {
      setPracticeMessage("Connect to the tracker server first.");
      return;
    }
    const now = new Date().toISOString();
    setSessionStartedAt(now);
    setSessionActive(true);
    sendReset();
    setPracticeMessage("Session started — counts reset on tracker");
  }, [connected, sendReset]);

  const endSession = useCallback(async () => {
    if (!sessionActive || sessionStartedAt === null || !live) {
      setPracticeMessage("No active session to save.");
      return;
    }
    const session = buildSessionFromLive(live, sessionStartedAt);
    await appendCompletedSession(session);
    setSessionActive(false);
    setSessionStartedAt(null);
    sendReset();
    setPracticeMessage("Session saved to History.");
  }, [sessionActive, sessionStartedAt, live, sendReset]);

  if (!settingsLoaded) {
    return (
      <SafeAreaView style={styles.center} edges={["top", "bottom"]}>
        <ActivityIndicator color={cream} size="large" />
      </SafeAreaView>
    );
  }

  const display = live ?? {
    attempts: 0,
    made: 0,
    missed: 0,
    putt_sequence: [] as string[],
  };
  const seqPreview =
    display.putt_sequence.length > 0
      ? display.putt_sequence.join(" · ")
      : "—";

  return (
    <SafeAreaView style={styles.container} edges={["top", "bottom"]}>
      <Text style={styles.title}>Practice</Text>
      <Text style={styles.sub}>
        Putts are counted on your PC from the Tapo stream (same logic as{" "}
        <Text style={styles.mono}>track_putts.py</Text>). Run{" "}
        <Text style={styles.mono}>putting_ws_server.py</Text> on the machine that
        sees the camera.
      </Text>

      <Text style={styles.label}>Tracker WebSocket URL</Text>
      <TextInput
        style={styles.input}
        value={urlInput}
        onChangeText={setUrlInput}
        placeholder={DEFAULT_WS_URL}
        placeholderTextColor="rgba(244,241,232,0.4)"
        autoCapitalize="none"
        autoCorrect={false}
      />

      <View style={styles.row}>
        <Pressable
          style={[styles.secondaryBtn, connecting && styles.btnDisabled]}
          onPress={connect}
          disabled={connecting}
        >
          {connecting ? (
            <ActivityIndicator color={cream} />
          ) : (
            <Text style={styles.secondaryBtnText}>Connect</Text>
          )}
        </Pressable>
        <Pressable style={styles.secondaryBtn} onPress={disconnect}>
          <Text style={styles.secondaryBtnText}>Disconnect</Text>
        </Pressable>
      </View>

      <Text style={styles.status}>
        {connected ? "● Live" : "○ Offline"}{" "}
        {statusLine ? ` — ${statusLine}` : ""}
      </Text>
      {practiceMessage ? (
        <Text style={styles.practiceMessage}>{practiceMessage}</Text>
      ) : null}

      <Text style={styles.section}>Live (from camera)</Text>
      <View style={styles.statsRow}>
        <Stat label="Attempts" value={display.attempts} />
        <Stat label="Made" value={display.made} />
        <Stat label="Missed" value={display.missed} />
      </View>
      <Text style={styles.seqLabel}>Sequence</Text>
      <Text style={styles.seqValue}>{seqPreview}</Text>

      <View style={styles.sessionRow}>
        <Pressable
          style={[
            styles.primaryBtn,
            (!connected || sessionActive) && styles.btnDisabled,
          ]}
          onPress={startSession}
          disabled={!connected || sessionActive}
        >
          <Text style={styles.primaryBtnText}>Start session</Text>
        </Pressable>
        <Pressable
          style={[
            styles.endBtn,
            (!sessionActive || !live) && styles.btnDisabled,
          ]}
          onPress={endSession}
          disabled={!sessionActive || !live}
        >
          <Text style={styles.endBtnText}>End & save</Text>
        </Pressable>
      </View>

      {sessionActive ? (
        <Text style={styles.hint}>
          Session in progress — tap End & save to write this stretch to History (uses live
          counts from the camera). The Record tab can show the same live counts while you
          film on the phone if you stay connected here.
        </Text>
      ) : null}
    </SafeAreaView>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <View style={styles.stat}>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: green,
    paddingHorizontal: 20,
    paddingTop: 8,
  },
  center: {
    flex: 1,
    backgroundColor: green,
    justifyContent: "center",
    alignItems: "center",
  },
  title: {
    fontSize: 26,
    fontWeight: "700",
    color: cream,
    marginBottom: 6,
  },
  sub: {
    fontSize: 14,
    color: cream,
    opacity: 0.88,
    marginBottom: 16,
    lineHeight: 20,
  },
  mono: {
    fontFamily: "monospace",
    fontSize: 12,
    color: madeGold,
  },
  label: {
    color: cream,
    fontSize: 13,
    marginBottom: 6,
    opacity: 0.9,
  },
  input: {
    backgroundColor: "rgba(0,0,0,0.25)",
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: cream,
    fontSize: 15,
    marginBottom: 12,
  },
  row: {
    flexDirection: "row",
    gap: 10,
    marginBottom: 8,
  },
  sessionRow: {
    flexDirection: "row",
    gap: 10,
    marginTop: 20,
    marginBottom: 12,
  },
  secondaryBtn: {
    flex: 1,
    borderWidth: 1,
    borderColor: cream,
    borderRadius: 12,
    paddingVertical: 12,
    alignItems: "center",
    justifyContent: "center",
    minHeight: 48,
  },
  secondaryBtnText: {
    color: cream,
    fontSize: 15,
    fontWeight: "600",
  },
  primaryBtn: {
    flex: 1,
    backgroundColor: cream,
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
  },
  primaryBtnText: {
    color: green,
    fontSize: 15,
    fontWeight: "700",
  },
  endBtn: {
    flex: 1,
    backgroundColor: madeGold,
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
  },
  endBtnText: {
    color: "#1a1a1a",
    fontSize: 15,
    fontWeight: "700",
  },
  btnDisabled: {
    opacity: 0.45,
  },
  status: {
    color: cream,
    fontSize: 13,
    marginBottom: 6,
    opacity: 0.9,
  },
  practiceMessage: {
    color: madeGold,
    fontSize: 13,
    marginBottom: 10,
    lineHeight: 18,
  },
  section: {
    color: cream,
    fontSize: 16,
    fontWeight: "600",
    marginBottom: 8,
  },
  statsRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 10,
    marginBottom: 12,
  },
  stat: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.2)",
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
  },
  statValue: {
    fontSize: 24,
    fontWeight: "700",
    color: cream,
  },
  statLabel: {
    fontSize: 12,
    color: cream,
    opacity: 0.8,
    marginTop: 4,
  },
  seqLabel: {
    color: cream,
    opacity: 0.75,
    fontSize: 12,
    marginBottom: 4,
  },
  seqValue: {
    color: cream,
    fontSize: 14,
    lineHeight: 20,
  },
  hint: {
    color: cream,
    opacity: 0.7,
    fontSize: 12,
    lineHeight: 18,
  },
});
