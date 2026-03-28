import React, { useCallback, useEffect, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import {
  appendCompletedSession,
  endSession,
  loadDraftSession,
  newSession,
  saveDraftSession,
  undoLastPutt,
  withPutt,
} from "../storage";
import type { PuttOutcome, Session } from "../types";
import { summarizeSession } from "../types";

const green = "#0d4f2b";
const cream = "#f4f1e8";
const missRed = "#c44c4c";
const madeGold = "#d4a84b";

export default function PracticeScreen() {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    (async () => {
      let draft: Session | null = null;
      try {
        draft = await loadDraftSession();
      } catch {
        draft = null;
      }
      if (alive) {
        setSession(draft);
        setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const startSession = useCallback(async () => {
    const s = newSession();
    setSession(s);
    await saveDraftSession(s);
  }, []);

  const logPutt = useCallback(
    async (outcome: PuttOutcome) => {
      if (!session) return;
      const next = withPutt(session, outcome);
      setSession(next);
      await saveDraftSession(next);
    },
    [session],
  );

  const undo = useCallback(async () => {
    if (!session || session.putts.length === 0) return;
    const next = undoLastPutt(session);
    setSession(next);
    await saveDraftSession(next);
  }, [session]);

  const finishSession = useCallback(async () => {
    if (!session) return;
    const done = endSession(session);
    await appendCompletedSession(done);
    await saveDraftSession(null);
    setSession(null);
  }, [session]);

  if (loading) {
    return (
      <SafeAreaView style={styles.center} edges={["top", "bottom"]}>
        <ActivityIndicator color={cream} size="large" />
      </SafeAreaView>
    );
  }

  if (!session) {
    return (
      <SafeAreaView style={styles.container} edges={["top", "bottom"]}>
        <Text style={styles.title}>Practice</Text>
        <Text style={styles.sub}>
          Log each putt as you go. Sessions are saved on this device.
        </Text>
        <Pressable style={styles.primaryBtn} onPress={startSession}>
          <Text style={styles.primaryBtnText}>Start session</Text>
        </Pressable>
      </SafeAreaView>
    );
  }

  const { attempts, made, missed } = summarizeSession(session);

  return (
    <SafeAreaView style={styles.container} edges={["top", "bottom"]}>
      <Text style={styles.title}>Current session</Text>
      <View style={styles.statsRow}>
        <Stat label="Attempts" value={attempts} />
        <Stat label="Made" value={made} />
        <Stat label="Missed" value={missed} />
      </View>

      <View style={styles.actions}>
        <Pressable
          style={[styles.bigBtn, styles.madeBtn]}
          onPress={() => logPutt("made")}
        >
          <Text style={styles.bigBtnText}>Made</Text>
        </Pressable>
        <Pressable
          style={[styles.bigBtn, styles.missBtn]}
          onPress={() => logPutt("miss")}
        >
          <Text style={styles.bigBtnText}>Miss</Text>
        </Pressable>
      </View>

      <View style={styles.row}>
        <Pressable style={styles.secondaryBtn} onPress={undo}>
          <Text style={styles.secondaryBtnText}>Undo last</Text>
        </Pressable>
        <Pressable style={styles.secondaryBtn} onPress={finishSession}>
          <Text style={styles.secondaryBtnText}>End & save</Text>
        </Pressable>
      </View>
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
    marginBottom: 8,
  },
  sub: {
    fontSize: 16,
    color: cream,
    opacity: 0.85,
    marginBottom: 28,
    lineHeight: 22,
  },
  statsRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 32,
    gap: 12,
  },
  stat: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.2)",
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: "center",
  },
  statValue: {
    fontSize: 28,
    fontWeight: "700",
    color: cream,
  },
  statLabel: {
    fontSize: 13,
    color: cream,
    opacity: 0.8,
    marginTop: 4,
  },
  actions: {
    gap: 14,
    marginBottom: 24,
  },
  bigBtn: {
    borderRadius: 16,
    paddingVertical: 22,
    alignItems: "center",
  },
  madeBtn: { backgroundColor: madeGold },
  missBtn: { backgroundColor: missRed },
  bigBtnText: {
    fontSize: 22,
    fontWeight: "700",
    color: "#1a1a1a",
  },
  row: {
    flexDirection: "row",
    gap: 12,
    marginTop: "auto",
    marginBottom: 16,
  },
  secondaryBtn: {
    flex: 1,
    borderWidth: 1,
    borderColor: cream,
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
  },
  secondaryBtnText: {
    color: cream,
    fontSize: 15,
    fontWeight: "600",
  },
  primaryBtn: {
    backgroundColor: cream,
    borderRadius: 14,
    paddingVertical: 16,
    alignItems: "center",
    alignSelf: "stretch",
  },
  primaryBtnText: {
    color: green,
    fontSize: 18,
    fontWeight: "700",
  },
});
