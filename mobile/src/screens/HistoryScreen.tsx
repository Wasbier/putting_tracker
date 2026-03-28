import { useFocusEffect } from "@react-navigation/native";
import React, { useCallback, useState } from "react";
import {
  FlatList,
  RefreshControl,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { loadCompletedSessions } from "../storage";
import type { Session } from "../types";
import { summarizeSession } from "../types";

const green = "#0d4f2b";
const cream = "#f4f1e8";

export default function HistoryScreen() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    try {
      const data = await loadCompletedSessions();
      setSessions(data);
    } catch {
      setSessions([]);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      load();
    }, [load]),
  );

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  }, [load]);

  return (
    <SafeAreaView style={styles.container} edges={["top", "bottom"]}>
      <Text style={styles.title}>History</Text>
      <Text style={styles.sub}>Saved sessions on this device</Text>
      <FlatList
        style={styles.list}
        data={sessions}
        keyExtractor={(item) => item.id}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={cream} />
        }
        ListEmptyComponent={
          <Text style={styles.empty}>No completed sessions yet.</Text>
        }
        contentContainerStyle={sessions.length === 0 ? styles.emptyList : undefined}
        renderItem={({ item }) => <SessionRow session={item} />}
      />
    </SafeAreaView>
  );
}

function SessionRow({ session }: { session: Session }) {
  const { attempts, made, missed } = summarizeSession(session);
  const end = session.endedAt ?? session.startedAt;
  const date = new Date(end);
  const label = date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });

  return (
    <View style={styles.row}>
      <Text style={styles.rowDate}>{label}</Text>
      <Text style={styles.rowStats}>
        {attempts} att · {made} made · {missed} miss
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: green,
    paddingHorizontal: 20,
  },
  title: {
    fontSize: 26,
    fontWeight: "700",
    color: cream,
    marginBottom: 4,
  },
  sub: {
    fontSize: 14,
    color: cream,
    opacity: 0.8,
    marginBottom: 16,
  },
  row: {
    backgroundColor: "rgba(0,0,0,0.2)",
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
  },
  rowDate: {
    color: cream,
    fontSize: 16,
    fontWeight: "600",
  },
  rowStats: {
    color: cream,
    opacity: 0.85,
    marginTop: 4,
    fontSize: 14,
  },
  empty: {
    color: cream,
    opacity: 0.75,
    fontSize: 16,
    textAlign: "center",
    marginTop: 40,
  },
  emptyList: {
    flexGrow: 1,
  },
  list: {
    flex: 1,
  },
});
