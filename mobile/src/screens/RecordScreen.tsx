import { CameraView, useCameraPermissions } from "expo-camera";
import React, { useCallback, useRef, useState } from "react";
import {
  ActivityIndicator,
  FlatList,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import {
  listRecordings,
  moveRecordingFromCache,
  type RecordingFile,
} from "../recordings";
import { useTrackerSocket } from "../TrackerSocketContext";

const green = "#0d4f2b";
const cream = "#f4f1e8";
const missRed = "#c44c4c";

export default function RecordScreen() {
  const { connected, live } = useTrackerSocket();
  const [permission, requestPermission] = useCameraPermissions();
  const [mode, setMode] = useState<"list" | "camera">("list");
  const [recordings, setRecordings] = useState<RecordingFile[]>([]);
  const [recording, setRecording] = useState(false);
  const [saving, setSaving] = useState(false);
  const camRef = useRef<CameraView>(null);
  const recordPromiseRef = useRef<Promise<{ uri: string } | undefined> | null>(
    null,
  );

  const refreshList = useCallback(async () => {
    try {
      const list = await listRecordings();
      setRecordings(list);
    } catch {
      setRecordings([]);
    }
  }, []);

  React.useEffect(() => {
    if (mode === "list") refreshList();
  }, [mode, refreshList]);

  const openCamera = useCallback(async () => {
    if (!permission?.granted) {
      const r = await requestPermission();
      if (!r.granted) return;
    }
    setMode("camera");
  }, [permission?.granted, requestPermission]);

  const toggleRecord = useCallback(async () => {
    const cam = camRef.current;
    if (!cam) return;

    if (!recording) {
      setRecording(true);
      try {
        recordPromiseRef.current = cam.recordAsync({
          maxDuration: 300,
        });
      } catch {
        setRecording(false);
        recordPromiseRef.current = null;
      }
      return;
    }

    try {
      cam.stopRecording();
    } catch {
      /* already stopped */
    }

    setRecording(false);
    setSaving(true);
    try {
      const vid = recordPromiseRef.current
        ? await recordPromiseRef.current
        : null;
      if (vid?.uri) {
        await moveRecordingFromCache(vid.uri);
        await refreshList();
      }
    } catch {
      /* record failed or was cancelled */
    } finally {
      recordPromiseRef.current = null;
      setSaving(false);
    }
  }, [recording, refreshList]);

  const closeCamera = useCallback(() => {
    if (recording || saving) return;
    setMode("list");
  }, [recording, saving]);

  if (mode === "camera") {
    return (
      <View style={styles.cameraWrap}>
        <CameraView
          ref={camRef}
          style={StyleSheet.absoluteFill}
          facing="back"
          mode="video"
          mute={false}
        />
        <SafeAreaView style={styles.cameraOverlay} edges={["top", "bottom"]}>
          <Pressable
            style={[styles.backBtn, (recording || saving) && styles.backBtnDisabled]}
            onPress={closeCamera}
            disabled={recording || saving}
          >
            <Text style={styles.backBtnText}>← Close</Text>
          </Pressable>
          {recording ? (
            <View style={styles.liveBanner}>
              {connected && live ? (
                <>
                  <Text style={styles.liveBannerTitle}>Live (Tapo on PC)</Text>
                  <View style={styles.liveBannerRow}>
                    <Text style={styles.liveStat}>
                      A {live.attempts}
                    </Text>
                    <Text style={styles.liveStat}>M {live.made}</Text>
                    <Text style={styles.liveStatMiss}>X {live.missed}</Text>
                  </View>
                </>
              ) : (
                <Text style={styles.liveBannerOffline}>
                  Connect on Practice (same Wi‑Fi) to see live putt counts from your
                  Tapo stream while you record here.
                </Text>
              )}
            </View>
          ) : null}
          <View style={styles.cameraFooter}>
            {saving ? (
              <ActivityIndicator color={cream} size="large" />
            ) : (
              <Pressable
                style={[
                  styles.recordCircle,
                  recording && styles.recordCircleActive,
                ]}
                onPress={toggleRecord}
              >
                <Text style={styles.recordHint}>
                  {recording ? "Stop" : "Record"}
                </Text>
              </Pressable>
            )}
            <Text style={styles.cameraHint}>
              Phone clip is separate from the Tapo feed. With Practice connected,
              counts above are from the PC tracker, not from this camera.
            </Text>
          </View>
        </SafeAreaView>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={["top", "bottom"]}>
      <Text style={styles.title}>Record</Text>
      <Text style={styles.sub}>
        Film on the phone for your own clips. If you connect on Practice first,
        live putt counts from the Tapo stream appear on screen while recording.
      </Text>

      <Pressable style={styles.primaryBtn} onPress={openCamera}>
        <Text style={styles.primaryBtnText}>Open camera</Text>
      </Pressable>

      <Text style={styles.listHeading}>Saved clips</Text>
      <FlatList
        style={styles.list}
        data={recordings}
        keyExtractor={(item) => item.uri}
        ListEmptyComponent={
          <Text style={styles.empty}>No recordings yet.</Text>
        }
        renderItem={({ item }) => (
          <View style={styles.clipRow}>
            <Text style={styles.clipName} numberOfLines={1}>
              {item.name}
            </Text>
          </View>
        )}
      />
    </SafeAreaView>
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
    opacity: 0.85,
    marginBottom: 20,
    lineHeight: 20,
  },
  primaryBtn: {
    backgroundColor: cream,
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: "center",
    marginBottom: 24,
  },
  primaryBtnText: {
    color: green,
    fontSize: 17,
    fontWeight: "700",
  },
  listHeading: {
    color: cream,
    fontSize: 16,
    fontWeight: "600",
    marginBottom: 10,
  },
  clipRow: {
    backgroundColor: "rgba(0,0,0,0.2)",
    borderRadius: 10,
    padding: 12,
    marginBottom: 8,
  },
  clipName: {
    color: cream,
    fontSize: 14,
  },
  empty: {
    color: cream,
    opacity: 0.7,
    fontSize: 15,
    marginTop: 8,
  },
  cameraWrap: {
    flex: 1,
    backgroundColor: "#000",
  },
  cameraOverlay: {
    flex: 1,
    justifyContent: "space-between",
  },
  liveBanner: {
    alignSelf: "center",
    marginTop: 8,
    backgroundColor: "rgba(0,0,0,0.65)",
    borderRadius: 12,
    paddingVertical: 10,
    paddingHorizontal: 16,
    maxWidth: "92%",
  },
  liveBannerTitle: {
    color: cream,
    fontSize: 11,
    fontWeight: "600",
    opacity: 0.85,
    marginBottom: 6,
    textAlign: "center",
  },
  liveBannerRow: {
    flexDirection: "row",
    justifyContent: "center",
    gap: 20,
  },
  liveStat: {
    color: cream,
    fontSize: 20,
    fontWeight: "700",
  },
  liveStatMiss: {
    color: missRed,
    fontSize: 20,
    fontWeight: "700",
  },
  liveBannerOffline: {
    color: cream,
    fontSize: 12,
    lineHeight: 17,
    textAlign: "center",
    opacity: 0.9,
  },
  backBtn: {
    alignSelf: "flex-start",
    marginLeft: 16,
    padding: 10,
  },
  backBtnDisabled: {
    opacity: 0.35,
  },
  backBtnText: {
    color: cream,
    fontSize: 17,
    fontWeight: "600",
  },
  cameraFooter: {
    alignItems: "center",
    paddingBottom: 24,
    gap: 12,
  },
  recordCircle: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: "rgba(255,255,255,0.35)",
    borderWidth: 4,
    borderColor: cream,
    justifyContent: "center",
    alignItems: "center",
  },
  recordCircleActive: {
    backgroundColor: missRed,
    borderColor: "#fff",
  },
  recordHint: {
    color: "#fff",
    fontWeight: "700",
    fontSize: 13,
  },
  cameraHint: {
    color: cream,
    opacity: 0.85,
    fontSize: 12,
    textAlign: "center",
    paddingHorizontal: 24,
    lineHeight: 18,
  },
  list: {
    flex: 1,
  },
});
