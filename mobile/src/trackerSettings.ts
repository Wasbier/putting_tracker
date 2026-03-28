import AsyncStorage from "@react-native-async-storage/async-storage";

const KEY = "@putting_tracker/ws_server_url_v1";

/** Default placeholder; user replaces with PC LAN IP running putting_ws_server.py */
export const DEFAULT_WS_URL = "ws://192.168.1.100:8765";

export async function loadTrackerWsUrl(): Promise<string> {
  try {
    const v = await AsyncStorage.getItem(KEY);
    return v && v.length > 0 ? v : DEFAULT_WS_URL;
  } catch {
    return DEFAULT_WS_URL;
  }
}

export async function saveTrackerWsUrl(url: string): Promise<void> {
  await AsyncStorage.setItem(KEY, url.trim());
}
