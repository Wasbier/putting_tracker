import * as FileSystem from "expo-file-system";

const DIR = "recordings";

/** Join documentDirectory (may or may not end with /) with a subfolder. */
function docPath(root: string, segment: string): string {
  const base = root.replace(/\/+$/, "");
  return `${base}/${segment.replace(/^\//, "")}`;
}

export async function ensureRecordingsDir(): Promise<string> {
  const root = FileSystem.documentDirectory;
  if (!root) throw new Error("documentDirectory unavailable");
  const path = docPath(root, DIR);
  const info = await FileSystem.getInfoAsync(path);
  if (!info.exists) {
    await FileSystem.makeDirectoryAsync(path, { intermediates: true });
  }
  return path;
}

export interface RecordingFile {
  uri: string;
  name: string;
  modified: number;
}

export async function listRecordings(): Promise<RecordingFile[]> {
  const dir = await ensureRecordingsDir();
  const entries = await FileSystem.readDirectoryAsync(dir);
  const out: RecordingFile[] = [];
  for (const name of entries) {
    if (!name.toLowerCase().endsWith(".mp4")) continue;
    const uri = docPath(dir, name);
    const info = await FileSystem.getInfoAsync(uri);
    out.push({
      uri,
      name,
      modified: info.modificationTime ?? 0,
    });
  }
  out.sort((a, b) => b.modified - a.modified);
  return out;
}

export async function moveRecordingFromCache(cacheUri: string): Promise<string> {
  const dir = await ensureRecordingsDir();
  const name = `putting_${Date.now()}.mp4`;
  const dest = docPath(dir, name);
  await FileSystem.copyAsync({ from: cacheUri, to: dest });
  try {
    await FileSystem.deleteAsync(cacheUri, { idempotent: true });
  } catch {
    /* ignore */
  }
  return dest;
}
