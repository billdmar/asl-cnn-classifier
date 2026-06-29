/**
 * Best-effort IndexedDB cache for the ONNX model bytes.
 *
 * The exported MobileNetV2 model is ~9 MB and, before this cache, re-downloaded
 * on every visit. We store the raw model bytes keyed by `${url}@${version}` so a
 * returning visitor (or a soft reload) loads the model from disk instead of the
 * network. The version is the build SHA, so a new deploy naturally misses the
 * old key (and {@link evictOtherVersions} reclaims the orphaned space).
 *
 * Hard rule: caching is *best-effort*. Every operation degrades to a no-op /
 * `null` on any failure (quota, private-mode, blocked DB, missing API) so it can
 * never break inference — the caller always falls back to a normal fetch.
 *
 * SSR/static-export safe: every export guards on `typeof indexedDB` first, so it
 * is inert at build time and on the server.
 */

const DB_NAME = "asl-model-cache";
const STORE_NAME = "models";

/** Build the composite key. Same url + different version ⇒ different entry. */
function cacheKey(url: string, version: string): string {
  return `${url}@${version}`;
}

/** True only in a browser context where IndexedDB is actually usable. */
function hasIndexedDb(): boolean {
  return typeof indexedDB !== "undefined";
}

/**
 * Open (creating/upgrading as needed) the cache DB. Promisified; rejects on the
 * usual IndexedDB error/blocked paths so callers can swallow uniformly.
 */
function openDb(): Promise<IDBDatabase> {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("indexedDB open failed"));
    request.onblocked = () => reject(new Error("indexedDB open blocked"));
  });
}

/** Promisify a single-shot IDBRequest. */
function awaitRequest<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("indexedDB request failed"));
  });
}

/**
 * Return the cached model bytes for `url`+`version`, or `null` if absent /
 * unavailable / on any error. Never throws.
 */
export async function getCachedModel(
  url: string,
  version: string,
): Promise<Uint8Array | null> {
  if (!hasIndexedDb()) return null;
  try {
    const db = await openDb();
    try {
      const tx = db.transaction(STORE_NAME, "readonly");
      const stored = await awaitRequest<unknown>(
        tx.objectStore(STORE_NAME).get(cacheKey(url, version)),
      );
      if (stored instanceof Uint8Array) return stored;
      if (stored instanceof ArrayBuffer) return new Uint8Array(stored);
      return null;
    } finally {
      db.close();
    }
  } catch {
    return null;
  }
}

/**
 * Store the model bytes for `url`+`version`. Best-effort: swallows quota and any
 * other error so a failed cache write never surfaces to the caller.
 */
export async function putCachedModel(
  url: string,
  version: string,
  bytes: Uint8Array,
): Promise<void> {
  if (!hasIndexedDb()) return;
  try {
    const db = await openDb();
    try {
      const tx = db.transaction(STORE_NAME, "readwrite");
      await awaitRequest(tx.objectStore(STORE_NAME).put(bytes, cacheKey(url, version)));
    } finally {
      db.close();
    }
  } catch {
    /* best-effort: quota / blocked / private-mode — ignore */
  }
}

/**
 * Delete any cached entries for `url` whose version differs from `version`.
 * Called after a successful put so a new deploy reclaims the old model's space.
 * Best-effort; never throws.
 */
export async function evictOtherVersions(url: string, version: string): Promise<void> {
  if (!hasIndexedDb()) return;
  const keep = cacheKey(url, version);
  const prefix = `${url}@`;
  try {
    const db = await openDb();
    try {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const keys = await awaitRequest<IDBValidKey[]>(store.getAllKeys());
      for (const key of keys) {
        if (typeof key === "string" && key.startsWith(prefix) && key !== keep) {
          store.delete(key);
        }
      }
    } finally {
      db.close();
    }
  } catch {
    /* best-effort */
  }
}

/**
 * Delete the entire cache DB (test hook + future "clear cache" affordance).
 * Best-effort; never throws.
 */
export async function resetModelCache(): Promise<void> {
  if (!hasIndexedDb()) return;
  try {
    await new Promise<void>((resolve) => {
      const request = indexedDB.deleteDatabase(DB_NAME);
      // Resolve on every terminal outcome — deletion is best-effort.
      request.onsuccess = () => resolve();
      request.onerror = () => resolve();
      request.onblocked = () => resolve();
    });
  } catch {
    /* best-effort */
  }
}
