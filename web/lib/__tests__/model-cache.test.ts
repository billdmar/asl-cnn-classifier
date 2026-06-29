/**
 * Tests for the best-effort IndexedDB model cache.
 *
 * jsdom ships no IndexedDB, so we install a minimal in-memory fake on
 * `globalThis.indexedDB` for the DB-backed cases, and delete it entirely to
 * exercise the SSR/no-IDB guards. No external dependency is added.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  evictOtherVersions,
  getCachedModel,
  putCachedModel,
  resetModelCache,
} from "@/lib/model-cache";

// --- Minimal in-memory IndexedDB fake -------------------------------------
//
// Implements just the slice model-cache.ts uses: open (+upgradeneeded),
// transaction → objectStore → {get, put, delete, getAllKeys}, and
// deleteDatabase. Requests resolve on a microtask so the success handlers the
// code attaches after creating the request still fire.

type Listener = (() => void) | null;

class FakeRequest<T> {
  result!: T;
  error: Error | null = null;
  onsuccess: Listener = null;
  onerror: Listener = null;
  onupgradeneeded: Listener = null;
  onblocked: Listener = null;

  succeed(result: T): void {
    this.result = result;
    queueMicrotask(() => this.onsuccess?.());
  }

  fail(error: Error): void {
    this.error = error;
    queueMicrotask(() => this.onerror?.());
  }
}

class FakeObjectStore {
  constructor(private readonly map: Map<string, unknown>) {}

  get(key: string): FakeRequest<unknown> {
    const req = new FakeRequest<unknown>();
    req.succeed(this.map.has(key) ? this.map.get(key) : undefined);
    return req;
  }

  put(value: unknown, key: string): FakeRequest<void> {
    const req = new FakeRequest<void>();
    this.map.set(key, value);
    req.succeed(undefined);
    return req;
  }

  delete(key: string): FakeRequest<void> {
    const req = new FakeRequest<void>();
    this.map.delete(key);
    req.succeed(undefined);
    return req;
  }

  getAllKeys(): FakeRequest<IDBValidKey[]> {
    const req = new FakeRequest<IDBValidKey[]>();
    req.succeed([...this.map.keys()]);
    return req;
  }
}

class FakeTransaction {
  constructor(private readonly map: Map<string, unknown>) {}
  objectStore(): FakeObjectStore {
    return new FakeObjectStore(this.map);
  }
}

class FakeDb {
  objectStoreNames = {
    contains: () => this.created,
  };
  private created = false;
  constructor(private readonly map: Map<string, unknown>) {}
  createObjectStore(): void {
    this.created = true;
  }
  transaction(): FakeTransaction {
    return new FakeTransaction(this.map);
  }
  close(): void {}
}

class FakeIndexedDb {
  private store = new Map<string, unknown>();

  open(): FakeRequest<FakeDb> {
    const req = new FakeRequest<FakeDb>();
    const db = new FakeDb(this.store);
    req.result = db;
    // upgradeneeded first (store creation), then success.
    queueMicrotask(() => {
      req.onupgradeneeded?.();
      req.onsuccess?.();
    });
    return req;
  }

  deleteDatabase(): FakeRequest<void> {
    const req = new FakeRequest<void>();
    this.store.clear();
    req.succeed(undefined);
    return req;
  }
}

const URL_A = "/model/model.onnx";

beforeEach(() => {
  (globalThis as { indexedDB?: unknown }).indexedDB = new FakeIndexedDb();
});

afterEach(async () => {
  await resetModelCache();
  delete (globalThis as { indexedDB?: unknown }).indexedDB;
});

describe("model-cache (with IndexedDB)", () => {
  it("round-trips put → get for the same url + version", async () => {
    const bytes = new Uint8Array([1, 2, 3, 4]);
    await putCachedModel(URL_A, "v1", bytes);
    const got = await getCachedModel(URL_A, "v1");
    expect(got).toBeInstanceOf(Uint8Array);
    expect(Array.from(got!)).toEqual([1, 2, 3, 4]);
  });

  it("returns null for a missing key", async () => {
    expect(await getCachedModel(URL_A, "v1")).toBeNull();
  });

  it("misses on a version mismatch (put v1, get v2)", async () => {
    await putCachedModel(URL_A, "v1", new Uint8Array([9]));
    expect(await getCachedModel(URL_A, "v2")).toBeNull();
    // ...but the matching version still hits.
    expect(await getCachedModel(URL_A, "v1")).not.toBeNull();
  });

  it("evictOtherVersions removes other versions of the same url, keeps current", async () => {
    await putCachedModel(URL_A, "v1", new Uint8Array([1]));
    await putCachedModel(URL_A, "v2", new Uint8Array([2]));
    await putCachedModel("/other.onnx", "v1", new Uint8Array([3]));

    await evictOtherVersions(URL_A, "v2");

    expect(await getCachedModel(URL_A, "v1")).toBeNull(); // evicted
    expect(await getCachedModel(URL_A, "v2")).not.toBeNull(); // kept
    // A different url is untouched even if its version differs from v2.
    expect(await getCachedModel("/other.onnx", "v1")).not.toBeNull();
  });
});

describe("model-cache (SSR / no IndexedDB)", () => {
  beforeEach(() => {
    delete (globalThis as { indexedDB?: unknown }).indexedDB;
  });

  it("getCachedModel returns null and does not throw", async () => {
    expect(await getCachedModel(URL_A, "v1")).toBeNull();
  });

  it("putCachedModel / evictOtherVersions / resetModelCache are no-throw no-ops", async () => {
    await expect(
      putCachedModel(URL_A, "v1", new Uint8Array([1])),
    ).resolves.toBeUndefined();
    await expect(evictOtherVersions(URL_A, "v1")).resolves.toBeUndefined();
    await expect(resetModelCache()).resolves.toBeUndefined();
  });
});
