/** Turn Supabase / fetch errors into readable UI strings. */

export function formatError(err: unknown): string {
  if (err instanceof Error && err.message) return err.message;
  if (err && typeof err === "object") {
    const o = err as Record<string, unknown>;
    if (typeof o.message === "string" && o.message) {
      const parts = [o.message];
      if (typeof o.code === "string") parts.push(`(code ${o.code})`);
      if (typeof o.details === "string") parts.push(o.details);
      if (typeof o.hint === "string") parts.push(o.hint);
      return parts.join(" ");
    }
    try {
      return JSON.stringify(err);
    } catch {
      return String(err);
    }
  }
  return String(err);
}

export function isSchemaMigrationError(msg: string): boolean {
  const lower = msg.toLowerCase();
  return (
    lower.includes("expected_filename") ||
    lower.includes("storage_path") ||
    lower.includes("column") ||
    lower.includes("schema cache")
  );
}
