"use client";

import { CHEAT_SHEET } from "@/lib/constants";

export function CheatSheet({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal
      style={overlay}
      onClick={onClose}
    >
      <div style={panel} onClick={(e) => e.stopPropagation()}>
        <div style={header}>
          <h2 style={{ margin: 0, fontSize: "1.25rem" }}>Annotation cheat sheet</h2>
          <button type="button" onClick={onClose} style={closeBtn}>
            ✕
          </button>
        </div>
        <pre style={pre}>{CHEAT_SHEET}</pre>
      </div>
    </div>
  );
}

const overlay: React.CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "rgba(0,0,0,0.75)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  zIndex: 1000,
  padding: 16,
};

const panel: React.CSSProperties = {
  background: "#1a1a1a",
  border: "1px solid #333",
  borderRadius: 12,
  maxWidth: 520,
  width: "100%",
  maxHeight: "85vh",
  overflow: "auto",
  padding: 20,
};

const header: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  marginBottom: 12,
};

const closeBtn: React.CSSProperties = {
  background: "transparent",
  border: "none",
  color: "#aaa",
  fontSize: "1.25rem",
};

const pre: React.CSSProperties = {
  whiteSpace: "pre-wrap",
  fontSize: "0.85rem",
  lineHeight: 1.5,
  color: "#ccc",
  margin: 0,
  fontFamily: "ui-monospace, monospace",
};
