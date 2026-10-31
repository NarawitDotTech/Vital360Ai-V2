import type { ReactNode } from "react";

/** Re-mounts on every navigation, so each page fades/slides in (CSS only; disabled by prefers-reduced-motion). */
export default function Template({ children }: { children: ReactNode }) {
  return <div className="v-enter">{children}</div>;
}
