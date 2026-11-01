import type { ReactNode } from "react";

export function PageTitle({ children }: { children: ReactNode }) {
  return (
    <h1 className="text-2xl leading-normal font-semibold text-balance sm:text-3xl">
      {children}
    </h1>
  );
}
