import React, { createContext, useContext, useEffect, useState } from "react";

interface LoadContextValue { stretch: number; queueLength: number; }
const LoadContext = createContext<LoadContextValue>({ stretch: 100, queueLength: 0 });

export function LoadProvider({ children }: { children: React.ReactNode }) {
  const [queueLength, setQueueLength] = useState(0);
  const [stretch, setStretch] = useState(100);
  useEffect(() => {
    const poll = async () => {
      try {
        const r = await fetch("/api/limits");
        const d = await r.json();
        const q = d.queue_length ?? 0;
        setQueueLength(q);
        const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        if (reduced) { setStretch(100); return; }
        const mapped = Math.min(140, Math.max(70, 70 + q * 3.5));
        setStretch(mapped);
      } catch {}
    };
    poll();
    const id = setInterval(poll, 2000);
    return () => clearInterval(id);
  }, []);
  useEffect(() => {
    document.documentElement.style.setProperty("--stretch", String(stretch));
  }, [stretch]);
  return <LoadContext.Provider value={{ stretch, queueLength }}>{children}</LoadContext.Provider>;
}

export function useLoad() { return useContext(LoadContext); }
