/**
 * One Service + LoadGen for the whole page, driven by a requestAnimationFrame
 * loop. Components subscribe with `useTick()` and read the mutable sim
 * objects directly on render; renders are throttled to about 12 Hz.
 */

import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { LoadGen } from "../sim/loadgen";
import { formatSelfCheck, runSelfCheck } from "../sim/selfcheck";
import { Service } from "../sim/service";

export interface ServiceHandle {
  service: Service;
  gen: LoadGen;
  /** Render tick, bumped ~12 times per second while the loop runs. */
  tick: number;
  /** Force a re-render right away (after a control action). */
  refresh: () => void;
}

const Ctx = createContext<ServiceHandle | null>(null);

const RENDER_INTERVAL_MS = 80;
const MAX_DT_S = 0.1;

export function ServiceProvider({ children }: { children: ReactNode }) {
  const { service, gen } = useMemo(() => {
    const svc = new Service({ seed: 7 });
    svc.boot("v1");
    const g = new LoadGen(svc, { rps: 200, seed: 1 });
    return { service: svc, gen: g };
  }, []);
  const [tick, setTick] = useState(0);
  const lastFrame = useRef<number | null>(null);
  const lastRender = useRef(0);

  useEffect(() => {
    // Console self-check: parity with torch, rejection rules, zero-drop burst.
    const lines = runSelfCheck();
    const failed = lines.filter((l) => !l.pass).length;
    const banner = `ModelGate browser port self-check: ${lines.length - failed}/${lines.length} passed`;
    if (failed) console.warn(banner + "\n" + formatSelfCheck(lines));
    else console.info(banner + "\n" + formatSelfCheck(lines));
  }, []);

  useEffect(() => {
    let raf = 0;
    const frame = (now: number) => {
      const last = lastFrame.current ?? now;
      lastFrame.current = now;
      const dt = Math.min(MAX_DT_S, Math.max(0, (now - last) / 1000));
      const outcomes = gen.tick(dt);
      gen.pushSparkline(outcomes);
      if (now - lastRender.current >= RENDER_INTERVAL_MS) {
        lastRender.current = now;
        setTick((t) => t + 1);
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    const onVisibility = () => {
      // Skip the accumulated gap when the tab comes back.
      if (document.visibilityState === "visible") lastFrame.current = null;
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      cancelAnimationFrame(raf);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [gen]);

  const value = useMemo<ServiceHandle>(
    () => ({ service, gen, tick, refresh: () => setTick((t) => t + 1) }),
    [service, gen, tick],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useService(): ServiceHandle {
  const v = useContext(Ctx);
  if (!v) throw new Error("useService must be used inside ServiceProvider");
  return v;
}
