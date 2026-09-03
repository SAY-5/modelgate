import { useReducedMotion } from "framer-motion";
import { useEffect, useState } from "react";

/**
 * Animate a number from 0 to `target` over `duration` ms on a rAF clock
 * (performance.now, never Date.now). Reduced motion resolves immediately.
 */
export function useCountUp(target: number, duration = 1600, delay = 0, active = true): number {
  const reduce = useReducedMotion();
  const [value, setValue] = useState(0);

  useEffect(() => {
    if (!active) return;
    if (reduce || duration <= 0) {
      setValue(target);
      return;
    }
    let raf = 0;
    let start: number | null = null;
    const step = (now: number) => {
      if (start === null) start = now + delay;
      const t = Math.min(1, Math.max(0, (now - start) / duration));
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(Math.round(target * eased));
      if (t < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, duration, delay, reduce, active]);

  return value;
}
