"use client";

import { Square, Volume2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { bcp47 } from "@/i18n";
import { useI18n } from "./I18nProvider";

function pickVoice(lang: string): SpeechSynthesisVoice | undefined {
  const voices = window.speechSynthesis.getVoices();
  const base = lang.split("-")[0].toLowerCase();
  return (
    voices.find((v) => v.lang.toLowerCase() === lang.toLowerCase()) ??
    voices.find((v) => v.lang.toLowerCase().startsWith(base))
  );
}

/**
 * "Read aloud" via the Web Speech API. Hidden when there is no speech support or no voice for the
 * active language (Thai voices are often missing), so the page never shows a button that stays silent.
 */
export function SpeakButton({ text, label }: { text: string; label?: string }) {
  const { locale, t } = useI18n();
  const [available, setAvailable] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const lang = bcp47[locale];

  useEffect(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    const synth = window.speechSynthesis;
    const check = () => setAvailable(!!pickVoice(lang));
    check();
    synth.addEventListener?.("voiceschanged", check);
    return () => {
      synth.removeEventListener?.("voiceschanged", check);
      synth.cancel();
    };
  }, [lang]);

  if (!available) return null;

  const toggle = () => {
    const synth = window.speechSynthesis;
    if (speaking) {
      synth.cancel();
      setSpeaking(false);
      return;
    }
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang;
    const v = pickVoice(lang);
    if (v) u.voice = v;
    u.rate = 0.9;
    u.onend = u.onerror = () => setSpeaking(false);
    setSpeaking(true);
    synth.cancel();
    synth.speak(u);
  };

  return (
    <Button
      type="button"
      variant="outline"
      size="lg"
      className="self-start"
      onClick={toggle}
      aria-pressed={speaking}
    >
      {speaking ? (
        <Square aria-hidden="true" />
      ) : (
        <Volume2 aria-hidden="true" />
      )}
      {speaking ? t.common.stopReading : (label ?? t.common.readAloud)}
    </Button>
  );
}
