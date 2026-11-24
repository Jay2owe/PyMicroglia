"use strict";
// Tells the local server this page is still open; the browser launcher stops after 90 s without it.
(() => {
  const beat = () => fetch("/api/heartbeat", {cache: "no-store"}).catch(() => {});
  beat();
  setInterval(beat, 15000);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) beat(); });
})();
