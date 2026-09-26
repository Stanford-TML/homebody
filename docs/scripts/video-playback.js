import { initializeVideoControls } from "./video-controls.js";
import { prepareMux, suspendMux, muxCanShareBandwidth } from "./mux-video.js?v=3";

// Give visible videos and the next selected video a head start without loading
// every demo or restarting media that the browser has already buffered.
export function initializeVideoPlayback() {
  initializeVideoControls();
  const states = new Map();
  let frame = 0;

  function rendered(video) {
    return !video.hidden &&
      !video.closest('[hidden], [inert], details:not([open])') &&
      video.getClientRects().length > 0;
  }

  function eligible(video) {
    return rendered(video) && video.dataset.playbackDisabled !== "true";
  }

  function bufferedAhead(video) {
    for (let i = 0; i < video.buffered.length; i++) {
      if (video.buffered.start(i) <= video.currentTime && video.buffered.end(i) >= video.currentTime) {
        return (video.buffered.end(i) - video.currentTime) / (video.playbackRate || 1);
      }
    }
    return 0;
  }

  function prepare(video, preview = false) {
    if (!(preview ? rendered(video) : eligible(video))) return;
    if (video.dataset.poster) {
      video.poster = video.dataset.poster;
      delete video.dataset.poster;
    }
    video.preload = "auto";
    // In particular, do not call load() again on the already-loading teaser.
    if (!prepareMux(video) && video.networkState === HTMLMediaElement.NETWORK_EMPTY) video.load();
  }

  function updatePlayback(video) {
    const state = states.get(video);
    const shouldPlay = state.visible && !document.hidden &&
      !state.userPaused && eligible(video);
    if (video.hasAttribute("data-controls-on-demand") && (!state.visible || !eligible(video))) video.controls = false;
    if (video.ended && state.holdSeconds > 0) state.holding = true;
    if (state.holding) {
      if (shouldPlay && state.holdTimer === null) {
        state.holdTimer = setTimeout(() => {
          state.holdTimer = null;
          state.holding = false;
          video.currentTime = 0;
          updatePlayback(video);
        }, state.holdSeconds * 1000);
      } else if (!shouldPlay && state.holdTimer !== null) {
        clearTimeout(state.holdTimer);
        state.holdTimer = null;
      }
    } else if (shouldPlay) {
      prepare(video);
      video.play().catch(() => {});
    } else if (!video.paused) {
      state.automaticPause = true;
      video.pause();
    }
  }

  function updatePreloads() {
    frame = 0;
    const imagesPending = [...document.images].some(image => {
      if (image.complete || image.closest('[hidden], details:not([open])')) return false;
      const rect = image.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.top < innerHeight &&
        rect.right > 0 && rect.left < innerWidth;
    });
    let next = null;
    let nearest = Infinity;
    const visible = new Set();
    for (const [video] of states) {
      if (!eligible(video)) continue;
      const rect = video.getBoundingClientRect();
      if (rect.bottom > 0 && rect.top < innerHeight) visible.add(video);
      else if (rect.top >= innerHeight && rect.top < innerHeight * 2 && rect.top < nearest) {
        nearest = rect.top;
        next = video;
      }
    }
    // The adjacent task is a visible carousel preview, not a hidden section.
    // Give it a head start only after the selected task has eight seconds ready.
    const tasks = [...states.keys()].filter(video => video.matches(".task-recording"));
    const selected = tasks.find(video => eligible(video) && (visible.has(video) || video === next));
    const adjacent = !imagesPending && selected && bufferedAhead(selected) >= 8
      && (!selected.dataset.muxPlaybackId || muxCanShareBandwidth(selected))
      ? tasks.find(video => video !== selected && rendered(video) && bufferedAhead(video) < 8)
      : null;
    const activeMux = [...visible].find(video => video.dataset.muxPlaybackId);
    if (activeMux && (bufferedAhead(activeMux) < 8 || !muxCanShareBandwidth(activeMux))) next = null;
    for (const [video] of states) {
      const rect = video.getBoundingClientRect();
      const priority = video.hasAttribute("data-preload-priority") && video.currentTime === 0 &&
        rect.bottom > 0 && rect.top < innerHeight * 2;
      if (!document.hidden && eligible(video) && (visible.has(video) || priority || (!imagesPending && video === next))) {
        prepare(video);
      } else if (!document.hidden && video === adjacent) {
        prepare(video, true);
      } else {
        // Preserve buffered media and the playhead when a carousel selection changes.
        video.preload = "none";
        suspendMux(video);
      }
      updatePlayback(video);
    }
  }

  function schedule() {
    if (!frame) frame = requestAnimationFrame(updatePreloads);
  }

  const observer = "IntersectionObserver" in window ? new IntersectionObserver(entries => {
    for (const entry of entries) {
      const video = entry.target;
      const state = states.get(video);
      const wasVisible = state.visible;
      state.visible = entry.isIntersecting && entry.intersectionRatio >= 0.25;
      if (!state.visible && video.hasAttribute("data-controls-on-demand")) video.controls = false;
      if (state.visible && !wasVisible) state.userPaused = false;
      updatePlayback(video);
    }
    schedule();
  }, { threshold: [0, 0.25] }) : null;

  document.querySelectorAll("video:not(.mf-loop-video)").forEach(video => {
    video.muted = true;
    video.defaultMuted = true;
    video.playsInline = true;
    const holdSeconds = Math.max(0, Number(video.dataset.loopHold) || 0);
    video.loop = holdSeconds === 0;
    const rate = Number(video.dataset.playbackRate);
    if (Number.isFinite(rate) && rate > 0) {
      video.defaultPlaybackRate = video.playbackRate = rate;
      video.addEventListener("loadedmetadata", () => { video.playbackRate = rate; });
    }
    const state = { visible: !observer, userPaused: false, automaticPause: false,
      holding: false, holdTimer: null, holdSeconds };
    states.set(video, state);
    if (video.matches(".task-recording")) {
      // Recheck when the active buffer fills, drains, or playback resumes.
      for (const event of ["progress", "loadeddata", "timeupdate", "waiting"]) {
        video.addEventListener(event, schedule);
      }
    }
    if (holdSeconds > 0) video.addEventListener("ended", () => {
      state.holding = true;
      updatePlayback(video);
    });
    video.addEventListener("seeking", () => {
      clearTimeout(state.holdTimer);
      state.holdTimer = null;
      state.holding = false;
    });
    video.addEventListener("canplay", () => updatePlayback(video));
    video.addEventListener("playbackmodechange", () => {
      updatePlayback(video);
      schedule();
    });
    video.addEventListener("pause", () => {
      if (state.automaticPause) state.automaticPause = false;
      else if (state.visible && !document.hidden && !video.ended && eligible(video)) state.userPaused = true;
      updatePlayback(video);
    });
    video.addEventListener("play", () => {
      state.userPaused = false;
      updatePlayback(video);
    });
    video.addEventListener("pointerdown", () => prepare(video), { once: true });
    video.addEventListener("focus", () => prepare(video));
    observer?.observe(video);
  });
  window.addEventListener("scroll", schedule, { passive: true });
  window.addEventListener("resize", schedule);
  document.addEventListener("toggle", schedule, true);
  // Let visible still images finish before speculative video downloads begin.
  document.addEventListener("load", schedule, true);
  document.addEventListener("error", schedule, true);
  document.addEventListener("homebody:images-ready", schedule);
  document.addEventListener("visibilitychange", () => {
    states.forEach((state, video) => updatePlayback(video));
    schedule();
  });
  updatePreloads();
}
