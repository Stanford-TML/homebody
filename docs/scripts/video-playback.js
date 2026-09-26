// Give visible videos and the next selected video a head start without loading
// every demo or restarting media that the browser has already buffered.
export function initializeVideoPlayback() {
  const states = new Map();
  let frame = 0;

  function eligible(video) {
    return !video.hidden && video.dataset.playbackDisabled !== "true" &&
      !video.closest('[hidden], [inert], details:not([open])') &&
      video.getClientRects().length > 0;
  }

  function prepare(video) {
    if (!eligible(video)) return;
    if (video.dataset.poster) {
      video.poster = video.dataset.poster;
      delete video.dataset.poster;
    }
    video.preload = "auto";
    // In particular, do not call load() again on the already-loading teaser.
    if (video.networkState === HTMLMediaElement.NETWORK_EMPTY) video.load();
  }

  function updatePlayback(video) {
    const state = states.get(video);
    const shouldPlay = state.visible && !document.hidden &&
      !state.userPaused && eligible(video);
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
    for (const [video, state] of states) {
      const priority = video.hasAttribute("data-preload-priority") && video.currentTime === 0;
      if (!document.hidden && eligible(video) && (visible.has(video) || video === next || priority)) {
        prepare(video);
      } else {
        // Preserve buffered media and the playhead when a carousel selection changes.
        video.preload = "none";
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

  document.querySelectorAll("video").forEach(video => {
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
    if (holdSeconds > 0) video.addEventListener("ended", () => {
      state.holding = true;
      updatePlayback(video);
    });
    if (video.hasAttribute("data-controls-on-demand")) {
      video.controls = false;
      video.addEventListener("click", () => { video.controls = true; });
      video.addEventListener("keydown", event => {
        if (!video.controls && (event.key === "Enter" || event.key === " ")) {
          event.preventDefault();
          video.controls = true;
        } else if (event.key === "Escape") video.controls = false;
      });
      video.addEventListener("blur", () => { video.controls = false; });
    }
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
  document.addEventListener("visibilitychange", () => {
    states.forEach((state, video) => updatePlayback(video));
    schedule();
  });
  updatePreloads();
}
