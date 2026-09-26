// Public playback IDs only. Asset IDs and credentials are not needed in the browser.
// Keep the native <video> UI and load the pinned HLS library only when needed.
const streams = new WeakMap();
let library;

function startupProfile() {
  const connection = navigator.connection;
  const desktop = matchMedia('(min-width: 900px)').matches;
  const slow = connection?.saveData || /^(slow-2g|2g|3g)$/.test(connection?.effectiveType || '');
  if (!desktop || slow) return { height: 480, budget: 900000, estimate: 1e6 };
  // Prefer a sharp first segment on laptops, including browsers without network hints.
  // This is a startup preference, not a minimum quality: ABR can still step down.
  return { height: 1080, budget: 4e6, estimate: 5e6 };
}

export function muxCanShareBandwidth(video) {
  const state = streams.get(video);
  const estimate = state?.hls?.bandwidthEstimate || (navigator.connection?.downlink || 0) * 1e6;
  return !navigator.connection?.saveData && estimate > 3e6;
}

export function suspendMux(video) {
  const state = streams.get(video);
  if (!state) return;
  state.wanted = false;
  if (state.loading) {
    state.hls?.stopLoad();
    state.loading = false;
  }
  // Native HLS owns its network scheduler; preload is only a hint on Safari.
  video.preload = 'none';
}

export function prepareMux(video) {
  if (!video.dataset.muxPlaybackId) return false;
  let state = streams.get(video);
  if (state) {
    state.wanted = true;
    if (state.ready && state.hls && !state.loading) {
      state.loading = true;
      state.hls.startLoad(video.currentTime, true);
    }
    return true;
  }
  state = { wanted: true, ready: false, loading: false, hls: null, fallback: false };
  streams.set(video, state);
  const startup = startupProfile();
  video.dataset.streamStartPreference = String(startup.height);
  const url = `https://stream.mux.com/${video.dataset.muxPlaybackId}.m3u8` +
    (startup.height === 1080 ? '?rendition_order=desc' : '');

  function fallback() {
    video.dataset.streamError = video.error?.message || 'HLS unavailable or fatal streaming error';
    if (state.fallback) return;
    state.fallback = true;
    state.hls?.destroy();
    state.hls = null;
    state.loading = false;
    const position = video.currentTime;
    video.dataset.streamEngine = 'mp4-fallback';
    video.src = video.dataset.videoFallback;
    video.preload = state.wanted ? 'auto' : 'none';
    video.addEventListener('loadedmetadata', () => {
      if (position > 0 && position < video.duration) video.currentTime = position;
    }, { once: true });
    video.load();
  }

  const nativeHls = !!video.canPlayType('application/vnd.apple.mpegurl');
  function useNative() {
    video.dataset.streamEngine = 'native-hls';
    video.addEventListener('error', fallback, { once: true });
    video.src = url;
    video.load();
  }
  // Chromium can advertise HLS without actually supporting a native stream.
  const safari = /Safari/.test(navigator.userAgent) && !/Chrome|Chromium|CriOS|Edg|OPR|FxiOS/.test(navigator.userAgent);
  if (nativeHls && safari) { useNative(); return true; }

  library ||= new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = new URL('./vendor/hls-1.7.3.light.min.js', import.meta.url).href;
    script.async = true;
    script.onload = () => resolve({ default: window.Hls });
    script.onerror = reject;
    document.head.append(script);
  });
  library.then(({ default: Hls }) => {
    if (!Hls.isSupported()) { if (nativeHls) useNative(); else fallback(); return; }
    const hls = state.hls = new Hls({
      autoStartLoad: false,
      capLevelToPlayerSize: true,
      abrEwmaDefaultEstimate: startup.estimate,
      maxBufferLength: 12,
      maxMaxBufferLength: 24,
      backBufferLength: 10
    });
    video.dataset.streamEngine = 'hls.js';
    hls.on(Hls.Events.MANIFEST_PARSED, () => {
      // Pick the initial rendition for the screen/network, then adapt subsequent segments.
      const affordable = hls.levels.map((level, index) => ({ level, index }))
        .filter(({ level }) => level.height <= startup.height && level.bitrate <= startup.budget)
        .sort((a, b) => b.level.bitrate - a.level.bitrate);
      hls.startLevel = affordable[0]?.index ?? 0;
      video.dataset.streamStartHeight = String(hls.levels[hls.startLevel]?.height || '');
      state.ready = true;
      if (state.wanted) {
        state.loading = true;
        hls.startLoad(video.currentTime, true);
      }
    });
    hls.on(Hls.Events.LEVEL_SWITCHED, (_, data) => {
      video.dataset.streamHeight = String(hls.levels[data.level]?.height || '');
    });
    hls.on(Hls.Events.ERROR, (_, data) => { if (data.fatal) fallback(); });
    hls.attachMedia(video);
    hls.loadSource(url);
  }).catch(fallback);
  return true;
}
