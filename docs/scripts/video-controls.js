// One controls policy for static videos and clips inserted by the skill figures.
// Playback, synchronization and looping stay with each video's existing controller.
export function initializeVideoControls() {
  const attached = new WeakSet();
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting || entry.intersectionRatio < 0.25) entry.target.controls = false;
    }
  }, { threshold: [0, 0.25] });

  function attach(video) {
    if (attached.has(video)) return;
    attached.add(video);
    video.controls = false;
    video.setAttribute('data-controls-on-demand', '');
    if (!video.hasAttribute('tabindex')) video.tabIndex = 0;
    video.title = 'Tap or press Enter to show video controls';
    function reveal() {
      const wasHidden = !video.controls;
      video.controls = true;
      if (wasHidden && video.paused && video.dataset.playbackDisabled !== 'true' &&
          !video.closest('[hidden], [inert], details:not([open])')) {
        video.play().catch(() => {});
      }
    }
    video.addEventListener('click', reveal);
    video.addEventListener('keydown', event => {
      if (!video.controls && (event.key === 'Enter' || event.key === ' ')) {
        event.preventDefault();
        reveal();
      } else if (event.key === 'Escape') video.controls = false;
    });
    video.addEventListener('blur', () => { video.controls = false; });
    video.addEventListener('playbackmodechange', () => {
      if (video.hidden || video.dataset.playbackDisabled === 'true') video.controls = false;
    });
    observer.observe(video);
  }
  function scan(root) {
    if (root.nodeType !== Node.ELEMENT_NODE && root !== document) return;
    if (root.matches?.('video')) attach(root);
    root.querySelectorAll('video').forEach(attach);
  }
  scan(document);
  new MutationObserver(records => {
    for (const record of records) {
      record.addedNodes.forEach(scan);
      for (const node of record.removedNodes) {
        if (node.nodeType !== Node.ELEMENT_NODE) continue;
        if (node.matches('video')) observer.unobserve(node);
        node.querySelectorAll('video').forEach(video => observer.unobserve(video));
      }
    }
  }).observe(document.body, { childList: true, subtree: true });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) document.querySelectorAll('video').forEach(video => { video.controls = false; });
  });
}
