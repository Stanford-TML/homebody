// Fetch one matched pair at a time; the next pair follows after it is decoded.
export function initializeGalleryImages() {
  const track = document.querySelector("#comparison-track");
  if (!track) return;
  const slides = [...track.querySelectorAll(".comparison-slide")];
  const pending = new Map();
  let nearby = false;
  let current = 0;

  function loadPair(index, priority) {
    const slide = slides[index];
    if (!slide) return Promise.resolve();
    if (pending.has(index)) return pending.get(index);
    const promise = Promise.all([...slide.querySelectorAll("picture img")].map(image => {
      image.fetchPriority = priority;
      image.loading = "eager";
      const source = image.parentElement.querySelector("source");
      if (source?.dataset.gallerySrcset) {
        source.srcset = source.dataset.gallerySrcset;
        delete source.dataset.gallerySrcset;
      }
      image.src = image.dataset.gallerySrc;
      delete image.dataset.gallerySrc;
      return image.decode().catch(() => {});
    }));
    pending.set(index, promise);
    return promise;
  }

  function update(index = current) {
    current = index;
    if (!nearby || document.hidden || track.closest("[hidden]")) return;
    loadPair(index, "high").then(() => {
      document.dispatchEvent(new Event("homebody:images-ready"));
      if (nearby && current === index && !document.hidden && !track.closest("[hidden]")) {
        loadPair(index + 1, "low");
      }
    });
  }

  track.addEventListener("gallerychange", event => update(event.detail));
  if ("IntersectionObserver" in window) {
    new IntersectionObserver(entries => {
      nearby = entries[0].isIntersecting;
      update();
    }, { rootMargin: "600px 0px" }).observe(track);
  } else {
    nearby = true;
    update();
  }
  document.addEventListener("visibilitychange", () => update());
}
