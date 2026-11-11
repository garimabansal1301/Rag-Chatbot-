// Hide image cards whose thumbnail fails to load, and the whole "Related images" section once it is empty.
// A capturing listener on document also covers messages that HTMX inserts later ("error" does not bubble).
document.addEventListener(
  "error",
  (event) => {
    const img = event.target;
    if (!(img instanceof HTMLImageElement)) return;
    const card = img.closest(".image-card");
    if (!card) return;
    const section = card.closest(".related-images");
    card.remove();
    if (section && !section.querySelector(".image-card")) section.remove();
  },
  true,
);
